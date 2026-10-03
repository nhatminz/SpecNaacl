#!/usr/bin/env python3
"""Real frozen-model generation wall clock/tokens/s on the production rollout.

No optimizer/backward/checkpoint mutation. Loads existing target/draft/mapping,
uses production dataset loader and exact training prompt collator. Warmup/JIT/
load excluded. Prints JSON; redirect stdout to a NEW benchmark report file.
"""
import argparse
import ast
import json
from pathlib import Path
import statistics
import sys
import time
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('target-model', 'draft-config', 'draft-checkpoint', 'vocab-mapping', 'dataset-path'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--target-adapter', default='')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--responses', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--max-prompt-length', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=.95)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--verification-capacity', type=int, default=512)
    parser.add_argument('--max-verification-num', type=int, default=512)
    parser.add_argument('--max-draft-k', type=int, default=8)
    parser.add_argument('--max-draft-length', type=int, default=5)
    parser.add_argument('--feature-dim', type=int, default=8)
    parser.add_argument('--reflex-lr', type=float, default=.05)
    parser.add_argument('--reflex-weight-decay', type=float, default=0.)
    parser.add_argument('--reflex-proposal-strategy', choices=['fused', 'sort', 'hybrid', 'torch'], default='fused')
    parser.add_argument('--reflex-correction-strategy', choices=['serial', 'parallel', 'tiled'], default='serial')
    parser.add_argument('--reflex-feedback-strategy', choices=['serial', 'parallel'], default='serial')
    parser.add_argument('--reflex-feature-strategy', choices=['auto', 'triton', 'torch'], default='auto')
    parser.add_argument('--reflex-update-stream', action='store_true')
    parser.add_argument('--kv-gather-strategy', choices=['stacked', 'per_layer'], default='stacked')
    parser.add_argument('--attn-implementation', default='eager')
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--modes', default='fastgrpo,torch-zero,torch-root,triton-root,torch-visited_path,triton-visited_path')
    parser.add_argument('--skip-zero-state-check', action='store_true')
    parser.add_argument('--profile-trace', default='', help='Optional Chrome/Perfetto trace of ONE extra rollout (not timed)')
    parser.add_argument('--component-timing', action='store_true',
                        help='Diagnostic target/draft phase timing; synchronizes and is not a throughput run')
    args = parser.parse_args(argv)
    for name in ('batch_size', 'responses', 'warmup', 'iterations', 'max_length', 'max_prompt_length',
                 'verification_capacity', 'max_verification_num', 'max_draft_k', 'max_draft_length', 'feature_dim'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    valid_modes = {'fastgrpo', 'torch-zero', 'triton-zero', 'torch-root', 'triton-root',
                   'torch-visited_path', 'triton-visited_path', 'triton-root-stream',
                   'triton-visited_path-stream'}
    if not set(args.modes.split(',')) <= valid_modes:
        parser.error('invalid --modes')
    return args


def production_collator(tokenizer, max_prompt_length):
    # Reuse ONLY the collator class; importing grpo_speculative starts training.
    path = ROOT / 'grpo_speculative.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TrainDataCollator')
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['TrainDataCollator'](tokenizer, max_prompt_length)


@torch.inference_mode()
def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Real rollout benchmark requires CUDA and existing production checkpoints; no synthetic speedup substituted.')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from helper.eagle3_specforge import Eagle3FastGRPOAdapter
    from helper.get_QAs import get_QAs_from_path
    from helper.specualtive_generate import speculative_generate
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    target = AutoModelForCausalLM.from_pretrained(args.target_model, dtype=torch.bfloat16,
                attn_implementation=args.attn_implementation, local_files_only=True)
    model = Eagle3FastGRPOAdapter(target, args.draft_config, args.draft_checkpoint,
                                  args.vocab_mapping, ttt_length=7).cuda().eval()
    if args.target_adapter:
        from peft import PeftModel
        model.target_model = PeftModel.from_pretrained(model.target_model, args.target_adapter).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, padding_side='left', local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if target.config.model_type == 'llama':
        tokenizer.pad_token, tokenizer.pad_token_id = '<|end_of_text|>', 128001
    rows = get_QAs_from_path(args.dataset_path, 'train')
    needed = args.batch_size * args.iterations
    if len(rows) < needed:
        raise ValueError(f'need at least {needed} dataset rows; got {len(rows)}')
    collator = production_collator(tokenizer, args.max_prompt_length)
    batches = [collator(rows[i:i + args.batch_size]) for i in range(0, needed, args.batch_size)]
    for batch in batches:
        if batch['input_ids'].shape[1] >= args.max_length:
            raise ValueError('prompt consumes max_length; choose a suitable prompt pool or increase total max_length')
        for key in ('input_ids', 'attention_mask'):
            batch[key] = batch[key].cuda()
    base = dict(do_sample=not args.greedy, repeated_generate_nums=args.responses,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k or None,
        verification_capacity=args.verification_capacity, max_verification_num=args.max_verification_num,
        max_draft_k=args.max_draft_k, max_draft_token_length=args.max_draft_length,
        max_length=args.max_length, statistical_time=args.component_timing, reflex_feature_dim=args.feature_dim,
        reflex_weight_decay=args.reflex_weight_decay, reflex_seed=args.seed,
        reflex_profile=False, reflex_diagnostics=False,
        reflex_proposal_strategy=args.reflex_proposal_strategy,
        reflex_correction_strategy=args.reflex_correction_strategy,
        reflex_feedback_strategy=args.reflex_feedback_strategy,
        reflex_feature_strategy=args.reflex_feature_strategy)
    base['kv_gather_strategy'] = args.kv_gather_strategy
    forwards = [0]
    # Count actual backbone forwards, NOT inferred speculative rounds.
    backbone = target.model
    hook = backbone.register_forward_pre_hook(lambda *unused: forwards.__setitem__(0, forwards[0] + 1))

    def run(mode, batch, seed, lr=None):
        if seed is not None:
            torch.manual_seed(seed)  # identical seed schedule; no model state updates
        stream_mode = mode.endswith('-stream')
        core_mode = mode[:-7] if stream_mode else mode
        backend, scope = ('torch', 'root') if core_mode == 'fastgrpo' else core_mode.split('-', 1)
        if scope == 'zero':
            scope, lr = 'root', 0.0  # isolate tensor-path engineering from learning
        before = forwards[0]
        output = speculative_generate(model, batch['input_ids'], batch['attention_mask'], tokenizer,
            reflex_mode='off' if mode == 'fastgrpo' else 'active', reflex_backend=backend,
            reflex_feedback_scope=scope, reflex_lr=args.reflex_lr if lr is None else lr,
            reflex_update_stream=stream_mode or args.reflex_update_stream, **base)
        return output, forwards[0] - before

    try:
        if not args.skip_zero_state_check:
            off, off_count = run('fastgrpo', batches[0], args.seed, lr=0)
            active, active_count = run('torch-root', batches[0], args.seed, lr=0)
            if off['generated_token_ids'] != active['generated_token_ids'] or off_count != active_count:
                raise AssertionError('zero-state Torch rollout differs from OFF; do not use timing results')
        reports = {}
        for mode in args.modes.split(','):
            for i in range(args.warmup):
                run(mode, batches[i % len(batches)], args.seed + i)
            torch.cuda.synchronize()
            rows_out = []
            for i, batch in enumerate(batches):
                torch.manual_seed(args.seed + 100 + i)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                output, count = run(mode, batch, None)
                torch.cuda.synchronize()
                wall = time.perf_counter() - started
                tokens = sum(output['response_generated_tokens'])
                rows_out.append(dict(generation_wall_s=wall, generated_tokens=tokens, tokens_per_s=tokens / wall,
                    target_forwards=count, accepted_length_sum=output['total_acc_length'],
                    verification_rounds=output['total_decoded_token_num'],
                    target_phase_s=output['target_time_cost'] if args.component_timing else None,
                    draft_phase_s=output['draft_time_cost'] if args.component_timing else None,
                    committed_draft_forward_s=output['check_time_cost'] if args.component_timing else None,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            total_wall = sum(row['generation_wall_s'] for row in rows_out)
            rounds = sum(row['verification_rounds'] for row in rows_out)
            reports[mode] = dict(total_generation_wall_s=total_wall,
                tokens_per_s=sum(row['generated_tokens'] for row in rows_out) / total_wall,
                median_generation_wall_s=statistics.median(row['generation_wall_s'] for row in rows_out),
                weighted_aal=sum(row['accepted_length_sum'] for row in rows_out) / max(rounds, 1), rollouts=rows_out)
        if args.profile_trace:
            path = Path(args.profile_trace)
            if path.exists():
                raise FileExistsError('profile trace must be a new path')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as profiler:
                run(args.modes.split(',')[-1], batches[0], args.seed + 100)
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(path))
        # Guidance only: never silently changes a production backend/config.
        winner = max(reports, key=lambda mode: reports[mode]['tokens_per_s'])
        return dict(benchmark='real_frozen_production_rollout', gpu=torch.cuda.get_device_name(), gpu_count=1,
            config=vars(args), reports=reports, fastest_measured_mode=winner,
            note='Generation only, frozen persistent weights; not training throughput or task quality. *-zero controls isolate tensor-path engineering from acceptance learning. Different proposal trees need not reproduce identical tokens for a fixed seed.')
    finally:
        hook.remove()


if __name__ == '__main__':
    with redirect_stdout(sys.stderr):
        report = benchmark(parse_args())
    print(json.dumps(report, indent=2))
