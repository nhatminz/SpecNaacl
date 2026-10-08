#!/usr/bin/env python3
"""Compare only old/finite OPD samplers; optionally measure paired real rollouts.

Run on B200 with --require-b200. No default is changed by this benchmark.
All device synchronization/timing is outside the sampler calls under test.

Example (from SpecNaacl, using the same trained checkpoint for both modes):
  python scripts/benchmark_opd_sampler.py --require-b200 --output sampler.json \
    --target-model /path/to/target --draft-checkpoint /path/to/draft \
    --dataset-path /path/to/train.parquet
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--require-b200', action='store_true')
    p.add_argument('--shapes', default='1x1x151936,8x8x151936,64x8x151936')
    p.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--top-p', type=float, default=.95, help='0 disables top-p')
    p.add_argument('--top-k', type=int, default=0, help='0 disables top-k')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target-model', default='')
    p.add_argument('--draft-checkpoint', default='')
    p.add_argument('--dataset-path', default='')
    p.add_argument('--target-adapter', default='')
    p.add_argument('--generation-iterations', type=int, default=2)
    p.add_argument('--generation-rounds', type=int, default=5)
    p.add_argument('--generation-warmup', type=int, default=1)
    p.add_argument('--generation-seeds', default='42,43')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--responses', type=int, default=8)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--max-prompt-length', type=int, default=256)
    p.add_argument('--verification-capacity', type=int, default=512)
    p.add_argument('--max-verification-num', type=int, default=160)
    p.add_argument('--max-draft-k', type=int, default=8)
    p.add_argument('--max-draft-length', type=int, default=5)
    p.add_argument('--min-draft-length', type=int, default=3)
    p.add_argument('--draft-length-c', type=float, default=.75)
    p.add_argument('--fast-lr', type=float, default=.01)
    p.add_argument('--update-stream', choices=[0, 1], type=int, default=1)
    p.add_argument('--attn-implementation', default='sdpa')
    a = p.parse_args()
    resources = [a.target_model, a.draft_checkpoint, a.dataset_path]
    if any(resources) and not all(resources):
        p.error('generation requires --target-model, --draft-checkpoint and --dataset-path')
    if min(a.iterations, a.rounds, a.generation_iterations, a.generation_rounds) < 1:
        p.error('iteration/round counts must be positive')
    if min(a.warmup, a.generation_warmup) < 0:
        p.error('warmup counts cannot be negative')
    if a.temperature <= 0 or not 0 <= a.top_p <= 1 or a.top_k < 0:
        p.error('temperature>0, 0<=top-p<=1, top-k>=0 required')
    a.parsed_shapes = [tuple(map(int, s.split('x'))) for s in a.shapes.split(',')]
    if any(len(s) != 3 or min(s) < 1 or (a.top_k and a.top_k > s[-1]) for s in a.parsed_shapes):
        p.error('shapes must be positive BxTxV with top-k<=V')
    a.seeds = [int(s) for s in a.generation_seeds.split(',')]
    return a


def load_old_sampler():
    path = ROOT / 'tests/references/opd_sampling_before_finite.py'
    spec = importlib.util.spec_from_file_location('opd_sampler_before_finite', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def microbenchmark(a, torch, old, optimized, dtype):
    rows = []
    for shape in a.parsed_shapes:
        gen = torch.Generator(device='cuda').manual_seed(a.seed)
        logits = torch.randn(shape, device='cuda', dtype=dtype, generator=gen)
        kwargs = dict(top_p=a.top_p or None, top_k=a.top_k or None,
                      temperature=a.temperature, return_probs=False)
        def call(mode):
            return (old.sampling(logits, **kwargs) if mode == 'strict'
                    else optimized.sampling(logits, mode='finite', **kwargs))
        # Probability/sort parity outside timing, before discarding [N,V].
        def capture(tokens, probs, metadata):
            return tuple(x.clone() for x in metadata) if metadata is not None else None
        parity_kwargs = dict(kwargs, return_probs=True, metadata_builder=capture)
        torch.manual_seed(a.seed)
        before = old.sampling(logits, **parity_kwargs)
        rng = torch.cuda.get_rng_state()
        torch.manual_seed(a.seed)
        after = optimized.sampling(logits, mode='finite', **parity_kwargs)
        parity = torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])
        parity = parity and torch.equal(rng, torch.cuda.get_rng_state())
        parity = parity and (before[2] is None and after[2] is None or
                            before[2] is not None and after[2] is not None and
                            all(torch.equal(x, y) for x, y in zip(before[2], after[2])))
        if not parity: raise AssertionError(f'sampler parity failed for {shape}; discard timing')
        del before, after
        for mode in ('strict', 'finite'):
            for _ in range(a.warmup): call(mode)
        times = {mode: [] for mode in ('strict', 'finite')}
        for repeat in range(a.rounds):
            order = ('strict', 'finite') if repeat % 2 == 0 else ('finite', 'strict')
            for mode in order:
                torch.manual_seed(a.seed + repeat)
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                began = time.perf_counter(); start.record()
                for _ in range(a.iterations): call(mode)
                end.record(); end.synchronize()
                times[mode].append(dict(wall_ms=(time.perf_counter()-began)*1000/a.iterations,
                                        cuda_ms=start.elapsed_time(end)/a.iterations))
        result = dict(shape=list(shape), parity=True, samples=times)
        for mode in times:
            result[mode] = {name: statistics.median(t[name] for t in times[mode]) for name in ('wall_ms', 'cuda_ms')}
        result['speedup'] = result['strict']['wall_ms']/result['finite']['wall_ms']
        rows.append(result)
    return rows


def generation_benchmark(a, torch, optimized, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from helper.fastgrpo_model import FastGRPOModel
    from helper.get_QAs import get_QAs_from_path
    from helper.specualtive_generate import speculative_generate
    from scripts.benchmark_opd_reflex import collator

    torch.manual_seed(a.seed)
    target = AutoModelForCausalLM.from_pretrained(a.target_model, torch_dtype=dtype,
        attn_implementation=a.attn_implementation, local_files_only=True).cuda().eval()
    config = deepcopy(target.config); config.num_hidden_layers=1; config.rope_scaling=None; config.torch_dtype=target.dtype
    model = FastGRPOModel(config, target).cuda().eval()
    checkpoint = Path(a.draft_checkpoint)
    if checkpoint.is_dir(): checkpoint=checkpoint/'draft.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['draft_model']
    projector = state.pop('opd_projector', None)
    model.draft_model.load_state_dict(state)
    if a.target_adapter:
        from peft import PeftModel
        model.target_model=PeftModel.from_pretrained(model.target_model, a.target_adapter).eval()
    model.enable_opd(8)
    if projector is not None: model.load_opd_projector(projector)
    tokenizer = AutoTokenizer.from_pretrained(a.target_model, padding_side='left', local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    dataset = get_QAs_from_path(a.dataset_path, 'train')
    needed = a.batch_size*a.generation_iterations
    if len(dataset)<needed: raise ValueError(f'need {needed} prompts')
    make_batch = collator(tokenizer, a.max_prompt_length)
    batches = []
    for offset in range(0, needed, a.batch_size):
        batch = make_batch(dataset[offset:offset+a.batch_size])
        if batch['input_ids'].shape[1]>=a.max_length: raise ValueError('prompt exhausts max-length')
        batches.append({k:batch[k].cuda() for k in ('input_ids', 'attention_mask')})
    counts = dict(target=0, draft=0)
    target_hook = target.model.layers[0].register_forward_pre_hook(lambda *_: counts.__setitem__('target', counts['target']+1))
    draft_hook = model.draft_model.register_forward_pre_hook(lambda *_: counts.__setitem__('draft', counts['draft']+1))
    kwargs = dict(method='opd_reflex', do_sample=True, repeated_generate_nums=a.responses,
        temperature=a.temperature, top_p=a.top_p or None, top_k=a.top_k or None,
        max_length=a.max_length, verification_capacity=a.verification_capacity,
        max_verification_num=a.max_verification_num, max_draft_k=a.max_draft_k,
        max_draft_token_length=a.max_draft_length, min_draft_token_length=a.min_draft_length,
        draft_token_length_c=a.draft_length_c, opd_fast_lr=a.fast_lr,
        opd_update_stream=bool(a.update_stream), statistical_time=False)
    previous_mode = optimized.SAMPLER_MODE
    def run(mode, seed, batch):
        optimized.SAMPLER_MODE=mode
        torch.manual_seed(seed)
        before=counts.copy()
        with torch.inference_mode():
            out=speculative_generate(model, batch['input_ids'], batch['attention_mask'], tokenizer, **kwargs)
        return out, {k:counts[k]-before[k] for k in counts}
    schedule = [(seed+i, i, batch) for seed in a.seeds for i, batch in enumerate(batches)]
    records = {mode: [] for mode in ('strict', 'finite')}
    expected = {}
    try:
        # Warm the entire schedule in both modes, including every dynamic shape.
        for mode in records:
            for _ in range(a.generation_warmup):
                for seed, _, batch in schedule: run(mode, seed, batch)
        for repeat in range(a.generation_rounds):
            order = ('strict', 'finite') if repeat % 2 == 0 else ('finite', 'strict')
            for mode in order:
                tokens=0; wall=0.; details=[]
                for seed, index, batch in schedule:
                    torch.cuda.synchronize(); began=time.perf_counter()
                    out, forwards=run(mode, seed, batch)
                    torch.cuda.synchronize(); elapsed=time.perf_counter()-began
                    rng=torch.cuda.get_rng_state()
                    signature={k:out[k] for k in ('generated_token_ids', 'total_acc_length',
                        'total_decoded_token_num', 'verification_batches', 'total_accepted_draft_tokens')}
                    key=(seed, index)
                    if key not in expected: expected[key]=(signature, rng, forwards)
                    else:
                        original, original_rng, original_forwards=expected[key]
                        if signature != original or not torch.equal(rng, original_rng) or forwards != original_forwards:
                            raise AssertionError('generation token/RNG/acceptance/forward parity failed; discard timing')
                    generated=sum(out['response_generated_tokens']); tokens+=generated; wall+=elapsed
                    details.append(dict(seed=seed, batch=index, generated_tokens=generated,
                        wall_s=elapsed, forwards=forwards,
                        tokens_sha256=hashlib.sha256(json.dumps(out['generated_token_ids']).encode()).hexdigest()))
                records[mode].append(dict(generated_tokens=tokens, wall_s=wall, tokens_per_s=tokens/wall, rollouts=details))
    finally:
        optimized.SAMPLER_MODE=previous_mode
        target_hook.remove(); draft_hook.remove()
    result=dict(parity=True, samples=records)
    for mode in records:
        result[mode]=dict(tokens_per_s=statistics.median(r['tokens_per_s'] for r in records[mode]))
    result['speedup']=result['finite']['tokens_per_s']/result['strict']['tokens_per_s']
    return result


def main():
    a = arguments()
    import torch
    from helper import opd_sampling
    if not torch.cuda.is_available(): raise RuntimeError('CUDA benchmark required')
    gpu = torch.cuda.get_device_name()
    b200 = 'B200' in gpu.upper()
    if a.require_b200 and not b200: raise RuntimeError(f'B200 required; found {gpu}')
    dtype = {'bf16':torch.bfloat16, 'fp16':torch.float16, 'fp32':torch.float32}[a.dtype]
    old = load_old_sampler()
    output = Path(a.output)
    if output.exists(): raise FileExistsError(f'{output}; choose a new report path')
    report = dict(gpu=gpu, b200=b200, torch=torch.__version__, cuda=torch.version.cuda,
        config=vars(a), default_mode=opd_sampling.SAMPLER_MODE,
        sampler=microbenchmark(a, torch, old, opd_sampling, dtype))
    if a.target_model:
        report['generation']=generation_benchmark(a, torch, opd_sampling, dtype)
    report['b200_timings_favor_finite'] = bool(b200 and a.warmup>0 and a.generation_warmup>0 and
        a.rounds>=3 and a.generation_rounds>=3 and
        all(r['speedup']>1 for r in report['sampler']) and
        report.get('generation', {}).get('parity', False) and report['generation']['speedup']>1)
    report['default_changed'] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    for row in report['sampler']:
        print(f"Sampler {row['shape']}: {row['strict']['wall_ms']:.4f} -> {row['finite']['wall_ms']:.4f} ms; parity PASS")
    if 'generation' in report:
        row=report['generation']
        print(f"Generation: {row['strict']['tokens_per_s']:.2f} -> {row['finite']['tokens_per_s']:.2f} tokens/s; parity PASS")
    print(f'{gpu}; default={opd_sampling.SAMPLER_MODE}; report={output}')


if __name__ == '__main__':
    main()
