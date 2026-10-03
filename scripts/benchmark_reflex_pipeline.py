#!/usr/bin/env python3
"""Component/whole verification-cycle benchmark, NOT model rollout speedup.

Production proposal/extraction/update APIs, real state transitions. CUDA events
and wall clock reported independently, with warmup/prepare outside component
timings. CPU mode is a tiny smoke check only. No profiler in production.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from helper.fast_lk_reflex import FastLKReflex
from helper.tree_verification import PackedTree, trace_verified_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--backend', choices=['torch', 'triton', 'auto'], default='auto')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--vocab', type=int, default=16000)
    parser.add_argument('--target-vocab', type=int, default=151936)
    parser.add_argument('--hidden', type=int, default=2048)
    parser.add_argument('--contexts', type=int, default=8)
    parser.add_argument('--feature-dim', type=int, default=8)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--feedback-scope', choices=['root', 'visited_path'], default='root')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    parser.add_argument('--greedy', action='store_true')
    args = parser.parse_args(argv)
    for field in ('batch', 'vocab', 'target_vocab', 'hidden', 'contexts', 'feature_dim', 'topk', 'depth', 'warmup', 'iterations'):
        if getattr(args, field) <= 0:
            parser.error(f'{field} must be positive')
    if args.topk > args.vocab or args.target_vocab < args.vocab:
        parser.error('topk must fit compact vocab; target vocab must cover compact vocab')
    return args


@torch.inference_mode()
def benchmark(args):
    device = torch.device(args.device)
    cuda = device.type == 'cuda'
    if cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. --device cpu is smoke-only, use very small shapes.')
    if cuda:
        torch.cuda.set_device(device)
    torch.manual_seed(42)
    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[args.dtype]
    b, c, v, d, depth = args.batch, args.contexts, args.vocab, args.feature_dim, args.depth
    root_z = torch.randn(b, 1, v, device=device, dtype=dtype)
    root_h = torch.randn(b, 1, args.hidden, device=device, dtype=dtype)
    branch_z = torch.randn(b, c, v, device=device, dtype=dtype)
    branch_h = torch.randn(b, c, args.hidden, device=device, dtype=dtype)
    mapping = torch.arange(v, device=device)
    width = depth + 2
    target = torch.randn(b, width, args.target_vocab, device=device).softmax(-1)
    if args.greedy:
        target = target.argmax(-1)
    parents = torch.arange(-1, width - 1, device=device).expand(b, -1).contiguous()
    tokens = torch.arange(10, 10 + width, device=device).expand(b, -1).contiguous()
    feedback_ids = torch.tensor([0] + [1 + c * i for i in range(depth)] + [-1], device=device).expand(b, -1).contiguous()
    tree = PackedTree(parents, tokens, feedback_ids, depth + 1)
    samples = torch.cat((tokens[:, 1:], torch.full((b, 1), 2, device=device)), 1)
    timings, parity = {}, {}
    probability_errors = []

    def engine(backend):
        result = FastLKReflex(feature_dim=d, backend=backend, feedback_scope=args.feedback_scope)
        result.start(b, v, args.hidden, device, max_contexts=1 + c * depth,
                     max_path_length=width, max_proposal_contexts=c, max_topk=args.topk)
        return result

    reference, candidate = engine('torch'), engine(args.backend)
    engines = [('torch', reference)] + ([(candidate.backend, candidate)] if candidate.backend != 'torch' else [])

    def prepare(instance):
        instance.propose(root_z, root_h, args.topk, mapping, root=True)
        for _ in range(depth):
            instance.propose(branch_z, branch_h, args.topk, mapping)

    def feedback(instance, path):
        if args.feedback_scope == 'root':
            if args.greedy:
                instance.update_from_target_tokens(target[:, 0], mapping)
            else:
                instance.update_from_target_probs(target[:, 0], mapping)
        else:
            instance.update_visited(target, mapping, path, greedy=args.greedy)

    # Verify proposals and state transitions BEFORE timing; no speed claim on
    # failed correctness. All proposal features/context caches are real APIs.
    for _ in range(3):
        expected_q = reference.correct(branch_z, branch_h)
        expected = expected_q.topk(args.topk)
        if candidate._kernels:
            actual_values, actual_ids, _ = candidate._kernels.propose(branch_z, candidate._feature(branch_h),
                candidate.state, args.topk, candidate._proposal_workspace)
        else:
            actual_values, actual_ids = candidate.correct(branch_z, branch_h).topk(args.topk)
        torch.testing.assert_close(actual_values, expected.values, rtol=3e-4, atol=2e-7)
        torch.testing.assert_close(expected_q.gather(-1, actual_ids), expected.values, rtol=3e-4, atol=2e-7)
        probability_errors.append((actual_values - expected.values).abs())
        prepare(reference)
        prepare(candidate)
        expected_path = trace_verified_path(tree, samples, 2)
        actual_path = trace_verified_path(tree, samples, 2, kernels=candidate._kernels, workspace=candidate.path_workspace)
        assert torch.equal(actual_path.packed_indices, expected_path.packed_indices)
        feedback(reference, expected_path)
        feedback(candidate, actual_path)
        torch.testing.assert_close(candidate.state, reference.state, rtol=3e-4, atol=3e-6)
    parity['max_state_abs_error'] = float((candidate.state - reference.state).abs().max().item())
    errors = torch.stack(probability_errors)
    parity['max_topk_probability_abs_error'] = float(errors.max().item())
    parity['mean_topk_probability_abs_error'] = float(errors.mean().item())
    parity['passed'] = True

    def measure(operation, setup=lambda: None):
        # Per-call events avoid including setup/proposal recaching in feedback
        # latency. Synchronization is strictly confined to this benchmark.
        for _ in range(args.warmup):
            setup()
            operation()
        gpu_ms, wall_ms = [], []
        start = torch.cuda.Event(enable_timing=True) if cuda else None
        end = torch.cuda.Event(enable_timing=True) if cuda else None
        for _ in range(args.iterations):
            setup()
            if cuda:
                torch.cuda.synchronize(device)
            began = time.perf_counter()
            if cuda:
                start.record()
            operation()
            if cuda:
                end.record()
                end.synchronize()
            elapsed = (time.perf_counter() - began) * 1000
            wall_ms.append(elapsed)
            gpu_ms.append(start.elapsed_time(end) if cuda else elapsed)
        return {'gpu_ms_median': statistics.median(gpu_ms) if cuda else None,
                'wall_ms_median': statistics.median(wall_ms), 'wall_ms_mean': statistics.mean(wall_ms)}

    for name, instance in engines:
        # Use a root-scope instance for repeated branch-only timing (no feedback
        # cache growth). Full-cycle timing below exercises requested scope.
        core = engine(name)
        core.feedback_scope = 'root'
        psi = core._feature(branch_h)
        def correction():
            if core._kernels:
                return core._kernels.correct_logits(branch_z, psi, core.state)
            return torch.baddbmm(branch_z.float(), psi, core.state.transpose(1, 2))
        def dense():
            return correction().softmax(-1).topk(args.topk)
        def fused():
            if core._kernels:
                return core._kernels.propose(branch_z, psi, core.state, args.topk, core._proposal_workspace)
            return dense()
        def extract():
            return trace_verified_path(tree, samples, 2, kernels=instance._kernels, workspace=instance.path_workspace)
        path = extract()
        def cycle():
            prepare(instance)
            feedback(instance, extract())
        timings[name] = dict(correction_only=measure(correction),
            dense_correction_softmax_topk=measure(dense), fused_correction_topk=measure(fused),
            visited_path_extraction=measure(extract),
            feedback_update=measure(lambda: feedback(instance, path), lambda: prepare(instance)),
            whole_reflex_verification_cycle=measure(cycle))
    return dict(benchmark='synthetic_production_reflex_pipeline', gpu_name=torch.cuda.get_device_name(device) if cuda else None,
                shape=vars(args), parity=parity, timings=timings,
                note='Component measurements only. CPU timings are smoke-only. No target/draft forwards, AAL or training speedup measured.')


if __name__ == '__main__':
    print(json.dumps(benchmark(parse_args()), indent=2))
