#!/usr/bin/env python3
"""Synthetic Reflex OFF/torch/Triton overhead + numerical parity, no training.

Uses already-produced verification probabilities, like the real rollout.
Does not measure target/draft forwards, AAL, task quality or end-to-end speedup.
CUDA events time warmed GPU execution; compilation/setup and verification are
excluded and separately identified. CPU mode is only a smoke test.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from helper.fast_lk_reflex import FastLKReflex, reflex_or_baseline_probabilities


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("auto", "torch", "triton"), default="auto")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--target-vocab-size", type=int, default=151936)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--draft-k", type=int, default=8)
    parser.add_argument("--draft-depth", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--verify-rounds", type=int, default=8)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for name in ("batch", "vocab_size", "target_vocab_size", "hidden_size", "feature_dim",
                 "draft_k", "draft_depth", "rounds", "warmup", "verify_rounds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.target_vocab_size < args.vocab_size or args.draft_k > args.vocab_size:
        parser.error("target vocabulary must cover compact vocabulary; draft-k must fit")
    return args


@torch.inference_mode()
def benchmark(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu only for a small smoke test")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    torch.manual_seed(args.seed)
    logits = torch.randn(args.batch, 1, args.vocab_size, device=device, dtype=dtype)
    hidden = torch.randn(args.batch, 1, args.hidden_size, device=device, dtype=dtype)
    branch_logits = torch.randn(args.batch, args.draft_k, args.vocab_size, device=device, dtype=dtype)
    branch_hidden = torch.randn(args.batch, args.draft_k, args.hidden_size, device=device, dtype=dtype)
    target = torch.randn(args.batch, args.target_vocab_size, device=device).softmax(-1)
    tokens = target.argmax(-1)
    mapping = torch.arange(args.vocab_size, device=device)

    def engine(backend):
        instance = FastLKReflex(feature_dim=args.feature_dim, learning_rate=args.lr,
                                weight_decay=args.weight_decay, seed=args.seed, backend=backend)
        instance.start(args.batch, args.vocab_size, args.hidden_size, device)
        return instance

    reference, candidate = engine("torch"), engine(args.backend)
    max_q_error = torch.zeros((), device=device)
    max_state_error = torch.zeros((), device=device)
    matching = torch.zeros((), device=device)
    comparisons = 0

    def update(instance):
        if args.greedy:
            return instance.update_from_target_tokens(tokens, mapping)
        return instance.update_from_target_probs(target, mapping)

    # Verify repeated real state transitions before offering timing numbers.
    verify_started = time.perf_counter()
    for _ in range(args.verify_rounds):
        expected = reference.correct(logits, hidden, cache_root=True)
        actual = candidate.correct(logits, hidden, cache_root=True)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-7)
        max_q_error = torch.maximum(max_q_error, (actual - expected).abs().max())
        matching += (actual.topk(args.draft_k).indices == expected.topk(args.draft_k).indices).sum()
        comparisons += args.batch * args.draft_k
        update(reference)
        update(candidate)
        torch.testing.assert_close(candidate.state, reference.state, rtol=2e-4, atol=3e-6)
        max_state_error = torch.maximum(max_state_error, (candidate.state - reference.state).abs().max())
        torch.testing.assert_close(candidate.correct(branch_logits, branch_hidden),
                                   reference.correct(branch_logits, branch_hidden), rtol=2e-4, atol=2e-7)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    verification_compile_time_s = time.perf_counter() - verify_started

    def one_round(instance):
        reflex_or_baseline_probabilities(logits, hidden, instance, cache_root=instance is not None)
        for _ in range(args.draft_depth):
            reflex_or_baseline_probabilities(branch_logits, branch_hidden, instance)
        if instance is not None:
            update(instance)

    def measure(instance):
        if instance is not None:
            instance.state.zero_()
        for _ in range(args.warmup):
            one_round(instance)
        if instance is not None:
            instance.state.zero_()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            started = time.perf_counter()
        for _ in range(args.rounds):
            one_round(instance)
        if device.type == "cuda":
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            elapsed_ms = (time.perf_counter() - started) * 1000
        return elapsed_ms / args.rounds

    # These are component-cycle measurements, not full rollout samples/s.
    off_ms = measure(None)
    torch_ms = measure(reference)
    candidate_ms = measure(candidate) if candidate.backend != "torch" else torch_ms
    return {
        "benchmark": "synthetic_reflex_component_cycle", "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_count_used": int(device.type == "cuda"), "backend": candidate.backend,
        "dtype": args.dtype, "batch": args.batch, "compact_vocab": args.vocab_size,
        "target_vocab": args.target_vocab_size, "hidden_size": args.hidden_size,
        "feature_dim": args.feature_dim, "draft_k": args.draft_k, "draft_depth": args.draft_depth,
        "rounds": args.rounds, "warmup_rounds": args.warmup, "greedy": args.greedy,
        "verification_including_first_jit_compile_s": verification_compile_time_s,
        "parity_passed": True, "max_probability_abs_error": float(max_q_error.item()),
        "max_state_abs_error": float(max_state_error.item()),
        "root_topk_id_agreement": float(matching.item()) / comparisons,
        "off_ms_per_round": off_ms, "torch_ms_per_round": torch_ms,
        "candidate_ms_per_round": candidate_ms,
        "added_ms_vs_off": candidate_ms - off_ms,
        "torch_over_candidate_ratio": torch_ms / max(candidate_ms, 1e-12),
        "note": "No target/draft forwards or task quality measured; CPU timings are smoke-only.",
    }


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(benchmark(args), indent=2))


if __name__ == "__main__":
    main()
