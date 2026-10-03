#!/usr/bin/env python3
"""CUDA-event comparison of selectable Reflex kernels; no model speedup claim.

Run this on the deployment GPU before changing the conservative defaults.
All candidates are checked against the same Torch FP32 equations first.
``--sweep`` compiles BV=64/128/256, BC=1/2/4/8, warps=2/4/8.
Full model wall-clock/AAL still requires benchmark_reflex_rollout.py.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from helper import fast_lk_reflex_kernels as kernels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("batch", 64), ("vocab", 16000), ("target-vocab", 151936),
                          ("hidden", 2048), ("contexts", 8), ("dim", 8), ("topk", 8),
                          ("path-length", 6), ("warmup", 10), ("iterations", 30)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--compile-feature", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if min(vars(args)[key] for key in ("batch", "vocab", "target_vocab", "hidden",
                                      "contexts", "dim", "topk", "path_length", "iterations")) < 1:
        parser.error("all shapes and iterations must be positive")
    if args.topk > args.vocab or args.vocab > args.target_vocab or args.path_length > args.contexts:
        parser.error("require topk <= vocab <= target-vocab and path-length <= contexts")
    return args


@torch.inference_mode()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script never substitutes CPU timings")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    b, c, v, d, p = args.batch, args.contexts, args.vocab, args.dim, args.path_length
    hidden = torch.randn(b, c, args.hidden, device=device, dtype=torch.bfloat16)
    projection = torch.randn(args.hidden, d, device=device) * args.hidden ** -0.5
    raw = torch.randn(b, c, v, device=device, dtype=torch.bfloat16)
    state = torch.randn(b, v, d, device=device) * .02
    psi_ref = torch.nn.functional.normalize(hidden.float().matmul(projection), dim=-1, eps=1e-6)
    corrected_ref = torch.baddbmm(raw.float(), psi_ref, state.transpose(1, 2))
    q_ref = corrected_ref.softmax(-1)
    top_ref = q_ref.topk(args.topk)
    maximum = corrected_ref.amax(-1)
    norm = torch.stack((maximum, (corrected_ref - maximum[..., None]).exp().sum(-1)), -1)
    target = torch.randn(b, p, args.target_vocab, device=device).softmax(-1)
    mapping = torch.arange(v, device=device)
    indices = torch.arange(p, device=device).expand(b, -1).contiguous()
    contexts = indices.clone()
    results = []

    def timing(name, fn, verify=None, setup=None):
        try:
            result = fn()
            if verify is not None:
                verify(result)
            for _ in range(args.warmup):
                if setup is not None:
                    setup()
                fn()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(args.iterations):
                if setup is not None:
                    setup()
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            results.append(dict(name=name, gpu_ms_median=statistics.median(samples),
                                gpu_ms_p90=sorted(samples)[int(.9 * (len(samples) - 1))], parity=True))
        except (RuntimeError, AssertionError, ValueError) as exc:
            results.append(dict(name=name, parity=False, error=str(exc)[:500]))

    timing("feature:torch-fp32", lambda: torch.nn.functional.normalize(
        hidden.float().matmul(projection), dim=-1, eps=1e-6))
    for strategy in ("triton", "torch"):
        timing("feature:" + strategy,
               lambda strategy=strategy: kernels.feature(hidden, projection, strategy=strategy),
               lambda value: torch.testing.assert_close(value, psi_ref, rtol=2e-4, atol=2e-6))
    bf16_projection = projection.to(torch.bfloat16)
    bf16_feature = lambda: torch.nn.functional.normalize(
        hidden.matmul(bf16_projection).float(), dim=-1, eps=1e-6)
    timing("feature:bf16-cublas-semantic-candidate", bf16_feature)
    mixed_error = float((bf16_feature() - psi_ref).abs().max().item())
    results[-1]["max_abs_vs_fp32"] = mixed_error
    try:
        torch.testing.assert_close(bf16_feature(), psi_ref, rtol=2e-4, atol=2e-6)
    except AssertionError:
        results[-1]["parity"] = False
        results[-1]["note"] = "Not eligible for FP32 Reflex semantics"
    if args.compile_feature:
        try:
            compiled = torch.compile(lambda x, r: torch.nn.functional.normalize(
                x.float().matmul(r), dim=-1, eps=1e-6))
            timing("feature:torch-compile-fp32", lambda: compiled(hidden, projection),
                   lambda value: torch.testing.assert_close(value, psi_ref, rtol=2e-4, atol=2e-6))
        except Exception as exc:
            results.append(dict(name="feature:torch-compile-fp32", parity=False, error=str(exc)[:500]))
    timing("correction:torch-baddbmm", lambda: torch.baddbmm(raw.float(), psi_ref, state.transpose(1, 2)))
    timing("normalization:torch-softmax", lambda: corrected_ref.softmax(-1))
    timing("topk:torch", lambda: q_ref.topk(args.topk))
    configs = ([(bv, bc, warps) for bv in (64, 128, 256) for bc in (1, 2, 4, 8)
                for warps in (2, 4, 8)] if args.sweep else [(256, 2, 4)])
    for strategy in ("serial", "parallel", "tiled"):
        for bv, bc, warps in configs:
            if strategy != "tiled" and bc != (2 if not args.sweep else 1):
                continue
            name = f"correction:{strategy}:bv{bv}:bc{bc}:w{warps}"
            timing(name, lambda strategy=strategy, bv=bv, bc=bc, warps=warps:
                   kernels.correct_logits(raw, psi_ref, state, strategy=strategy,
                                          block_vocab=bv, context_tile=bc, num_warps=warps),
                   lambda value: torch.testing.assert_close(value, corrected_ref, rtol=2e-4, atol=2e-6))
    timing("proposal:torch-baddbmm-softmax-topk",
           lambda: torch.baddbmm(raw.float(), psi_ref, state.transpose(1, 2)).softmax(-1).topk(args.topk))
    proposal_configs = ([(bv, warps) for bv in (64, 128, 256) for warps in (2, 4, 8)]
                        if args.sweep else [(256, 4)])
    for strategy in ("fused", "sort"):
        for bv, warps in proposal_configs:
            name = f"proposal:{strategy}:bv{bv}:w{warps}"
            def verify_proposal(result):
                values, ids, _norm = result
                torch.testing.assert_close(values, top_ref.values, rtol=3e-4, atol=2e-7)
                torch.testing.assert_close(q_ref.gather(-1, ids), top_ref.values, rtol=3e-4, atol=2e-7)
                torch.testing.assert_close(_norm[..., 0], maximum, rtol=2e-4, atol=2e-6)
            timing(name, lambda strategy=strategy, bv=bv, warps=warps:
                   kernels.propose(raw, psi_ref, state, args.topk, strategy=strategy,
                                   block_vocab=bv, num_warps=warps), verify_proposal)
    for strategy in ("serial", "parallel", "tiled"):
        timing("proposal:hybrid:" + strategy,
               lambda strategy=strategy: kernels.correct_logits(raw, psi_ref, state,
                   strategy=strategy).softmax(-1).topk(args.topk),
               lambda result: torch.testing.assert_close(result.values, top_ref.values,
                                                         rtol=3e-4, atol=2e-7))
    serial_state = state.clone()
    alpha_ref = kernels.update_path(serial_state, raw, psi_ref, norm, target, mapping,
        indices, contexts, greedy=False, eps=1e-8, learning_rate=.05, decay=.99)
    for strategy in ("serial", "parallel"):
        for bv in ((64, 128, 256) if args.sweep else (256,)):
            for warps in ((2, 4, 8) if args.sweep else (4,)):
                working = state.clone()
                def operation(strategy=strategy, bv=bv, warps=warps):
                    return kernels.update_path(working, raw, psi_ref, norm, target, mapping,
                        indices, contexts, greedy=False, eps=1e-8, learning_rate=.05, decay=.99,
                        strategy=strategy, block_vocab=bv, num_warps=warps)
                def verify_update(alpha):
                    torch.testing.assert_close(alpha, alpha_ref, rtol=3e-4, atol=3e-6)
                    torch.testing.assert_close(working, serial_state, rtol=3e-4, atol=3e-6)
                timing(f"feedback:{strategy}:bv{bv}:w{warps}", operation, verify_update,
                       lambda: working.copy_(state))
    report = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__,
                  triton=__import__("triton").__version__, shape=vars(args), results=results,
                  note="Synthetic primitive timings only; use full rollout benchmark before selecting B200 defaults.")
    output = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main(parse_args())
