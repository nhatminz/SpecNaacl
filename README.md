# SpecNaacl: FastGRPO EAGLE-3 + Fast LK Reflex

This directory is a self-contained copy of the existing FastGRPO workspace.
It preserves FastGRPO's GRPO reward/loss, target-update schedule, speculative
tree verifier, concurrency-aware scheduler, and persistent online EAGLE update.
It adds an optional trajectory-local Fast LK Reflex correction during rollout.
For fair timing, use `METHOD=fastgrpo` or `METHOD=specnaacl` inside this same
repository; both modes share persistent training and authoritative target
sampling/verification semantics. ACTIVE uses the tensor-tree execution path.
The sibling original `fastgrpo/` checkout is provenance/external reference,
not the wall-clock baseline for this comparison.

- `REFLEX_MODE=off`: compact EAGLE logits, softmax, top-k, then fixed `d2t` mapping.
- `REFLEX_MODE=active`: `A psi` at every proposal context, followed by an analytic
  LK update reusing exact target sampling probabilities after temperature/top-p/top-k.
- `REFLEX_FEEDBACK_SCOPE=root|visited_path`: root ablation (default) or mean
  feedback from proposal contexts actually entered by the verifier; one state
  update per verification round, never counterfactual branches.
- No Reflex backward, optimizer, persistent model update, or extra target forward.
- `REFLEX_PROFILE=0` and `REFLEX_DIAGNOSTICS=0` are the defaults; diagnostic LK
  loss is not evaluated in that default path.
- `REFLEX_BACKEND=auto|torch|triton` selects FP32 Reflex engineering kernels.
  Auto uses fused Triton on supported CUDA setups; Torch is the reference path.
  Auto means availability, not the fastest measured backend. Run CUDA parity,
  [component benchmark](scripts/benchmark_reflex_pipeline.py) and
  [real rollout benchmark](scripts/benchmark_reflex_rollout.py)
  before trusting GPU speed/quality; local CPU tests are not a B200 benchmark.
- SpecForge `0.2.0` source is vendored at commit
  `3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83`.
- The inherited FastGRPO source provenance is
  `yedaotian9/FastGRPO@38e252493149072d2c5905f0a47de1d935d7170a`.

Start with [huongdanchay.md](huongdanchay.md). Technical details are in
[METHOD_FAST_LK_REFLEX.md](METHOD_FAST_LK_REFLEX.md), and all launcher knobs are
listed in [RUNNING.md](RUNNING.md).

All new outputs are written under `SpecNaacl/outputs/{pretrain,train}/...`.
No launcher clones repositories or downloads models/datasets.
