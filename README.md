# SpecNaacl: FastGRPO EAGLE-3 + Fast LK Reflex

This directory is a self-contained copy of the existing FastGRPO workspace.
It preserves FastGRPO's GRPO reward/loss, target-update schedule, speculative
tree verifier, concurrency-aware scheduler, and persistent online EAGLE update.
It adds an optional trajectory-local Fast LK Reflex correction during rollout.

- `REFLEX_MODE=off`: the original EAGLE-3 proposal path.
- `REFLEX_MODE=active`: compact-vocabulary `z = z0 + A psi`, followed by the
  analytic root LK update from the target logits already produced by verification.
- No Reflex backward, optimizer, persistent model update, or extra target forward.
- SpecForge `0.2.0` source is vendored at commit
  `3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83`.
- The inherited FastGRPO source provenance is
  `yedaotian9/FastGRPO@38e252493149072d2c5905f0a47de1d935d7170a`.

Start with [huongdanchay.md](huongdanchay.md). Technical details are in
[METHOD_FAST_LK_REFLEX.md](METHOD_FAST_LK_REFLEX.md), and all launcher knobs are
listed in [RUNNING.md](RUNNING.md).

All new outputs are written under `SpecNaacl/outputs/{pretrain,train}/...`.
No launcher clones repositories or downloads models/datasets.
