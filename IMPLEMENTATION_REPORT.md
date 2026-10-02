# Implementation report

## Scope completed

The in-repository comparison now has one runtime and two explicit methods:

- `METHOD=fastgrpo`: the current persistent online EAGLE-3 update pipeline with
  FastLKReflex disabled.
- `METHOD=specnaacl`: the identical pipeline with FastLKReflex enabled.

The original sibling `fastgrpo/` repository is an external implementation
reference, not the timing baseline. The fair launchers share the target and
draft checkpoints, compact-vocabulary proposal path, target sampler, tree
builder/verifier, online draft update, data order, seed, batching, timing,
logging, and checkpoint implementation. A launcher test compares the complete
dry-run commands after removing only `--method` and `--reflex_mode`.

## FastLKReflex overhead changes

- `REFLEX_PROFILE=0`, `REFLEX_DIAGNOSTICS=0`, and rollout timing diagnostics
  are off by default.
- The analytic update computes alpha and its logit gradient without computing
  `-log(alpha)`. The diagnostic loss and device scalar accumulators exist only
  when diagnostics are explicitly enabled.
- Reflex profiling no longer calls `cuda.synchronize()` in correction or
  update. Optional profile time measures host dispatch overhead. Existing
  rollout timing synchronizations are all guarded by `statistical_time`, whose
  default is now false.
- Correction uses batched `torch.baddbmm` for `A @ psi`; the update uses
  in-place batched `state.baddbmm_` and does not materialize the outer product.
  With zero weight decay it uses `beta=1` and never calls `state.mul_(1)`.
- The fixed random projection is cached by device, hidden size, feature size,
  and seed. Root probabilities/features are retained by reference, then
  cleared immediately after the update.
- Finished trajectories are removed in one `index_select` per verification
  round, preserving active-batch order.
- Target sampling and Reflex reuse the same already-computed normalized target
  distribution. Reflex cannot invoke a target model because its sampling API
  accepts logits only. Greedy decoding builds supervision directly in the
  compact vocabulary instead of allocating a full-target-vocabulary one-hot.
- Reflex state and correction remain in the EAGLE compact vocabulary. No
  full-target-vocabulary Reflex state was introduced.
- Diagnostic/profile fields are omitted from runtime aggregation and output
  logs when their switches are off; there is no per-token or per-round Reflex
  log.

The numerical tests cover zero-state proposal identity, fused nonzero
correction against the previous einsum formula, analytic-gradient parity, and
compact greedy supervision against the full-vocabulary one-hot reference.

## Dependency environment

`requirements.txt` contains exact direct pins only and no Python standard
library modules. The supported Python versions are 3.12.12 and 3.12.13 (the
default for new environments). The anchored library stack is PyTorch 2.13.0,
Transformers 5.12.1, and SGLang 0.5.18. SpecForge is installed without dependency
resolution:

```bash
python -m pip install --no-deps -e third_party/SpecForge --no-build-isolation
```

`ENVIRONMENT.md` contains exact online and offline-wheelhouse setup commands for
the B200 host. No `requirements-optional.txt` was added: SpecForge's optional
FlashAttention v2 import has a flex-attention fallback. Dependencies required
by SGLang itself are still installed through SGLang's package metadata.

## Files changed

- Runtime: `grpo_speculative.py`, `helper/fast_lk_reflex.py`,
  `helper/sampling.py`, `helper/specualtive_generate.py`,
  `helper/method_config.py`.
- Launch/config: `configs/_shared/b200_common.env`,
  `scripts/launch/train_model.sh`, `scripts/run_fastgrpo_fair.sh`,
  `scripts/run_specnaacl.sh`, `scripts/validate_environment.py`.
- Dependencies: `requirements.txt`, `requirements-policy-lag.txt`,
  `ENVIRONMENT.md`, `DEPENDENCIES_POLICY_LAG.md`,
  `pretrain_eagle3_sharegpt_b200.sh`, `policy_lag_analysis.py`.
- Tests: `tests/test_fast_lk_reflex.py`, `tests/test_sampling.py`,
  `tests/test_requirements.py`, `tests/test_shell_scripts.py`.
- Documentation: `README.md`, `README_B200_POLICY_LAG.md`, `RUNNING.md`,
  `METHOD_FAST_LK_REFLEX.md`, `huongdanchay.md`, this report.

## Verification on this workstation

Passed:

```text
python3 -m compileall -q .
find . -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
python3 -m pytest -q tests/test_model_configs.py tests/test_shell_scripts.py tests/test_requirements.py
    8 passed
uv pip compile requirements.txt --python-version 3.12 --index-strategy first-index --prerelease allow
    resolved 235 packages
```

The required full commands were also attempted:

- `pytest -q` stopped during collection because this workstation's Python 3.8
  environment has no `torch`; the three affected modules are
  `test_checkpointing.py`, `test_fast_lk_reflex.py`, and `test_sampling.py`.
- `python3 -m pip check` failed because the workstation is not the project
  environment: it lacks Torch and several unrelated system packages and has
  pre-existing SpaCy/Pydantic and SpaCy/Typer conflicts.

These failures were not hidden with skips. Run both commands in the clean B200
environment from `ENVIRONMENT.md` before a real experiment.

## Not yet verified

- Tensor/autograd tests with the pinned PyTorch 2.13.0 stack.
- An end-to-end EAGLE-3 rollout/training run with the real model and draft
  checkpoint on B200.
- Multi-GPU/NCCL checkpoint and resume behavior under the pinned stack.
- B200 wall-clock or throughput. No speedup is claimed; no long training or
  performance benchmark was launched on this workstation.
