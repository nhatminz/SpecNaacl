# Fast LK Reflex

For the native EAGLE hidden feature `h`, a fixed deterministic random projection
forms a small normalized context feature:

```text
psi = normalize(h R),       dim(psi) = 8 by default
z   = z0 + A psi
```

`A psi` is the chosen fast-adapter parameterization. `R` and `A` are ordinary
non-gradient tensors. `R` is fixed for the rollout and seeded reproducibly; `A`
starts at zero separately for every response and is removed when that response
leaves the active batch. `A` is defined on SpecForge's compact EAGLE vocabulary.

Both `REFLEX_MODE=off` and `REFLEX_MODE=active` use the same EAGLE-3 compact
logits, compact softmax, top-k and fixed `d2t` mapping. The only ACTIVE-mode
difference is the additive `A psi` correction. Therefore a zero fast state is
exactly proposal-equivalent to OFF mode.

After each target verification, only the root proposal is updated. The target
token is sampled from `build_sampling_probs`, which applies the configured
temperature, top-p and top-k and returns the final normalized sampling
distribution. For stochastic decoding Reflex reuses that same tensor. For
greedy decoding it constructs the mathematically equivalent one-hot directly
in compact-vocabulary space. It adds neither a target forward nor a separate
full-vocabulary softmax/one-hot. The fixed compact-vocabulary entries are
gathered and conditioned before computing:

```text
alpha = sum_i min(p_i, q_i)
L     = -log(alpha + eps)
m_i   = 1[q_i < p_i]
S     = sum_i m_i q_i
g_i   = q_i (S - m_i) / (alpha + eps)

A <- (1 - lr * weight_decay) A - lr * g psi^T
```

Here `p` is the conditional compact-vocabulary teacher distribution:
`p_full[d2t] / (sum(p_full[d2t]) + eps)`. This is a conditional
compact-vocabulary LK objective. It is not claimed to equal the exact
full-vocabulary rejection-acceptance probability of FastGRPO's verifier.

The gradient and update are analytic batched GPU tensor operations. Reflex never
calls `backward`, creates an optimizer, changes EAGLE parameters, or adds a target
forward. The existing FastGRPO verifier decides actual acceptance exactly as
before. Persistent online EAGLE training remains a separate SpecForge operation
after rollout and is not replaced by Reflex.

The logged AAL keeps FastGRPO's weighted definition:
`total_acc_length / total_decoded_token_num` over verification rounds. Accepted
length includes the verified root/target bonus token; draft acceptance rate is
accepted draft tokens divided by proposed draft tokens.

Fast state arithmetic is FP32 even inside the model's BF16/FP16 autocast.
Previously the autocast context could cast the entire `A` at every correction,
and cached BF16 `psi` was incompatible with the FP32 in-place update. The model
and EAGLE training precision are unchanged; this guard applies only to Reflex.

`REFLEX_BACKEND=auto|torch|triton` selects the engineering implementation:

- `auto` (default): CUDA + installed Triton + feature dimension <=64 use Triton;
  CPU, missing Triton or larger feature dimensions use Torch.
- `torch`: FP32 reference correction via `baddbmm` and in-place rank-one update.
- `triton`: require CUDA/Triton and dimension <=64, otherwise fail clearly.
  Compilation/runtime errors are surfaced, not silently swallowed or retried.

The Triton path fuses conversion/projection/normalization, then conversion plus
`A psi` addition (the shared Torch softmax/top-k/mapping stay unchanged). Large
projection shapes use guarded FP32 cuBLAS instead of the fused feature kernel.
The LK update uses three tiled launches to condition the existing teacher,
reduce alpha/selected mass, and update `A` in-place, without materializing dense
compact teacher/mask/gradient/outer-product tensors. It supports full ~152k
Qwen vocabulary too; no sample/token, feature dimension or update is dropped.
Runtime strides and 64-bit batch offsets support strided verification-root
views and shrinking active batches without fixing a CUDA graph's batch size.

`R` stays seeded and cached per canonical CUDA device/shape/seed. Finished
trajectories are compacted in stable order into a reusable second `A` buffer,
then the buffers swap. This doubles fast-adapter storage only (not model/KV
storage), avoiding a new large state allocation for every finished-row event.
Both buffers are released at rollout completion, and each new rollout starts
with zero state. Checkpoint/resume and persistent EAGLE parameters are unchanged.

Fused reductions can differ from cuBLAS in FP32 rounding; bit-identical proposals
at near-ties are not promised. CPU tests validate AMP isolation, exact Torch
equations and a simulation of actual tiled kernel source. The real Triton
compiler/GPU parity tests and performance still need a CUDA server. No B200
speedup or task-quality preservation has been measured on the local CPU machine.

Before enabling fused training on the server, run:

```bash
pytest -q tests/test_fast_lk_reflex.py -k real_triton
python scripts/benchmark_reflex.py --device cuda --backend triton \
  --batch 64 --hidden-size 2048 --vocab-size 32000 \
  --target-vocab-size 151936 --dtype bf16 --verify-rounds 20
```

Set vocab/hidden/batch sizes to the actual draft/run. The benchmark checks
repeated-state numerical parity first and reports warmed OFF/Torch/candidate
milliseconds per component cycle, added milliseconds vs OFF, top-k agreement,
GPU name and backend. First JIT compilation is excluded from warmed timing and
reported as part of verification/setup time. It does not run a model, measure
AAL or measure end-to-end training/task quality. CPU mode is smoke-only:

```bash
python scripts/benchmark_reflex.py --device cpu --backend torch --batch 2 \
  --hidden-size 7 --feature-dim 3 --vocab-size 17 --target-vocab-size 23 \
  --draft-k 3 --draft-depth 2 --rounds 3 --warmup 1 --verify-rounds 3
```

Use `REFLEX_BACKEND=triton` before the existing training command after CUDA
checks/benchmark pass. Use `REFLEX_BACKEND=torch` for the reference implementation
or if a server/compiler-specific issue is encountered. Diagnostics/profile remain
off by default; use the existing AAL/reward metrics on matched real runs to
validate end-to-end benefit and task quality rather than assuming zero overhead.

The default path neither profiles Reflex nor computes diagnostic LK loss.
`REFLEX_DIAGNOSTICS=1` enables rollout-aggregate LK alpha/loss only;
`REFLEX_PROFILE=1` measures aggregate host dispatch time without synchronizing
CUDA in the Reflex hot path and reports `reflex_profile_time_ms`. Both switches
are intended only for dedicated diagnostic runs and neither emits
per-token/per-round disk logs.
