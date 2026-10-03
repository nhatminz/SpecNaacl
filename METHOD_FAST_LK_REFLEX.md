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

Both modes use the same EAGLE-3 compact logits and fixed `d2t` mapping.
ACTIVE adds `A psi` before normalized top-k. Torch uses the unfused reference;
Triton fuses correction/normalization/top-k, without dense branch q. Zero-state
Torch proposals reproduce OFF exactly; fused rounding/tie order need not be
bit-identical. The target sampler and token-matching verification stay authoritative.

After each target verification, `REFLEX_FEEDBACK_SCOPE=root` (default) uses
the root proposal; `visited_path` uses only actually entered proposal heads.
Unexpanded leaves have no draft-head q/psi and are excluded, even when their
target bonus token is emitted. Feedback stops at the first EOS. The target
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

In `visited_path`, replace `g psi^T` by its **mean over eligible visited heads**,
evaluated with the SAME pre-update A; apply decay and write A only once per round.
This changes feedback, not what the verifier commits. `--reflex_update_scope`
remains a CLI alias of `--reflex_feedback_scope`.

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

The Triton path fuses conversion/projection/normalization, then uses two proposal
launches: tiled correction/max/sum/top-k and summary merge. One CTA reuses an A
vocab tile across all small C contexts. Only selected ids/probabilities and FP32
max/sum normalization leave the merge; compact-id ascending ties are deterministic
(Torch top-k tie order is unspecified). Large
projection shapes use guarded FP32 cuBLAS instead of the fused feature kernel.
The LK update uses three tiled launches to condition the existing teacher,
reduce alpha/selected mass, and update `A` in-place, without materializing dense
compact teacher/mask/gradient/outer-product tensors. It supports full ~152k
Qwen vocabulary too. Visited feedback reconstructs q from native logits plus
psi/max/sum; it never caches full tree q in FP32. Teacher-mass and LK-statistics
launches span ALL visited slots; the third launch aggregates the mean and writes
A once. There is no update launch per depth. The dense `correct()` API remains
for reference/parity, not the production Triton proposal path.
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

## Tensor-tree and deployment benchmark (2026-10-03)

ACTIVE keeps full/packed parents, tokens and head-context ids on the GPU. The
Triton verifier extracts the first matching child path in one launch, consuming
the original sample-once target tokens, stopping at EOS. Feedback runs BEFORE
the one small path packet goes to CPU. This replaces per-node D2H copies,
CPU Node/link construction/traversal and the transfer thread pool. Legacy OFF
code stays in place. KV padding/pruning still needs CPU bookkeeping; this is
NOT a fully host-free rollout. Prefill transfers the attention metadata once
instead of reading one GPU scalar per prompt token. Target tree masks have a
reused flat pool and, on Triton, one construction launch.

Proposal summaries, native-logit/psi/norm feedback pools, update partials, root
indices and path buffers are reused. A compaction stays double-buffered; no
unmeasured indirection, stream overlap or hybrid dispatch has been enabled.
`auto` is the existing availability resolver, NOT an autotuner or speed claim.
Start with explicit `REFLEX_BACKEND=torch`; qualify Triton on your actual server:

```bash
pytest -q tests/test_fast_lk_reflex.py tests/test_reflex_cuda_pipeline.py
python scripts/benchmark_reflex_pipeline.py --backend triton --feedback-scope root \
  --batch 64 --vocab 16000 --contexts 8 --feature-dim 8 --topk 8
python scripts/benchmark_reflex_pipeline.py --backend triton --feedback-scope visited_path \
  --batch 64 --vocab 16000 --contexts 8 --feature-dim 8 --topk 8
```

Repeat component measurements for actual vocab/H and C=1/4/7/8, D=4/8, K=4/8.
They report correction-only, old dense pipeline, fused pipeline, path extraction,
feedback and whole Reflex cycle, GPU-event and host-wall times, and numerical
errors. CPU mode is smoke-only. Then measure REAL production-model rollout:

```bash
python scripts/benchmark_reflex_rollout.py \
  --target-model "$TARGET_MODEL_PATH" --draft-config "$DRAFT_CONFIG" \
  --draft-checkpoint "$DRAFT_CHECKPOINT" --vocab-mapping "$VOCAB_MAPPING" \
  --dataset-path "$DATASET_PATH" --batch-size 8 --responses 8 \
  --warmup 2 --iterations 5
```

Use `--target-adapter` for the same trained target LoRA. This benchmark loads
existing local weights, reuses the production dataset loader AND training prompt
collator, and measures synchronized generation wall clock/tokens/s, weighted AAL,
actual backbone forwards and peak memory. `*-zero` controls isolate tensor-path
engineering from acceptance learning. Frozen weights/no optimizer: these are NOT
training throughput or task-quality results. Optional `--profile-trace NEW.json`
records one extra, untimed rollout; there is no profiler in default training.

For an explicit short end-to-end training comparison (creates NEW isolated runs):

```bash
MODEL_KEY=qwen25_3b REFLEX_BACKEND=torch REFLEX_FEEDBACK_SCOPE=visited_path \
  BENCHMARK_STEPS=20 bash scripts/benchmark_reflex_training.sh
```

Keep the same data, seed, batch, accumulation, target/draft checkpoints, device
count and scope when comparing Torch/Triton. The read-only summarizer separates
tokens/generation-time from tokens/whole-job-time (the existing training summary's
`generation_tokens_per_s` is historically a whole-job metric). Model/data startup
and reward filtering are included in that short training benchmark. No full
pretrain is invoked. Keep backend AND feedback scope fixed for comparable resumes.
Choose the backend using repeated measured end-to-end runs, not component speed
or AAL alone; no B200 result has been produced by the CPU-only development host.

The default path neither profiles Reflex nor computes diagnostic LK loss.
`REFLEX_DIAGNOSTICS=1` enables rollout-aggregate LK alpha/loss only;
`REFLEX_PROFILE=1` measures aggregate host dispatch time without synchronizing
CUDA in the Reflex hot path and reports `reflex_profile_time_ms`. Both switches
are intended only for dedicated diagnostic runs and neither emits
per-token/per-round disk logs.
