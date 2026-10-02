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
FlashAttention v2 import is optional; pretraining now requires an explicit
`sdpa`/`flex_attention` selection if `fa` is unavailable, as detailed in the
pretraining optimization section below. Dependencies required
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

## EAGLE-3 throughput optimization and tqdm (2026-10-02)

### Correct workspace and merge scope

These changes are now applied directly to **`D:\VDT\SpecNaacl`** (origin
`nhatminz/SpecNaacl`), not the earlier `D:\analysis_spec` checkout. The working
tree was clean before this update. Matching vendored files received the earlier
pretraining changes; the divergent launcher and GRPO entrypoint were merged
selectively rather than replaced.

Preserved this repo's FastLKReflex implementation, fair-comparison pipeline,
GRPO distributed sampler and main-rank logic, Python/dependency validation and
exact pins, output/checkpoint directory layout, model-specific chat templates,
dataset conversion and feature-capture commands. `grpo_speculative.py` and
legacy `train_draft.py` changed only in their tqdm configuration. No GRPO loss,
reflex equations, sampling or optimizer behavior was edited.

### Optimizations applied

- Removed `torch.cuda.empty_cache()` from `OnlineEagle3Model.forward()`. Existing
  tensor deletions remain; CUDA allocator blocks can be reused each microbatch.
  No flush remains in the SpecForge pretraining hot path. GRPO and legacy draft
  training's unrelated cache-management code was not altered.
- EAGLE pretrain world size 1 explicitly calls `prepare_model(..., wrap=False)`
  and cannot wrap FSDP/DDP. The optimizer still targets the inner draft only.
- `PRETRAIN_DISTRIBUTED_MODE=auto|ddp|fsdp`: auto is plain at one rank, replicated
  DDP/NO_SHARD at multiple ranks. Sharded FSDP is explicit. Intermediate
  accumulation steps now run both forward/backward under `no_sync()`; clipping,
  LR scheduling and gradient/loss normalization equations remain unchanged.
- `PRETRAIN_ATTENTION_BACKEND=fa|sdpa|flex_attention`: new runs request FA and
  probe the actual EAGLE CUDA varlen forward/backward API. Missing or incompatible
  standard FlashAttention fails with a clear explicit-alternative message.
  Existing attention implementations/masks were not rewritten. An unset backend
  on resume retains the saved backend, or historical flex_attention.
- Deterministic offline length-aware global batch plans, with boundaries
  512/768/1024/1280/1536/1792/2048. Shuffle uses `seed + epoch`; ranks slice the
  same plan. The existing collator pads to each batch's actual longest sequence,
  not bucket boundaries. No packing or extra token truncation is introduced.
- Length indexing uses metadata-only FakeTensor loads on rank 0, file/stat-aware
  JSON caching and one broadcast to peers. Gzip indexing still needs an initial
  decompression pass. Index construction is startup work, not timed throughput.
- New runs retain the last short batch, dropping no source sample. Equal-size
  distributed rank shards can repeat at most `world_size - 1` refs in that tail,
  following ordinary DistributedSampler padding; single rank does not repeat.
  Non-divisible datasets now have a ceil-based epoch horizon. Existing fixed
  accumulation validation still rejects a final incomplete optimizer window
  rather than silently discarding samples; accumulation=1 remains the default.
- Removed redundant defensive clones only for newly loaded offline EAGLE file
  features. Existing normalizer/collator semantics and pinned/nonblocking H2D
  transfers are retained. Loader workers default to 8; training OMP/MKL defaults
  to 1 thread unless explicitly overridden.
- Compact teacher and CPU optimizer offload default to false. No activation
  offload or gradient checkpointing was enabled. Batch defaults, max length 2048,
  TTT 7, draft vocab, dataset and feature-capture semantics are preserved.

### Resume and progress

New checkpoints retain wrapper kind, sampler version, seed, dataset size, bucket
settings and length-index identity. Resume reconstructs the saved epoch's plan
before seeking its exact sample position, including a final short batch. Legacy
checkpoints retain their v1 sampler/drop-last order rather than switching to
bucketing mid-epoch; keep their original seed/batch/accumulation configuration.
Model/objective resume contracts remain strict. Historical one-rank FSDP state
can resume into the plain model. Multi-rank sharded FSDP state requires explicit
`fsdp` and the original world size. DDP optimizer state remains replicated and
restores with each rank's own RNG; no unrequested shard conversion is attempted.

EAGLE pretraining tqdm now works through the existing `2>&1 | tee` launcher,
including non-TTY output. Rank 0 displays optimizer-step count/ETA and epoch/batch;
loss, LR and samples/s reuse already-materialized periodic logging values, with
no extra tensor `.item()` or CUDA synchronization. It resumes from the saved
step and closes on exceptions. GRPO and legacy draft epoch/batch bars retain
their existing metrics, use clear labels and restored epoch counters, and limit
refresh to one second. Set `TQDM_DISABLE=1` to disable bars.

### Files changed/added for this update

- Launch/config: `pretrain_eagle3_sharegpt_b200.sh`,
  `configs/_shared/b200_common.env`, `scripts/launch/pretrain_model.sh`,
  `third_party/SpecForge/examples/configs/offline/colocated/qwen2.5-7b-eagle3-offline.yaml`,
  `third_party/SpecForge/specforge/config/schema.py`.
- EAGLE/runtime: `third_party/SpecForge/specforge/algorithms/eagle3/model.py`,
  `third_party/SpecForge/specforge/modeling/draft/llama3_eagle.py`,
  `third_party/SpecForge/specforge/launch.py`,
  `third_party/SpecForge/specforge/optimizer.py`,
  `third_party/SpecForge/specforge/training/assembly.py`, `backend.py`,
  `controller.py`, `schedule.py`, `trainer.py` in that training directory.
- New helpers: `third_party/SpecForge/specforge/training/length_bucketing.py`,
  `third_party/SpecForge/specforge/training/pretrain_attention.py`.
- Progress only: `grpo_speculative.py`, `train_draft.py`.
- New benchmark: `scripts/benchmark_pretrain.sh`, `scripts/benchmark_pretrain.py`.
- Tests: new `tests/test_eagle3_pretrain.py`, `tests/test_training_progress.py`;
  updated `pytest.ini`, `tests/test_shell_scripts.py`.
- Docs: `ENVIRONMENT.md`, this report (previous content preserved).

### Verification in this actual repo

Local test interpreter is Python 3.13.9, PyTorch 2.9.0+cpu, Transformers 5.18.0,
Datasets 5.0.1, pytest 9.1.1. This is **not** the pinned B200 environment; production
version guards and dependency pins have not been loosened or upgraded.

- `python -m compileall -q .`: PASS.
- `pytest -q`: **77 passed, 1 skipped**, seven existing LR-scheduler advisory
  warnings, with Git Bash explicitly selected in this Windows environment.
- The first run had **74 passed, 1 skipped, 2 failed**: existing shell tests
  invoked nonfunctional WSL Bash. Their harness now accepts `BASH_BIN` or resolves
  a full Bash path and uses the current test Python for dry-run UUID generation.
  No shell test assertions were skipped or relaxed.
- Four edited/new shell/config scripts pass `bash -n`; existing launcher dry-run
  and fair-method command-parity tests also pass with Git Bash.
- `python scripts/benchmark_pretrain.py --help`: PASS.
- `git diff --check`: PASS.

The one skipped test is actual two-process CPU DDP: this Windows PyTorch build
reports `unsupported gloo device`. Topology selection, draft-only optimizer,
attention option routing/FA error handling, sample coverage/rank lengths/seed
shuffle/padding reduction, cache and gzip indexing, threaded loader seek,
checkpoint/optimizer/RNG/data-order resume, second-epoch/short-tail resume,
accumulation sync-context scope, tqdm behavior and benchmark output isolation
are covered. Existing FastLKReflex, sampling and checkpoint tests pass as well.

To reproduce this workstation's test run in PowerShell:

```powershell
cd D:\VDT\SpecNaacl
$env:BASH_BIN = 'D:\Git\bin\bash.exe'
pytest -q
```

On the Linux B200 host, run normal `pytest -q` in the pinned environment again.
FA numerical parity, NCCL/DDP/FSDP execution, actual B200 throughput and peak VRAM
are still unmeasured. **No full pretrain or GPU benchmark was run, and no speedup
is claimed.**

### Run and benchmark

Keep existing model/dataset/run environment variables. Through a model wrapper:

```bash
PRETRAIN_ATTENTION_BACKEND=fa PRETRAIN_DISTRIBUTED_MODE=auto \
NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 bash pretrain_qwen25_3b.sh
```

If standard FA is unavailable, explicitly choose
`PRETRAIN_ATTENTION_BACKEND=sdpa`; mandatory SGLang `flash-attn-4` does not alone
prove availability of EAGLE's standard FA interface. Four GPUs use
`NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3`; supported counts are 1/2/4/8.
Nominal effective batch is per-GPU batch × accumulation × world size; the final
short batch may be smaller. Wrappers print that product and forward the backend,
topology, buckets, workers and memory-placement settings to pretraining.

Opt-in benchmark reuses existing features/config/vocab and uses a new sibling
output directory; it refuses an existing run/output redirect and never captures
or resumes production training:

```bash
TARGET_MODEL_PATH=/workspace/storage-shared/models/Qwen2.5-3B-Instruct \
PRETRAIN_ROOT=/path/to/existing/pretrain/run \
PRETRAIN_ATTENTION_BACKEND=fa PRETRAIN_DISTRIBUTED_MODE=auto \
NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 \
BENCHMARK_STEPS=30 BENCHMARK_WARMUP_STEPS=5 \
bash scripts/benchmark_pretrain.sh
```

The report is `benchmark_report.json`: `optimizer_step_time_s`, `samples/s`, GPU
count, attention backend, distributed mode, effective batch and timed samples.
Default 30 steps comprise five warmup + 25 measured optimizer steps. The full
production LR horizon is preserved; timing includes compute/data wait but
excludes model/index startup and final checkpoint. Multi-rank reporting uses
the slowest elapsed time and summed samples. CUDA synchronization exists only
in this opt-in benchmark; no heavy profiling was added to default training.
The benchmark retains this repo's strict environment validator and accepts both
`PRETRAIN_GRADIENT_ACCUMULATION` and wrapper `PRETRAIN_ACCUMULATION_STEPS` aliases.
