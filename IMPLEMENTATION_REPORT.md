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
library modules. The interpreter gate is Python >=3.12.0; `.python-version`
prefers the 3.12 series without locking a patch release. The anchored library stack is PyTorch 2.13.0,
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

## Standard FlashAttention import failure recovery (2026-10-02)

The supplied B200 log passes the pinned runtime dependency check, then fails
before capture/pretraining because `flash_attn_varlen_func` cannot be imported
from `flash_attn` (unknown location). This establishes that the standard EAGLE
FA interface is missing/incompatible, not that training ran out of memory.
The log alone does not establish which package/build caused the missing API.

Changed files:

- `third_party/SpecForge/specforge/training/pretrain_attention.py`: report the
  concrete varlen forward/backward and padding APIs required by EAGLE. Importing
  a namespace alone is not considered backend validation.
- `scripts/check_pretrain_attention.py`: standalone selected-backend check;
  optional real FA CUDA forward/backward probe. On failure return exit code 2
  with the explicit SDPA override and original cause, without a redundant
  Python exception traceback. SDPA selection is not advertised as a CUDA probe.
- `pretrain_eagle3_sharegpt_b200.sh`: use the checker and report the prepared run
  directory so a model wrapper can retry with an explicit backend and reuse
  preparation. Failure still stops before feature capture/training.
- `tests/test_pretrain_attention_check.py`, `tests/test_eagle3_pretrain.py`,
  `tests/test_shell_scripts.py`: checker exit/status/diagnostics, original missing
  API regression, real existing SDPA decoder attention with seven cached steps
  and backward when external FA APIs are absent, and launcher syntax coverage.
- `ENVIRONMENT.md`: diagnosis and explicit SDPA recovery instructions.

No automatic fallback, dependency installation/downgrade, or training-math
change was introduced. Default FA preference, checkpoint backend validation,
max length 2048, TTT length 7, feature capture, vocab, data, batch size,
accumulation, loss and optimizer behavior are unchanged. The supplied batch
size 64 is preserved, not certified to fit in GPU memory. Runtime recovery is
to add `PRETRAIN_ATTENTION_BACKEND=sdpa` to the original command; this explicitly
uses the already-existing backend rather than trying to adapt incompatible FA
APIs. No standard-FA package was installed on the remote host.

Validation in `D:\VDT\SpecNaacl` on the local Windows/CPU environment:

- `python -m compileall -q .`: pass.
- `pytest -q` with `BASH_BIN=D:/Git/bin/bash.exe`: **83 passed, 1 skipped**.
  The skipped two-process Gloo test has no supported transport on this Windows
  PyTorch build. Eight warnings are the existing scheduler notices plus the
  expected optional FA-unavailable warning in the SDPA regression.
- The first targeted test run failed because the newly added CPU decoder test
  invoked compiled helpers without a local MSVC compiler. The test now unwraps
  only the RMSNorm/rotary helper decorators; production compilation is untouched.
- `python scripts/check_pretrain_attention.py --backend sdpa --probe`: returns 0
  and correctly describes an explicit selection, not a passed CUDA probe.
- `git diff --check`: pass.

No full pretrain, CUDA probe or B200 throughput benchmark has been run for this
fix, and no speedup is claimed. The server's standard FA package remains absent
or incompatible until the user installs a compatible build; the recovery above
does not claim to repair that external package.

## Minimum Python version instead of patch allowlist (2026-10-02)

The only project-wide exact interpreter restriction was the allowlist in
`scripts/validate_environment.py`; pretrain, train and the benchmark invoke this
shared validator. Vendored SpecForge metadata requires Python >=3.11, not a
specific 3.12 patch. The existing 3.12 environment remains the conservative
production baseline; the project gate is now **Python >=3.12.0**, not a claim
that 3.12.0 is the lowest interpreter on which every source file could run.
Python 3.12.3 and future patch/minor versions are no longer rejected solely for
not equaling 3.12.12/3.12.13. Accepting an interpreter is separate from having
working binaries/dependencies for it.

Evidence: [CPython ABI stability](https://docs.python.org/3/c-api/stable.html)
documents compatibility across patch releases within a minor series for matching
builds (private extension APIs can still differ). The published Python metadata
for [Torch 2.13.0](https://pypi.org/project/torch/2.13.0/),
[Transformers 5.12.1](https://pypi.org/project/transformers/5.12.1/) and
[SGLang 0.5.18](https://pypi.org/project/sglang/0.5.18/) specifies lower bounds,
not a 3.12.12/3.12.13 allowlist. This evidence does not replace import checks or
actual GPU testing of the complete transitive stack.

Changed files:

- `scripts/validate_environment.py`: compare against `MIN_PYTHON_VERSION =
  (3, 12, 0)` and give recovery commands for the 3.12 series. Direct package pins,
  imports, CUDA/B200 checks and SpecForge capture API validation are unchanged.
- `.python-version`: prefer `3.12` rather than exactly `3.12.13` for uv; this file
  remains optional and is not used for runtime admission.
- `tests/test_environment.py`: admit 3.12.0/3.12.3/old and future patches/newer
  minors, reject versions below the boundary, preserve dependency/CUDA failures,
  and verify the optional uv preference is not an exact patch.
- `ENVIRONMENT.md`, `huongdanchay.md`, `DEPENDENCIES_POLICY_LAG.md`, and the
  dependency section in this report: align guidance with the lower-bound policy.

Targeted validation: **21 passed** in environment/requirements tests. Native
`python scripts/validate_environment.py --python-only` passes on the workstation's
Python 3.13.9. Tests for other interpreter versions substitute `sys.version_info`;
the complete pinned CUDA stack has not been executed under actual Python 3.12.3.
Final validation: `python -m compileall -q .` passes; `pytest -q` with
`BASH_BIN=D:/Git/bin/bash.exe` reports **95 passed, 1 skipped, 8 warnings**.
The skip is the same unsupported Gloo transport on the Windows CPU build;
warnings are the existing scheduler and optional FA availability notices.
`git diff --check` passes. No production GPU training was launched.
No dependencies, training math, optimizer settings, checkpoints or backend
selection were changed by this interpreter-policy update.

## GRPO helper import isolation fix (2026-10-03)

The new log for Qwen2.5-3B/SimpleLR/SpecNaacl passes validation with Python
3.12.3, Torch 2.13.0+cu130 and CUDA 13.0, then exits at the unconditional
`from helper.modeling_draft import Model`. The following torchrun
`ChildFailedError` is the worker-exit wrapper, not an additional root failure.
This log is not an OOM or a draft-checkpoint loading failure.

Two concrete issues in the local source were corrected:

- `grpo_speculative.py` prepended the parent directory after inserting the
  repository. A parent package named `helper` could therefore shadow the
  project's helper package. The entrypoint now moves its own root to index 0
  unconditionally and does not inject the parent. The project root was already
  present during torchrun startup; merely skipping insertion was insufficient.
- The legacy draft module was imported even for `--draft_backend=eagle3`.
  It is now imported only inside the existing legacy construction branch. The
  EAGLE adapter receives the same checkpoint/config/vocab and TTT options;
  the legacy branch still constructs and loads its original model.

The remote traceback does not prove which helper directory Python resolved or
whether its checkout was incomplete. No remote filesystem access was available;
the fix addresses both unnecessary legacy coupling and the provable local path
ordering bug rather than asserting an unobserved server package origin.

Additional changed files:

- `scripts/check_training_sources.py`: read-only stdlib source-integrity check
  for core helpers and the chosen backend. Missing files give their exact paths,
  advise syncing the project's helper directory, and explicitly distinguish it
  from a pip package. EAGLE is admitted without `modeling_draft.py`.
- `scripts/launch/train_model.sh`: run that check after Python-version admission
  and before full pinned runtime imports / torchrun / training-output writes.
- `tests/test_training_imports.py`: subprocess reproduction with a foreign parent
  helper, with/without the repo already in sys.path; execute the real AST backend
  selection branches to verify EAGLE does not import legacy while legacy still
  loads its checkpoint; incomplete-checkout errors and launcher precheck order.
- `ENVIRONMENT.md`: deployment diagnosis and recovery commands.

Targeted import/source tests: **11 passed**. Both native source-check commands
(`--backend eagle3` and `--backend legacy`) pass on the local checkout.
`python -m compileall -q .` passes; `pytest -q` with
`BASH_BIN=D:/Git/bin/bash.exe` reports **106 passed, 1 skipped, 8 warnings**.
The skip is the same unavailable Windows Gloo transport; warnings are existing
scheduler notices and expected optional FlashAttention availability. Tests use
CPU/source-isolated import/backend selection, not the complete production stack.
`git diff --check` passes. No loss, sampling,
Reflex update, optimizer, data, checkpoint/resume or GPU assignment was changed.
No real GRPO run, full pretrain or B200 benchmark was launched.

## EAGLE rollout RoPE position-ID dtype fix (2026-10-03)

The supplied Qwen2.5-3B/SimpleLR log loads the target and pretrained EAGLE draft
successfully, then fails during the first draft prefill at `cos[position_ids]`.
Dynamo reports that the indexing tensor is not an integer type. The warning
about `torch_dtype` deprecation and the optional FlashAttention warning are not
the fatal errors; torchrun's `ChildFailedError` only reports the worker exit.

Root cause in the project: prefill position IDs concatenate default floating
`torch.zeros` with integer `torch.arange`, promoting the result to floating point.
The legacy draft converts incoming positions with `.long()`, but the EAGLE flat
KV adapter previously forwarded them to SpecForge's compiled RoPE unchanged.
The indexing operation would also fail without compilation; disabling Dynamo
would not fix the invalid dtype.

Changed files:

- `helper/specualtive_generate.py`: specify `dtype=torch.long` on both parts of
  prefill position construction. Left-padding positions and token indices are
  unchanged; IDs no longer acquire a floating dtype during concatenation.
- `helper/eagle3_specforge.py`: explicitly use long default IDs and normalize
  supplied IDs to long on the draft/query device before RoPE. This restores the
  legacy input contract without modifying the actual rotary helper or model
  equations. Already-correct same-device long IDs require no conversion copy.
- `tests/test_eagle3_position_ids.py`: execute the real prefill construction
  block for bool/int masks; real tiny SpecForge EAGLE weights plus the adapter
  forward compare float32/float64/BF16/int32/int64 position inputs to a long-ID
  reference under FP32/BF16 model weights, including left-padding, flat-cache
  decode, default cache offsets, no-cache mode, backward and Dynamo RoPE tracing.
  Model/target/checkpoint loading is bypassed only for this tiny CPU fixture.

Position values, attention masks, RoPE equations, loss/TTT, feature capture,
sampling, Reflex, vocab, checkpoints/resume and optimizer behavior are unchanged.
No production compilation was disabled and no dependencies were changed. The
first targeted run passed 13 tests but the dynamic-shape Dynamo test required
an unavailable MSVC compiler for symbolic guards. That test now uses fixed
shapes with the eager graph backend; this is test-only and still exercises real
fake-tensor indexing/tracing, not an Inductor/CUDA compilation benchmark.

Validation: the targeted regression suite reports **14 passed**; final
`python -m compileall -q .` passes and `pytest -q` with
`BASH_BIN=D:/Git/bin/bash.exe` reports **120 passed, 1 skipped, 8 warnings**.
The skip is the unsupported Windows Gloo transport; warnings are the existing
scheduler and optional FlashAttention availability notices. `git diff --check`
passes. No actual CUDA/B200 GRPO training or full pretrain was run; no speedup is claimed. Deploy
the two modified helper files to the server and rerun the original train command;
pretrained draft checkpoints do not need to be regenerated for this dtype fix.

## EAGLE inference-rollout / autograd boundary fix (2026-10-03)

The supplied Qwen2.5-3B/SimpleLR log gets past generation, then fails on the
first EAGLE draft training forward at `draft_model.fc(hidden_states)` with
`RuntimeError: Inference tensors cannot be saved for backward`. Both regular
GRPO rollout and policy-lag fresh rollout run under `torch.inference_mode()`.
Their feature/target/token rows retain inference status after views or no-op
dtype conversions, but autograd must save draft inputs to compute weight
gradients. The optional FlashAttention/deprecated-dtype warnings are not the
cause; torchrun's `ChildFailedError` is the worker-exit wrapper.

Changed files:

- `helper/eagle3_specforge.py`: add `rollout_tensor_for_training`, which clones
  only inference tensors with inference mode disabled. Ordinary tensors are
  returned unchanged, preserving storage and any existing autograd graph.
- `grpo_speculative.py`: normalize input IDs, concatenated EAGLE features and
  final target hidden states at the shared `training_eagle3_specforge` boundary,
  per supervised response and after dtype conversion. Both ordinary GRPO and
  policy-lag draft-training branches use this function. Do not clone the target
  LM head or the entire rollout, and do not disable inference-mode generation.
- `tests/test_eagle3_training_inputs.py`: 21 regressions cover integer/FP32/BF16
  conversion, inherited inference contexts, ordinary-input graph preservation,
  successful embedding/linear backward, and real tiny EAGLE projection, SDPA,
  compact teacher and seven-step TTT. KL and alpha/TV/lambda LK tests compare
  identical metrics and every draft gradient against normal-tensor inputs in
  FP32/BF16, with and without a token budget. Original rollout values remain
  unchanged; target parameters receive no gradient; a zero budget skips training.
- `IMPLEMENTATION_REPORT.md`: record the cause, scoped fix and validation.

The local Windows CPU runtime has no Triton/CUDA compiler. These tests execute
the real OnlineEagle3Model source with its CUDA-only loss import replaced by
SpecForge's own reference loss function, and unwrap norm/RoPE compilation only
in the fixture. This validates the autograd boundary and objective parity, not
the production Triton kernel or CUDA execution. The first targeted test run
exposed a test-fixture issue: a uniform full-vocab teacher gave zero gradients
for acceptance-only losses. A concentrated teacher now exercises nonzero
gradients for all four objectives; no production loss code was changed.

Validation: targeted tests **21 passed**; `python -m compileall -q .` passes;
full `pytest -q` with `BASH_BIN=D:/Git/bin/bash.exe` reports **141 passed,
1 skipped, 8 warnings**. The skip is unsupported Windows Gloo transport;
warnings are existing scheduler notices and optional FlashAttention availability.
`git diff --check` passes. No real CUDA/B200 training, full pretrain or throughput
benchmark was run, and no speedup is claimed. Loss/objective, TTT length, feature
capture, sampling, Reflex, optimizer, vocab, dataset and checkpoint/resume remain
unchanged. Deploy both modified production files together and rerun the original
training command; pretrained checkpoints do not need to be regenerated.

## Reflex engineering throughput implementation (2026-10-03)

Scope: optimize trajectory-local Reflex only. Keep feature dimension/seed, random
projection values, all tree-context corrections, every root update, conditional
teacher probabilities, strict `q < p` subgradient, epsilon, LR/weight decay,
sampling/verification, EAGLE parameters/objective/TTT and checkpoint contents.
No model forward, target softmax, RNG draw, optimizer or backward is added.

Inspection found an AMP precision/overhead issue: the model autocast context
encloses Reflex, so float inputs to matmul/baddbmm can become BF16. This can
cast the entire FP32 `A` on each proposal depth, return BF16 proposals instead
of the OFF path's FP32 probabilities, and cache a BF16 `psi` incompatible with
FP32 in-place baddbmm update. Reflex now explicitly operates in FP32 independent
of model AMP, matching its stated equations and restoring zero-state/OFF parity.
This is intentionally different from the old accidental AMP rounding, not a
claim of bitwise identity to that path. Model and EAGLE training AMP are unchanged.

Optimizations and files:

- `helper/fast_lk_reflex.py`: one-time `auto|torch|triton` backend resolution,
  lazy Triton import, AMP isolation, canonical CUDA projection-cache keys, and
  stable finished-row compaction into two reusable state buffers. Only fast-state
  storage doubles; model/KV storage does not. Reset both buffers/root caches and
  timing at rollout lifecycle boundaries. Ordinary Torch remains the reference.
- `helper/fast_lk_reflex_kernels.py` (new): fused hidden conversion, seeded
  projection and normalization; tiled FP32 low-rank logit correction with the
  existing softmax; three-launch tiled conditional teacher/LK/rank-one update.
  Avoid dense compact p/mask/gradient and outer-product temporaries on this path.
  Support greedy missing-token zero-mass cases, non-contiguous root views,
  arbitrary non-power-of-two vocab sizes including full ~152k Qwen vocabularies,
  runtime strides and 64-bit batch offsets (large verification strides can exceed
  2**31). Large projection matrices use guarded FP32 cuBLAS; dimensions >64 use
  Torch in auto mode or fail clearly for explicit Triton. No runtime compilation
  failure is silently swallowed. No hot-path CUDA synchronization is introduced.
- `helper/specualtive_generate.py`: backend plumbing and effective backend in
  rollout results. Correction/update scopes, target supervision and tree
  selection/sampling logic are unchanged.
- `grpo_speculative.py`: CLI/backend forwarding, run config and final summary
  requested/effective backend metadata. Persistent training math is unchanged.
- `configs/_shared/b200_common.env`, `scripts/launch/train_model.sh`: expose
  `REFLEX_BACKEND` with auto default and record it in run metadata; fair OFF/ACTIVE
  launchers still differ only by method/reflex mode.
- `scripts/check_training_sources.py`: require the new kernel source too, so
  incomplete server deployments fail before expensive GPU model loading.
- `scripts/benchmark_reflex.py` (new): standalone synthetic OFF/Torch/candidate
  component-cycle benchmark with repeated-state numerical verification before
  timing. Report GPU/name/backend, warmed ms/cycle, added ms vs OFF, first-JIT
  verification time, probability/state error and root top-k agreement. Setup,
  target probability creation and JIT compilation are outside timed cycles. CPU
  mode is explicitly smoke-only. No default training profiler was added.
- `tests/test_fast_lk_reflex.py`: AMP/FP32 regression, exact zero-state/OFF parity,
  backend requirements/fallbacks, double-buffer lifecycle, 20-round exact Torch
  equation/gradient-state parity with greedy/sampling, decay and compaction.
  Six real CUDA/Triton tests cover BF16 model autocast, root/branch correction,
  repeated updates, strides/compaction and small/32k/full-vocab shapes.
- `tests/test_reflex_kernel_equations.py` (new): 12 source-isolated CPU tests
  execute the actual tiled kernel definitions via a small Torch-backed tl memory
  simulator; validate feature/correction/update equations, padding, strides,
  normalization, missing greedy tokens, LR=0 and decay. This is NOT validation
  of the actual Triton compiler, CUDA numerical behavior or GPU performance.
- `tests/test_reflex_benchmark.py` (new), `tests/test_shell_scripts.py`: CPU
  benchmark JSON/parity smoke and launcher backend override/fairness tests.
- `README.md`, `RUNNING.md`, `METHOD_FAST_LK_REFLEX.md`: runtime/backend/benchmark
  instructions and caveats. This report records the implementation and limits.

Numerical caveats: fused FP32 reductions need not be bit-identical to cuBLAS;
near-tie candidates may change, and no task-quality preservation or B200 speedup
has been measured here. Checkpoint loading/state/RNG restoration is unchanged,
but keep backend fixed for strictly comparable resumed runs. Zero overhead is
not promised: projection, correction, state update, dispatch, extra fast-state
storage and first-use JIT compilation still cost resources. Run CUDA parity and
the benchmark on the actual B200 before a long fused run, then compare matched
real AAL/reward/end-to-end throughput. Use `REFLEX_BACKEND=torch` if CUDA checks
fail or fused kernels are not faster on that setup.

Final validation: `python -m compileall -q .` passes. Full `pytest -q` with
`BASH_BIN=D:/Git/bin/bash.exe`: **164 passed, 7 skipped, 8 warnings**. Six skips
are the unavailable CUDA/Triton execution tests; the seventh is the pre-existing
unsupported Windows Gloo transport. Warnings are the existing optional
FlashAttention availability and scheduler notices. Targeted kernel-source tests
report **12 passed**. The small CPU benchmark smoke passes with zero probability
and state error and root top-k agreement 1; its CPU timing is not a performance
claim. Checkout-integrity check and `git diff --check` pass.

No real CUDA/B200 kernel execution, GRPO run, full pretrain or GPU benchmark was
performed. No speedup, zero-overhead behavior, full GPU numerical equivalence or
task-quality preservation is claimed. For numerical intent the kernels use
[`tl.div_rn`](https://triton-lang.org/main/python-api/generated/triton.language.div_rn.html)
for precise division rather than approximate reciprocal normalization. This
documentation check is not a substitute for the skipped CUDA/compiler tests.

## Multi-context / visited-path Reflex critical-path update (2026-10-03)

Repository actually modified: **`D:/VDT/SpecNaacl`**. Pre-change revision:
`a9d7daf8a2f186cc15ac4b5575db0e68cd3301a3`. No full pretrain, production GRPO run,
dependency installation or remote-server mutation was performed. The development
host has Python 3.13.9, PyTorch 2.9.0+cpu, no CUDA/GPU/Triton. Consequently CUDA
compilation, B200 numerical parity, real-model rollout throughput and training
speedup remain **unverified**, not inferred from CPU tests.

### Files and functions

- `helper/fast_lk_reflex_kernels.py`: multi-context `correct_logits/correct`,
  `_proposal_tiles/_proposal_merge` and `propose`, `_trace_path/trace_path`,
  `_tree_mask/tree_mask`, `_path_stats/_path_update/update_path`.
- `helper/fast_lk_reflex.py`: `FastLKReflex.propose`, native feedback pools,
  `update_visited`, cached-root reconstruction, reusable proposal/statistics/root/
  path pools, diagnostics/profile accounting and cleanup.
- `helper/tree_verification.py` (new): `PackedTree`, `pack_tree`, `VerifiedPath`,
  GPU/tensor `trace_verified_path` plus one small host-bookkeeping packet.
- `helper/specualtive_generate.py`: ACTIVE-only tensor tree construction, head
  context mapping, packed masks, GPU traversal/feedback integration, prefill
  metadata transfer, pooled masks and teacher tensor lifetime reduction.
- `grpo_speculative.py`, `configs/_shared/b200_common.env`,
  `scripts/launch/train_model.sh`: `REFLEX_FEEDBACK_SCOPE`, CLI alias compatibility,
  forwarding and run/summary metadata. No reward/loss/optimizer/TTT changes.
- `scripts/check_training_sources.py`: require the new helper before training;
  sync the ENTIRE matching source revision to the server, not just the entrypoint.
- New benchmark scripts: `scripts/benchmark_reflex_pipeline.py`,
  `scripts/benchmark_reflex_rollout.py`, `scripts/benchmark_reflex_training.sh`,
  `scripts/summarize_reflex_training.py`.
- New tests: `tests/test_reflex_tensor_path.py`, `tests/test_reflex_fused_source.py`,
  `tests/test_reflex_rollout.py`, `tests/test_reflex_cuda_pipeline.py`. Extended
  `tests/test_reflex_kernel_equations.py`, `tests/test_reflex_benchmark.py`,
  `tests/test_shell_scripts.py`.
- Updated `README.md`, `RUNNING.md`, `METHOD_FAST_LK_REFLEX.md` and this report.
  `helper/sampling.py`, the EAGLE adapter and pretraining math were inspected,
  not modified in this update.

### Data flow, verification and synchronization

Before: dense corrected FP32 logits -> dense compact q -> top-k; Python Node
allocation/parent linking; chosen/parents D2H; per-node token D2H in a thread pool;
full packed sampled-token list D2H; per-sequence Python traversal; root feedback.

After (ACTIVE only):

```text
draft head -> A-tile-reused correction/max/sum/top-k -> selected proposals
                   | native logits + psi + norm pools
packed tensor parents/tokens/context ids -> SAME packed target forward ONCE
  -> SAME target sampler ONCE -> GPU first-matching-child path (stop EOS)
  -> gather/reconstruct eligible feedback -> mean -> ONE A write per round
  -> ONE small emitted-token/packed-index packet to CPU -> existing KV bookkeeping
```

The target temperature/top-p/top-k distribution, sample-once call, confidence
selection/packing, matching-child order and EOS truncation semantics are retained.
Root mode retains immediate root feedback before traversal; visited mode defers
feedback until the authoritative path is known. There is no extra target forward
or teacher softmax. Persistent EAGLE/target training, feature capture and
checkpoint contents remain unchanged. Backend/scope should stay fixed for
strictly comparable resumes; visited feedback intentionally changes A learning.

Removed in ACTIVE: per-node `.cpu()`, candidate-parent/chosen `.tolist()`, packed
target-token `.tolist()` for traversal, CPU Node linking/traversal and transfer
thread/stream creation. Prompt mask metadata uses one bulk D2H instead of a GPU
scalar test per token. Path extraction and feedback have no host scalar reads;
CPU bookkeeping runs AFTER GPU feedback. No production `synchronize()` was added.
Explicit existing statistical-time mode still synchronizes by user request.
KV padding/pruning still requires host lists: the rollout is NOT completely
GPU-resident. The unavoidable small path transfer and existing KV compaction are
deliberately retained pending measured benefit of a more invasive rewrite.

### Kernel design and update semantics

For small D=4/8 and C<=8, correction/proposal grids are `(vocab_tile, batch)`,
not `(vocab_tile, context, batch)`. Each program loads FP32 A `[256,D]` once
outside the context loop, loads each psi and accumulates correction in FP32.
Native BF16/FP16 logits convert in registers. No context is dropped or truncated.

Fused proposal stage 1 computes tile max, sum-exp and K local winners while A
is resident. Stage 2 merges tile summaries with global max/sum-exp and produces
K ids, normalized probabilities and two FP32 normalization scalars per context.
Full corrected FP32 logits/q never reach global memory in production Triton
proposals. Tie-breaking is compact-id ascending; Torch tie order is unspecified.
Near-tie/rounding differences are allowed within numeric tolerances, not a promise
of identical seeded stochastic rollouts across backends.

The verifier keeps parents/token ids/head-context ids on GPU. One CTA per
trajectory follows first matching children, reading only ALREADY sampled target
tokens and stops at EOS. A node expanded by a draft head carries its head context
id; unexpanded leaves carry -1. The emitted target bonus at such a leaf is committed
normally but cannot provide a nonexistent draft q/psi. Counterfactual branches
are never feedback-eligible. A parent-closure async assertion rejects malformed
confidence-selected trees rather than silently changing mask semantics.

Visited q is reconstructed from cached NATIVE logits, psi and max/sum-exp, using
pre-update A. Three launches per round: compact teacher masses; LK alpha/selected
mass; mean-gradient aggregate and single A write/decay. Both statistics and update
reuse A tiles across visited slots. There is no per-depth/context update launch.
Reduction is over each trajectory's own eligible visited count, including valid
contexts whose greedy token is outside the controllable compact vocabulary (zero
gradient), excluding unexpanded/unvisited contexts. Default CUDA update has no
separate diagnostic/path-validation reduction launches; optional `validate=True`
is a debug API. The normal production path comes from the bounded verifier.

Torch/root keeps its original cached q/psi and in-place rank-one equation for
ablation. Torch/visited is the explicit dense reference for testing. Triton/root
uses the same generalized fused feedback with a preallocated root-index view.
Neither mode adds autograd, an optimizer, persistent parameters or extra RNG draws.

### Memory and dispatch

Reused across rounds: proposal partial max/sum/top-k pools; native-logit, FP32 psi
and norm feedback pools; feedback mass/statistic partials; root indices; path
buffers; flat packed-target mask pool. Mask capacity includes verification padding
and all permitted rounds, not only real-token max_length. Small returned selected
proposals/norm/alpha tensors still allocate; this is not an allocation-free API.

No full branch q is retained in Triton. Visited Torch computes a dense temporary
reference, but also caches native logits rather than full tree FP32 q. Root Torch
keeps its legacy root q for the closest ablation. Full teacher sampling probabilities
and target logits are released after feedback in ACTIVE, before the next draft.
Feedback caches need no finished-row compaction: next-round proposals overwrite
active rows. A itself retains the existing stable double-buffer compaction.

No unmeasured active-index indirection, multi-stream overlap, CUDA graph capture
or hybrid backend dispatch was enabled. `auto` remains availability-based, NOT
measured autotuning. Explicit `torch` is the reference; explicit `triton` is the
new optimized implementation and fails clearly when CUDA/Triton are unavailable.
Until the actual GPU tests/benchmark pass, use `REFLEX_BACKEND=torch` on the server.

### Correctness evidence

CPU tests cover actual kernel SOURCE equations, padded vocabulary tiles, strided
BF16/FP16 inputs, D=4/8, C=1/4/7/8, ties and selected probabilities; mean feedback
with different eligible counts, single decay, greedy compact misses, counterfactual
and unexpanded-leaf exclusion, AMP FP32 isolation, finished-row reuse and debug
validation. Tensor masks/path matching are differentially checked against Python
ancestor/matching traversal, including duplicate child tokens and EOS.

Four full production-control-flow rollouts use tiny deterministic model doubles:
greedy/sample x repeat 1/2, left padding, EOS/finished-row shrinking and returned
training features. OFF vs ACTIVE zero-state root/visited match committed tokens,
target masks, returned features/ids, acceptance counters and target forward count.
No actual model downloads or CUDA kernels are involved in those CPU tests.

Additionally, a one-time differential execution against the PRE-CHANGE file from
the revision above checked eight cases: greedy/sample x repeat 1/2 x OFF/ACTIVE
root, with LR=0.05. All generated tokens, masks, acceptance counters, returned
features/ids and target forward counts matched exactly. It is not claimed to
validate GPU compiler behavior, production model quality or B200 speed.

Real CUDA tests span B=1/3/64, C=1/4/7/8, V=16003, D=4/8, BF16/FP16, stochastic
teacher/greedy, compaction, fused probabilities, path extraction and masks. They
are SKIPPED locally. Old six root CUDA parity cases also remain in the suite.

Final rerun: `python -m compileall -q .` passed. Full `python -m pytest -q` with
`BASH_BIN=D:/Git/bin/bash.exe`: **251 passed, 103 skipped, 8 warnings** (55.18s).
102 skips require real CUDA/Triton; one is the existing unsupported Windows
Gloo transport. Warnings are existing optional FlashAttention/scheduler notices.
Targeted source/tensor/full-rollout tests: **81 passed**. Checkout source check,
shell syntax/dry runs, benchmark CLI/smokes and `git diff --check` passed.
CUDA skips are NOT counted as passes.

### Benchmark status and how to decide

| Measurement | Before/reference | New candidate | B200 result |
| --- | --- | --- | --- |
| Correction | Torch/cuBLAS | A-reused multi-context Triton | Not run |
| Normalized top-k | dense correction/softmax/top-k | tiled fused top-k | Not run |
| Feedback/update | Torch root/visited reference | 3-launch reconstruction/mean/write | Not run |
| Path extraction | tensor Torch reference | one Triton launch | Not run |
| Reflex cycle | full Torch proposal/extraction/update | full Triton equivalent | Not run |
| Production rollout | frozen FastGRPO + zero-state control | root/visited Torch/Triton | Not run |
| Real short training | 20-step FastGRPO isolated run | matched SpecNaacl isolated run | Not run |

Both small CPU pipeline smoke modes ran with B=2,V=17,target V=23,H=7,C=3,D=4,
K=3,depth=2. Proposal/state parity passed (zero error comparing Torch to Torch).
These timing values are NOT a before/after GPU comparison and NOT evidence of
speedup. Benchmark smoke and summarizer tests validate JSON schema/denominators.

`benchmark_reflex_pipeline.py` warms production APIs, validates numerical parity
BEFORE timings and reports CUDA-event AND host-wall latencies per component and
whole cycle. Preparation is outside feedback-only timing; the whole cycle includes
real cache/proposal work. `benchmark_reflex_rollout.py` loads existing production
weights/mapping/dataset, reuses the EXACT training prompt collator, performs a
zero-state Torch/OFF check, counts actual backbone forwards and reports generation
wall time, tokens/s, weighted AAL and memory. `*-zero` controls distinguish tensor
engineering gain from learned acceptance gain. Optional profiler captures ONE
additional untimed rollout; it never affects default training.

The optional training script performs 20 requested GRPO steps per method in new
isolated roots, with active/latest links isolated too. Its summarizer reports
tokens/generation-time and tokens/job-wall-time separately, GRPO steps/s, and a
wall-time ratio only when completed step counts match. Startup/reward filtering
are included, not hidden as optimizer-only timing. None of these GPU/GRPO
benchmarks was executed on this CPU host. Commands and checkpoint/environment
setup are in `METHOD_FAST_LK_REFLEX.md`.

No fastest backend has been selected from nonexistent B200 measurements. Choose
using repeated, matched real rollout/training runs AFTER CUDA parity passes;
compare reward/quality as well as AAL and wall time. A microkernel win alone is
not a SpecNaacl/FastGRPO end-to-end win. Zero overhead is not promised.

### Remaining bottlenecks / limits

- Target/draft model forwards, full target-vocabulary lm_head/temperature/top-p
  sampling, and packed target probabilities are intrinsic existing costs; no
  alternate sampler or additional target forward was introduced.
- Existing CPU KV/padding/finish bookkeeping and target/draft cache gathers/copies
  remain. One small path D2H per round remains. Default OFF retains legacy host
  tree costs, by request; do not attribute every future gain to AAL alone.
- Feature projection, native feedback caching/reconstruction, state traffic,
  three-stage feedback and kernel dispatch still cost time. Tiny root shapes
  could favor Torch/cuBLAS; no universal Triton advantage is claimed.
- First-use JIT and shape-specific compilation, persistent training/autograd,
  logging/checkpoint/data/reward work and existing GRPO allocator behavior are
  separate end-to-end costs. They were not changed to manufacture baseline gains.
- CUDA compiler/numerical/runtime behavior, race checks on real hardware and
  production quality/throughput remain deployment validation requirements.

## Reflex FP32 division JIT fix (2026-10-03)

### Root cause and minimal production change

The supplied training traceback fails in `_path_update`, reached through
`propose(root=True) -> sample_target_from_logits -> update_from_target_probs ->
_update_cached_root -> update_visited`. `count` is an integer, but Triton's
`tl.div_rn` requires both operands to be FP32. Unlike ordinary Torch division,
it does not silently promote the integer divisor. The Transformers deprecation,
FlashAttention warning and outer torchrun `ChildFailedError` are not this error.

`helper/fast_lk_reflex_kernels.py` now casts `max(count, 1)` to FP32 inside the
existing update kernel. The eligible-head mean, single decay/write per round,
LK objective, target sampling and GRPO math are unchanged. No new kernel launch,
CUDA allocation, host transfer/synchronization, target forward, or backend
fallback was added. Feedback still uses three fused launches per round. This is
a scalar cast in the existing kernel, NOT a measured throughput claim.

### Regression coverage / changed files

- `tests/test_reflex_kernel_equations.py`: CPU `div_rn` simulation now rejects
  non-FP32 operands, with tests for integer numerator/divisor. The former
  `torch.div` simulation silently promoted them and missed this JIT restriction.
- `tests/test_reflex_fused_source.py`: execute actual source with the strict
  simulator for root-only WIDTH=1 and visited paths with 1, 2 and 3 eligible
  heads, both greedy/stochastic feedback and feature dimensions 4/8.
- `tests/test_reflex_cuda_pipeline.py`: 16 additional real CUDA/JIT cases cover
  the production root proposal -> sampler -> native cached-root update path
  across three successive rounds, FP16/BF16, greedy/stochastic sampling, batches
  1/3 and feature dimensions 4/8. Check parity against Torch and retention of the
  Triton backend (no silent fallback).
- `IMPLEMENTATION_REPORT.md`: this fix and validation record.

### Validation and deployment limitation

- Before the kernel fix, the stricter simulator reproduced the offending
  division failure in the actual `_path_update` source (1 failed, stopped at
  first failure). After the fix: targeted kernel/source/CUDA suite **63 passed,
  112 skipped**.
- `python -m compileall -q .`: passed.
- Full `python -m pytest -q` with Git Bash configured: **261 passed, 119 skipped,
  8 warnings**. Skips are 118 CUDA/Triton cases and one unavailable Windows Gloo
  transport case. No full training/pretraining was run.
- Local PyTorch is 2.9.0+cpu, without CUDA. Real JIT compilation, GPU numerical
  parity and B200 throughput have NOT been verified here; no speedup or exact
  zero-cost promise is made. No slowdown-oriented fallback was introduced.

Sync the updated kernel to the server's actual SpecNaacl checkout, retain the
existing Reflex backend settings, and restart the original training process.
Triton recompiles changed kernel source; deleting its cache is not necessary.
For GPU regression validation on that server (without loading model weights):

```bash
python -m pytest -q tests/test_reflex_cuda_pipeline.py
```

## Reflex parallelism and critical-path update (2026-10-03)

This section supersedes earlier "remaining bottlenecks" notes for accepted-path
host roundtrips and one-by-one finished-row removal. The implementation and
B200 decision procedure are detailed in `REFLEX_B200_OPTIMIZATION.md`.

- Added opt-in context-parallel/tiled correction; context-parallel bitonic
  proposal selection; path-slot-parallel statistics and vectorized one-write
  LK update; Torch/Triton feature selection. The existing serial/fused kernels
  remain defaults because they won the available RTX 3090 B64/V16k synthetic
  comparison. Shape/launch sweeps are in `scripts/benchmark_reflex_strategies.py`.
- Replaced ACTIVE path token/index CPU roundtrip with one fused GPU padding
  kernel and preallocated output buffers. Only lengths and small scheduling
  metadata cross to CPU. Batched finished-row compaction applies once to all
  response-local tensors and each KV layer. A per-layer target suffix gather
  is opt-in; the former stacked path remains default. Synthetic before/after
  measurements are in `scripts/benchmark_reflex_bookkeeping.py`.
- Added opt-in side-stream LK feedback, with source/update events, allocator
  stream recording and a wait at the true A dependency (before A compaction or
  next proposal). Root feedback uses the exact same already-computed target
  distribution/token; no additional target forward or sampling. OFF retains
  its original asynchronous CPU tree transfer. Full CUDA tiny-model rollout
  tests compare side-stream and single-stream root/visited modes.
- Kept causal draft-depth and verified-tree-depth loops; those steps depend on
  their predecessors. No claim of B200 speedup, zero end-to-end regression,
  changed GRPO objective, or changed Reflex equation is made.

Validation on local RTX 3090/Torch 2.5.1/Triton 3.1: targeted optimized
strategy tests 10 passed; CPU+CUDA rollout tests 8 passed; previous kernel
source/CUDA set 175 passed; `compileall`, launcher shell syntax and `pip check`
passed. A broader local pytest run had 321 passes, but full collection is
blocked by missing local `pydantic`/`transformers`; three pre-existing dirty
`tests/test_requirements.py` cases reference absent offline-wheel files. No
full model rollout/training or B200 benchmark was run on this machine.
