# Learned OPD / historical baseline optimization — 2026-10-07

This supersedes the fixed-A/shared-baseline implementation at d9766ad.
FastGRPO now dispatches to frozen historical c3f05ad OFF: exact native softmax,
torch.topk(draft_k), original tree/Python verifier/RNG/stacked KV. No OPD engine
or scheduler/cache optimization reaches this baseline.

User explicitly accepted OPD always fused Top16, including cold; Top16 prefix
ties need NOT match historical K. Cold invariant is raw corrected logits = raw,
same full compact softmax distribution, no slow fallback. Mathematical
probabilities validated with FP32 tolerance, not false old-Torch bitwise claims.

## Implemented

- A is an nn.Parameter ON draft_model, head-aligned deterministic basis only
  at initialization; post-norm/head-input representation. Existing A checkpoint
  loaded unchanged. GPU gradient sum/weight accumulated before B_t changes.
  Cold B=0 has zero A gradient but its valid state weights still count.
  Apply at existing draft optimizer boundary before existing DDP sync.
  Pending gradient buffers checkpointed separately PER RANK for training resume.
  Older fixed-A optimizer checkpoints fail with an explicit initialization-only
  migration message rather than silently dropping/misassigning pending feedback.
  Auxiliary evaluation/analysis and frozen benchmark never accumulate A grads.
- Exact adaptive sparse/dense: ordered FP32 rank GEMM with tile B reuse,
  no FMA/TF32 association change. Sparse writes only S scalars; dense writes
  reused current-proposal workspace. SAME scan positions/normalization gives
  bitwise switch outputs. Current proposal scratch O(B*C*V), allocated once,
  not a probability cache of all rollout states. No guessed B200 crossover.
- TargetTop16 positive-only, sentinel -1/0 for absent entries; no arbitrary
  zero-probability target additions/updates. DraftTop16 preserved and tail intact.
- OPD metadata: THREE round transfers reduced to ONE fixed packet; GPU pad
  masks and position cumsums, no per-round Python padding-set walks. Necessary
  HF crop/active sampler batch host boundary remains (not claimed zero-sync).
- OPD KV suffix gathered into reused small per-head scratch, copied in-place
  and cropped. Accepted history prefix never copied after verification; no
  stacked all-layer/full-history concat. Native HF next update creates normal
  contiguous KV before attention. Finished batch copy still remains.
- Native target/draft transformer work unchanged per verification/expansion;
  no extra forwards/backward/full-vocab KL backward. Shared B local/reset,
  no per-response state or round all-reduce. Two reusable dependency events,
  late wait, no new profiling/default synchronization.
- Existing per-step CSV/JSONL definitions preserved; added cumulative/delta
  sparse/dense root-round usage and existing active-row mean.
- All 14 paired .sh export train-projector/mode/profile knobs. Benchmark is
  HISTORICAL FastGRPO vs optimized OPD, no shared Top16 baseline.
  Crossover tuner fingerprints GPU/Torch/Triton/CUDA/kernel and exact geometry.
  Untuned geometries stay sparse, incompatible profiles fail clearly.

## Evidence actually measured (NOT B200/full-model)

RTX3090 / Torch2.5.1+cu124 / Triton3.1 / Python3.10 existing test environment.
Final focused suite:107 passed,2 skipped (CPU stream cases),1 torch.load warning.
Revised 26-test suite
covers actual historical dispatch unchanged/no OPD constructor, positive-only
teacher1/3/8 support, sparse/dense/edge ties/full S oracle and bitwise switching,
GPU adaptive counters, learned A gradient/optimizer boundary/checkpoint,
one-packet scheduling and in-place KV suffix parity, per-rank pending-gradient
checkpoint restoration. Earlier broader focused run:127 passed,2 skipped.

Final full pytest attempt:189 passed,2 skipped,1 torch.load warning,
12 failures/37 setup errors: missing transformers for actual SpecForge native
training/position tests and three pre-existing missing offline packaging files.
These tests retained; whole-suite PASS or pinned-stack native run NOT claimed.
compileall/shell syntax/source integrity/CLI dry-run/diff checks/pip check pass.

Controlled component benchmark at V32768/H2048/r8, BF16,10 samples per median:
Final isolated run: B8,C8,S32768 sparse0.5033ms vs dense0.3363ms;
S8192 sparse0.3649 vs dense0.3363. B8,C1,S32768 sparse0.2642 vs dense0.2423.
All measured sparse/dense values/IDs
bitwise identical. Component speed is NOT generation speed or acceptance gain;
profiles from this GPU are not B200 profiles. Raw JSON/logs in
validation/opd_20261007, authoritative final component profile:
`crossover_isolated_rtx3090.json`. Do not use `crossover_final_rtx3090.json`
(concurrent tests) as a performance profile. The older archived profile also
has a pre-final kernel fingerprint; it is historical evidence, not runtime config.

Configured production target3B config / pretrained / simplelr data absent here.
Actual asset check fails clearly. No B200 model benchmark, full policy/draft
training, fake data/model or fabricated AAL/tokens/s result generated.

## Remaining bottlenecks / limits

One host scheduling boundary required by native dynamic batch/HF crop and exact
sampling; cannot defer it without unverified fixed-batch/RNG changes. Batch KV
compaction and HF next-cache append still copy history when necessary. Full
compact normalization/teacher scan, selected head rows, A gradient GEMM and
side-stream contention remain measurable costs. S may approach V, now exact
measured dense crossover available. Canonical dense GEMM prioritizes exactness;
cuBLAS/TensorCore speed claims require a separate numerical proof.

A initialized from a learned head is NOT itself trained until optimizer steps
occur. Use an OPD-trained checkpoint to measure learned A. Frozen sweep does not
silently train A. Default sparse/stream0/LR0.01 is untuned conservative; no claim
OPD always beats historical FastGRPO. Preserve/report negative AAL or cost>benefit.

## Commands / files

All commands and paths in huongdanchay.md; detailed algorithm METHOD_OPD_REFLEX.md.
Run scripts/tune_opd_proposals.sh on ACTUAL B200, export its real profile,
then sweep scripts/sweep_opd_reflex.sh and validate held-out end-to-end results.

Created/modified:
- `METHOD_OPD_REFLEX.md`
- `README.md`
- `configs/_shared/b200_common.env`
- `grpo_speculative.py`
- `helper/eagle3_specforge.py`
- `helper/opd_reflex.py`
- `helper/opd_reflex_kernels.py`
- `helper/specualtive_generate.py`
- `helper/tree_kernels.py`
- `huongdanchay.md`
- `scripts/benchmark_opd_reflex.py`
- `scripts/check_training_sources.py`
- `scripts/launch/train_model.sh`
- `tests/opd_fixtures.py`
- `tests/test_opd_contracts.py`
- `tests/test_opd_reflex.py`
- `train_llama31_8b.sh`
- `train_llama31_8b_fastgrpo.sh`
- `train_qwen25_14b.sh`
- `train_qwen25_14b_fastgrpo.sh`
- `train_qwen25_1p5b.sh`
- `train_qwen25_1p5b_fastgrpo.sh`
- `train_qwen25_3b.sh`
- `train_qwen25_3b_fastgrpo.sh`
- `train_qwen25_7b.sh`
- `train_qwen25_7b_fastgrpo.sh`
- `train_qwen3_1p7b.sh`
- `train_qwen3_1p7b_fastgrpo.sh`
- `train_qwen3_4b.sh`
- `train_qwen3_4b_fastgrpo.sh`
- `helper/historical_fastgrpo.py`
- `helper/opd_scheduling.py`
- `scripts/tune_opd_proposals.py`
- `scripts/tune_opd_proposals.sh`
- `tests/test_opd_revision.py`
- `IMPLEMENTATION_REPORT.md`, validation/opd_20261007 evidence.
No model/data/checkpoints/venv/sibling code deleted or altered.
