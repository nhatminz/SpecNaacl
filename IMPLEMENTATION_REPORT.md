# OPD Reflex implementation report — 2026-10-06

Only METHOD=fastgrpo/opd_reflex remains; LK implementation is retired. Existing
model/data/pretrain/output paths and dependency pins are unchanged. No full
training, Internet download or sibling-repo edits performed.

## Algorithm / hot path

Native unchanged FastGRPO GRPO/rewards/optimizer/update schedule and real
SpecForge EAGLE3 architecture/feature capture/loss/unrolling inherited.

Fixed checkpoint-persistent A:[H,8], ONE shared rollout-local B:[compactV,8].
Visited + expanded final-tree one-hop rejected siblings, GPU masks, no recursive
rejected descendants. Existing post-sampling teacher probabilities conditioned
on compact vocabulary. Top16 union + tail forward KL, exact prescribed union
coordinate gradient q-p; weighted batched SGD. No Adam/autograd/all-reduce on B.
A is fixed, outside optimizers. Both A/draft resume/export save/load verified.
B changes only after verification and is reset at rollout completion/start.

Sparse full-vocab proposal: raw inactive scan via bitmap + ONLY S active r-dots
+ exact global merge. Negative corrections included, no duplicates/ANN/gates.
One CUDA Top16 serves tree K and feedback. No all-state full-vocab probability
cache, no V*r feedback gradient, no per-response adapter. Exact cancellation
prunes zero rows; no magnitude threshold or eviction.

Cache normalized head inputs/u/Top16 IDs,q/normalization in reusable buffers.
Selected output-head rows reconstruct teacher-only union q, not another draft
transformer. Teacher workspace TOTAL capacity bounded by verification_capacity+B,
not B*max_verification_num. Two stream dependency events per rollout, late wait
before next proposal, transient tensor recording only. Mandatory metrics share
one end-rollout packet. No new production cuda.synchronize.

14 model launchers export knobs, default simplelr, target/draft LR1e-5, batch8,
accum4, responses8, online draft tokens2048/log1. Stream0/LR0.01 are conservative
UNTUNED defaults; only real end-to-end B200 data can choose performance winner.
Profiling/diagnostics OFF. Fixed cuda-tile validator namespace cuda.tile only;
no dependency version relaxation.

## Exactness qualifications

CUDA OFF/B0 OPD identity tested bitwise through proposals/tree/verifier/tokens,
history, CPU/CUDA RNG and target/draft forward counts. CPU golden rollouts match
prechange tokens/masks/history/counts. Positive OPD counts can legitimately
change with verification rounds: zero EXTRA means inherited work per round,
not forced equal totals despite different AAL.

Shared fused CUDA normalization and low-ID tie rule differ from historical
Torch softmax/topk. Both production modes use the SAME engine; no old-Torch
bitwise claim. CPU oracle retains Torch K-dependent tree topk via a DEBUG-only
second topk and is blocked on CUDA.

Selected BF16 head at H32/2048 compared to native dense rows with explicit
rtol1%/atol1e-6 q tolerance; independent reductions can round at BF16 boundaries.
This auxiliary tolerance NEVER relaxes exact target/proposal/B0 identity.
Shared FP32 atomic gradients checked with dense oracle tolerance, not falsely
claimed bitwise positive-update reproducibility. TEST-only autograd verifies
union q-p derivative of forward KL INCLUDING tail.

## Checks actually performed

Local RTX3090 / Torch2.5.1+cu124 / Triton3.1 / Python3.10 test interpreter
(not pinned B200 stack; existing installed environments untouched).

- Relevant suite: **102 passed,2 skipped** (CPU stream-only cases).
- Full pytest: **163 passed,2 skipped,12 failed,37 setup errors**.
  Nine failures/37 errors: missing transformers in actual SpecForge pretrain,
  position and training-input tests. Three retained pre-existing packaging tests
  refer to missing requirements-bootstrap.txt/requirements-external.txt/
  scripts/build_offline_wheelhouse.sh. Not removed or rewritten to hide failures.
- compileall . / authored shell syntax / diff check: PASS.
- Real entrypoint parser fed both actual launcher commands: PASS.
- All paired model dry runs/overrides, source integrity, sweep CLI dry run: PASS.
- pip check of test interpreter: No broken requirements found.
- Configured target3B config, pretrained/data assets absent locally. Production
  asset check fails clearly; no fake model/data or native train substituted.

Validation logs: validation/opd_20261006. No B200/full-checkpoint AAL or tok/s
available. Controlled unit test shows future fixed-state teacher mass increases
and target token enters Top2 without transformers: correctness/capability ONLY,
not claimed production benefit.

## Telemetry / benchmark / limitations

Definitions in METHOD_OPD_REFLEX.md. Exact per-step cumulative differences,
accepted_length/SEQUENCE verification rounds, target bonus included, prefill
token excluded. One timing.csv row per completed inherited GRPO label in both
methods, no moving averages or unweighted batch AAL. Same logging schema.

Frozen sweep warms measured seed/prompt schedules, measures end-to-end wall,
tokens/s, weighted AAL, rounds/acceptance/memory/counters, keeps negative results.
Separate profile replay excluded from throughput and cannot evict warmed cache.
recommendation=null unless some OPD config improves BOTH AAL and throughput.
Same seed is not assumed to imply identical responses. Plot exact step counters.

S can approach V. Remaining cost: full scan/top-k/normalization, compact teacher
extraction, selected head rows, side-stream bandwidth contention, HF eager/KV
and inherited host scheduling metadata. No guessed speedup or async advantage.

Memory: A O(Hr), B O(Vr), bitmap O(V/32), active-ID reserved O(V), cached head
inputs O(B*expandedContexts*H), proposal summaries O(B*C*ceil(V/256)*TopK),
teacher summaries O((verification_capacity+B)*ceil(V/256)*TopK). No B*V*r state
or allStates*V probability cache.

## Files created/changed

- `README.md`
- `RUNNING.md`
- `configs/_shared/b200_common.env`
- `grpo_speculative.py`
- `helper/eagle3_specforge.py`
- `helper/method_config.py`
- `helper/sampling.py`
- `helper/specualtive_generate.py`
- `helper/step_metrics.py`
- `huongdanchay.md`
- `scripts/check_training_sources.py`
- `scripts/launch/train_model.sh`
- `scripts/run_fastgrpo_fair.sh`
- `scripts/validate_environment.py`
- `tests/test_rollout_history.py`
- `tests/test_sampling.py`
- `tests/test_shell_scripts.py`
- `tests/test_step_metrics.py`
- `tests/test_training_imports.py`
- `train_fastgrpo.sh`
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
- `METHOD_OPD_REFLEX.md`
- `helper/opd_reflex.py`
- `helper/opd_reflex_kernels.py`
- `helper/tree_kernels.py`
- `scripts/benchmark_opd_reflex.py`
- `scripts/plot_opd_aal.py`
- `scripts/run_opd_reflex.sh`
- `scripts/sweep_opd_reflex.sh`
- `tests/fastgrpo_golden.json`
- `tests/opd_fixtures.py`
- `tests/test_opd_contracts.py`
- `tests/test_opd_reflex.py`
- `train_opd_reflex.sh`
- `IMPLEMENTATION_REPORT.md` and `validation/opd_20261006/`.

## Retired LK-only files (recoverable from Git history)

- `helper/fast_lk_reflex.py`
- `helper/fast_lk_reflex_kernels.py`
- `tests/test_fast_lk_reflex.py`
- `tests/test_reflex_rollout.py`
- `tests/test_reflex_fused_source.py`
- `tests/test_reflex_optimized_strategies.py`
- `tests/test_reflex_kernel_equations.py`
- `tests/test_reflex_benchmark.py`
- `tests/test_reflex_tensor_path.py`
- `tests/test_reflex_cuda_pipeline.py`
- `scripts/run_specnaacl.sh`
- `scripts/benchmark_reflex.py`
- `scripts/benchmark_reflex_pipeline.py`
- `scripts/benchmark_reflex_training.sh`
- `scripts/benchmark_reflex_strategies.py`
- `scripts/benchmark_reflex_bookkeeping.py`
- `scripts/benchmark_reflex_rollout.py`
- `scripts/summarize_reflex_training.py`
- `METHOD_FAST_LK_REFLEX.md`
- `REFLEX_B200_OPTIMIZATION.md`

No model/data/weights/outputs/venv deleted.
