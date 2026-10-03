# Reflex optimization and B200 decision gate

This change keeps FastGRPO verification, target sampling, GRPO loss, compact
EAGLE vocabulary and the LK update equation unchanged. The target is still
sampled once. `METHOD=fastgrpo` uses the same rollout implementation with
Reflex off; `METHOD=specnaacl` adds only Reflex state/proposals/feedback.

## What is implemented

- Correction candidates: existing A-reuse/serial-context, context-parallel,
  and tiled-context Triton kernels. The last two do not loop over context.
- Proposal candidates: existing fused tile-local top-K, context-parallel
  bitonic sort (local and global), Triton correction plus CUDA softmax/top-K,
  and Torch baddbmm plus softmax/top-K. The sort candidate has no sequential
  K loop; the existing fused candidate is retained where faster.
- Visited-path candidates: existing slot-serial kernels and independent
  `(vocab_tile, path_slot, batch)` statistics plus vectorized P reduction in
  the final one-read/one-write A update. No sequential update of A by slot.
- Feature projection: Triton and Torch FP32 strategies. Mixed-precision GEMM
  is *not* selected because it changes the FP32 feature semantics.
- Active accepted-path padding is now one GPU Triton kernel. Only variable
  path lengths and small scheduling/padding metadata reach CPU; committed
  token/KV indices do not go CPU → GPU each round. The causal tree-depth loop
  remains necessary. No extra target forward/softmax/top-K was added.
- All finished responses are filtered once with shared keep indices. Target
  and draft KV are gathered once per layer, rather than once per finished
  response per layer. Existing stacked target suffix gather is retained as
  default; an opt-in per-layer alternative is available.
- Optional `REFLEX_UPDATE_STREAM=1` schedules LK feedback on a side stream
  after target sampling/path trace. Accepted gather, padding, cache handling,
  and committed-token draft forward may overlap; an event is waited on before
  A compaction or the next proposal. This is off by default until B200
  end-to-end measurement shows a win. No global synchronization is added to
  the default production path. The legacy OFF CPU tree transfer still uses
  its existing executor/stream; ACTIVE has no CPU tree nodes to transfer.

The kernel-selection defaults are deliberately conservative:

```text
REFLEX_PROPOSAL_STRATEGY=fused
REFLEX_CORRECTION_STRATEGY=serial
REFLEX_FEEDBACK_STRATEGY=serial
REFLEX_FEATURE_STRATEGY=auto
REFLEX_UPDATE_STREAM=0
KV_GATHER_STRATEGY=stacked
```

`REFLEX_PROPOSAL_STRATEGY` accepts `fused|sort|hybrid|torch`;
`REFLEX_CORRECTION_STRATEGY` accepts `serial|parallel|tiled` (used by `hybrid`);
`REFLEX_FEEDBACK_STRATEGY` accepts `serial|parallel`; and
`REFLEX_FEATURE_STRATEGY` accepts `auto|triton|torch`. These flags do not
change checkpoint format. For a fair method comparison, set the same
`KV_GATHER_STRATEGY`, model, draft, dataset, seed and generation config in
both method launchers.

## B200 commands (offline, existing local model/draft/data)

Run from `/workspace/storage-shared/nlp/minhpn19/SpecNaacl` in the configured
Python environment. The scripts do not download dependencies or checkpoints.

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
mkdir -p outputs/benchmarks
python -m pytest -q tests/test_reflex_optimized_strategies.py tests/test_reflex_cuda_pipeline.py tests/test_reflex_rollout.py
python scripts/benchmark_reflex_strategies.py --batch 64 --vocab 16000 --target-vocab 151936 --hidden 2048 --contexts 8 --dim 8 --topk 8 --path-length 6 --warmup 20 --iterations 100 --sweep --output outputs/benchmarks/reflex_strategies_b200.json
python scripts/benchmark_reflex_bookkeeping.py --batch 64 --width 6 --past 512 --layers 16 --heads 8 --head-dim 128 --finished 16 --warmup 20 --iterations 100 --output outputs/benchmarks/reflex_bookkeeping_b200.json
```

Repeat strategy benchmarks for `--contexts 1`, `4`, `7`, `8`, top-K `4`, `7`,
`8`, D `4`/`8`, path length `1`–`6` and actual batch sizes. The JSON files
contain parity and CUDA-event primitive timings or synchronized wall timings.
`--sweep` compiles many kernels and is a separate one-time benchmark, never
performed in production. `torch.profiler` in the real rollout benchmark can
inspect occupancy/register pressure and bandwidth; no such profiler runs in
the training hot path.
For target verification/draft-forward phase breakdown, rerun the real rollout
command below with `--component-timing`; its synchronizations make that run
diagnostic-only. `--profile-trace outputs/benchmarks/reflex_trace.json` on an
untimed extra rollout exposes path trace, statistics and A-update kernel times.

Use the existing *real frozen-model* benchmark with the local 3B checkpoint
before selecting any candidate. Supply the already-created EAGLE-3 files:

```bash
PRETRAIN_ROOT=outputs/pretrain/qwen25_3b
MODEL=/workspace/storage-shared/models/Qwen2.5-3B-Instruct
DATA=/workspace/storage-shared/nlp/minhpn19/data/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet
python scripts/benchmark_reflex_rollout.py --target-model "$MODEL" --draft-config "$PRETRAIN_ROOT/latest_draft_config.json" --draft-checkpoint "$PRETRAIN_ROOT/latest_checkpoint" --vocab-mapping "$PRETRAIN_ROOT/latest_vocab_mapping.pt" --dataset-path "$DATA" --batch-size 8 --responses 8 --modes fastgrpo,triton-root,triton-visited_path --warmup 2 --iterations 5 > outputs/benchmarks/rollout_baseline_b200.json
python scripts/benchmark_reflex_rollout.py --target-model "$MODEL" --draft-config "$PRETRAIN_ROOT/latest_draft_config.json" --draft-checkpoint "$PRETRAIN_ROOT/latest_checkpoint" --vocab-mapping "$PRETRAIN_ROOT/latest_vocab_mapping.pt" --dataset-path "$DATA" --batch-size 8 --responses 8 --modes triton-visited_path,triton-visited_path-stream --warmup 2 --iterations 5 --reflex-proposal-strategy sort --reflex-feedback-strategy parallel --kv-gather-strategy per_layer > outputs/benchmarks/rollout_candidate_b200.json
```

If the target has an adapter, pass `--target-adapter /existing/path` to both
commands. Hold all other options and seeds fixed. Compare generated tokens/s,
model forward count, AAL and quality; same RNG seed does not imply identical
generated responses when proposals differ. Run the short real GRPO benchmark
only after rollout parity and throughput pass. Do **not** launch full training
from this benchmarking document.

To run the conservative default on the offline B200 machine:

```bash
DATASET=dapo MODEL_KEY=qwen25_3b bash scripts/run_specnaacl.sh
```

If B200 measurements select an alternative, export only the winning flags,
for example `REFLEX_PROPOSAL_STRATEGY=hybrid REFLEX_CORRECTION_STRATEGY=parallel`
before the same command. Keep `REFLEX_PROFILE=0`, `REFLEX_DIAGNOSTICS=0`, and
`STATISTICAL_TIME=False` for production timing. Output goes to the unique
`outputs/train/qwen25_3b/<run_name>/` printed by the launcher.

## Local validation and limits

On the available RTX 3090 (Torch 2.5.1+cu124, Triton 3.1.0), B64/C8/V16k/D8
synthetic medians were: correction serial 0.134 ms vs parallel 0.158 ms and
tiled 0.217 ms; proposal fused 0.476 ms vs sort 0.796 ms and best hybrid
0.639 ms; feedback serial 0.338 ms vs parallel 0.682 ms. Therefore the
parallel candidates are **not** production defaults. At B64/path width 6,
fused GPU padding plus minimal metadata took 0.18–0.19 ms vs legacy host
roundtrip/padding 0.30–0.31 ms; batched finished-row KV compaction took about
3.97 ms vs 46.3 ms for 16 one-by-one removals. In a synthetic target suffix
gather, per-layer took 4.73 ms vs stacked 7.55 ms; it is opt-in until real
B200 rollout confirms the whole-cache benefit. These are 3090 synthetic
measurements, **not B200 or end-to-end training speedups**.

Fixed-slot indirection, complete elimination of CPU scheduling metadata,
automatic B200 shape tables, and full model profiling are not claimed.
Only target-device B200 measurements can establish zero regression or a
training wall-clock gain.
