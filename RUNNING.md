# Running on the offline B200 machine

The project is expected at:

```text
/workspace/storage-shared/nlp/minhpn19/SpecNaacl
```

Models and datasets must already exist locally. Runtime scripts set Hugging Face
offline variables and never run `git clone`.

## Pretrain an EAGLE-3 draft

Every `pretrain_<model>.sh` calls the same generic launcher and then the real
vendored SpecForge offline pipeline: data conversion, target feature capture,
compact vocabulary construction, and SpecForge EAGLE-3 training-time unrolling.

Editable/overrideable variables are `MODEL`, `PRETRAIN_DATASET`,
`PRETRAIN_DATASET_PATH`, `PRETRAIN_LR`, `PRETRAIN_BATCH_SIZE`,
`PRETRAIN_MAX_LENGTH`, `PRETRAIN_EPOCHS`, `NPROC_PER_NODE`, `CUDA_VISIBLE_DEVICES`,
`RESUME`, and `RUN_NAME`.

Supported wrappers:

```text
pretrain_qwen25_1p5b.sh  pretrain_qwen25_3b.sh
pretrain_qwen25_7b.sh    pretrain_qwen25_14b.sh
pretrain_qwen3_1p7b.sh   pretrain_qwen3_4b.sh
pretrain_llama31_8b.sh
```

`PRETRAIN_DATASET` accepts `sharegpt`, `gsm8k`, `simplelr`, or `dapo`. Local
parquet rows are deterministically converted into the conversation format that
SpecForge consumes. `RESUME=auto` reuses the active incomplete run; an explicit
run directory can be passed through `RESUME=/path/to/run`.

## Paired model training (FastGRPO vs SpecNaacl)

All seven models have paired launchers (14 scripts): the ordinary file binds
`METHOD=specnaacl`, and the `_fastgrpo.sh` file binds `METHOD=fastgrpo`.
The paired launchers use local DAPO-Math-17k by default, target/draft LR
`1e-5`, batch size `8`, accumulation `4`, and 8 responses per prompt. They
require a pretrained EAGLE-3 draft in `outputs/pretrain/<model>/latest_*` (or
explicit `DRAFT_CHECKPOINT`, `DRAFT_CONFIG`, `VOCAB_MAPPING`).

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
# Choose the pair for your model; do not run every model unless intended.
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_1p5b.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_1p5b_fastgrpo.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_7b.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_7b_fastgrpo.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_14b.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_14b_fastgrpo.sh
CUDA_VISIBLE_DEVICES=0 bash train_llama31_8b.sh
CUDA_VISIBLE_DEVICES=0 bash train_llama31_8b_fastgrpo.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen3_4b.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen3_4b_fastgrpo.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_3b.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_3b_fastgrpo.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen3_1p7b.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen3_1p7b_fastgrpo.sh
```

Append `--max_grpo_steps 2` for a short run or set `DRY_RUN=true` to inspect
the command without touching checkpoints. Override `MODEL`, `DATASET_PATH`,
`TARGET_ADAPTER`, `DRAFT_CHECKPOINT`, `TARGET_LR`, `DRAFT_LR`, `BATCH_SIZE`,
`ACCUMULATION_STEPS`, `DRAFT_ACCUMULATION_STEPS`, `RESPONSES_PER_PROMPT`,
`GEN_MAX_LENGTH`, `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, or `RESUME` through
the environment.

The four previously unpaired wrappers (Qwen2.5-1.5B/7B/14B and Llama-3.1-8B)
now share these defaults rather than their old GSM8K/LR settings. Model paths,
matching draft checkpoint links, data resolution and unique output naming are
unchanged. For a smaller rollout batch on 14B, use the same override in both
methods, for example:

```bash
BATCH_SIZE=4 ACCUMULATION_STEPS=8 CUDA_VISIBLE_DEVICES=0 bash train_qwen25_14b.sh
BATCH_SIZE=4 ACCUMULATION_STEPS=8 CUDA_VISIBLE_DEVICES=0 bash train_qwen25_14b_fastgrpo.sh
```

All files use the shared launcher and model config. Only the method/Reflex
toggle differs within each pair. Pretraining files remain single-method
`pretrain_<model>.sh` files; pretraining itself has no FastGRPO/SpecNaacl switch.

SpecNaacl defaults to `REFLEX_BACKEND=triton`, `REFLEX_FEEDBACK_SCOPE=root`,
`REFLEX_UPDATE_STREAM=1`, matching `triton-root-stream`. FastGRPO creates no
Reflex update stream. Both use batched SpecForge EAGLE-3 training. Control VRAM
with `DRAFT_TRAIN_MAX_BATCH_SIZE=8`, `DRAFT_TRAIN_MAX_TOKENS=2048`, and
`DRAFT_TRAIN_MAX_PADDING_RATIO=1.25`; `DRAFT_TRAIN_MODE=per_response` is the
measured fallback. `DRAFT_TRAIN_PROFILE=1` logs optional timings.
`DRAFT_TRAIN_MAX_TOKENS` limits padded tokens per microbatch, not the total
supervised tokens retained from a rollout. An oversized singleton is still
legal, as before; no response is truncated just to meet this packing limit.
`REFLEX_PROFILE=1` reports feature projection, proposal/correction, feedback
update and estimated stream overlap; turn it off for throughput runs.

For B200 selection, run the opt-in short real-training benchmarks:

```bash
MODEL_KEY=qwen25_3b DATASET=dapo BENCHMARK_STEPS=3 bash scripts/benchmark_online_draft_training.sh
MODEL_KEY=qwen25_3b DATASET=dapo BENCHMARK_STEPS=3 bash scripts/benchmark_reflex_training.sh
python3 scripts/benchmark_reflex_rollout.py \
  --target-model /workspace/storage-shared/models/Qwen2.5-3B-Instruct \
  --draft-config outputs/pretrain/qwen25_3b/latest_draft_config.json \
  --draft-checkpoint outputs/pretrain/qwen25_3b/latest_checkpoint \
  --vocab-mapping outputs/pretrain/qwen25_3b/latest_vocab_mapping.pt \
  --dataset-path /workspace/storage-shared/nlp/minhpn19/data/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet \
  --modes fastgrpo,triton-root,triton-root-stream
```

If batched training is slower or uses too much memory, set
`DRAFT_TRAIN_MODE=per_response`; if stream overlap loses throughput, set
`REFLEX_UPDATE_STREAM=0`. B200 speedup remains unmeasured here.
Each short training benchmark writes both run summaries and
`benchmark_report.json` under its printed `outputs/benchmarks/...` directory.

## Generic training wrappers

The matching `train_<model>.sh` wrappers expose `MODEL`, `DATASET`,
`DRAFT_CHECKPOINT`, `TARGET_LR`, `DRAFT_LR`, `BATCH_SIZE`,
`ACCUMULATION_STEPS`, `GEN_MAX_LENGTH`, `MAX_PROMPT_LENGTH`, `NUM_EPOCHS`,
`NPROC_PER_NODE`, `REFLEX_FEATURE_DIM`, `REFLEX_LR`, and
`REFLEX_WEIGHT_DECAY`, and `REFLEX_BACKEND=auto|torch|triton`. `METHOD=fastgrpo` disables Reflex and
`METHOD=specnaacl` enables it; `DATASET` accepts `gsm8k`, `simplelr`, or `dapo`.
The paired per-model filenames bind METHOD explicitly; use the matching suffix
instead of overriding METHOD on those files. To select METHOD through the
environment, use `MODEL_KEY=<key> METHOD=<method> bash scripts/launch/train_model.sh`
or the generic `scripts/run_fastgrpo_fair.sh` / `scripts/run_specnaacl.sh` wrappers.

`REFLEX_FEEDBACK_SCOPE=root|visited_path` defaults to root. Example:
`REFLEX_BACKEND=torch REFLEX_FEEDBACK_SCOPE=visited_path bash train_qwen25_3b.sh`.
Visited feedback uses only entered draft-head contexts and averages their gradients
before one update per round. Qualify Triton with CUDA tests and the component/real
rollout benchmarks in [METHOD_FAST_LK_REFLEX.md](METHOD_FAST_LK_REFLEX.md) before
long GPU runs; `auto` chooses by availability, not by measured throughput.

See [METHOD_FAST_LK_REFLEX.md](METHOD_FAST_LK_REFLEX.md) for backend requirements,
numerical caveats and `scripts/benchmark_reflex.py`. Backend changes do not alter
checkpoint contents, but fused reduction rounding can change near-tie proposals;
keep the same backend for strict like-for-like resumed comparisons.

If `DRAFT_CHECKPOINT`, `DRAFT_CONFIG`, and `VOCAB_MAPPING` are omitted, the
launcher uses the matching model's `outputs/pretrain/<model>/latest_*` links.
Missing or incompatible weights fail; random initialization occurs only with an
explicit `DRAFT_INITIALIZATION_MODE=random` and still requires a compatible
config/mapping.

`NPROC_PER_NODE>1` uses `torchrun` plus synchronous gradient all-reduce. Each
rank holds a full target and draft model; FSDP is not used, so custom EAGLE KV
cache behavior is preserved. Policy-lag side-branch analysis remains explicitly
single-process.

`RESUME=auto` restores `checkpoints/resume/latest.pt`. Checkpoints contain target
LoRA, EAGLE state, both optimizers, accumulation gradients/counters, epoch/step,
per-rank Python/NumPy/Torch/CUDA RNG, the original world size, and cumulative
elapsed time. Exact resume rejects a different world size. There are currently
no schedulers; both scheduler fields are recorded as `null`. Logged and final
token/acceptance/reward/loss counters are reduced across ranks; throughput is
global rollout tokens divided by cumulative job wall time.

## Outputs

```text
outputs/pretrain/<model>/<unique_run>/
outputs/train/<model>/<unique_run>/
```

Run names include model, dataset, method, seed, UTC timestamp and an
8-character UUID. Each run has `checkpoints/`, `logs/`,
`config_resolved.yaml`, `summary.json`, and `summary.txt`. Training logs are
`logs/metrics.jsonl` and `logs/timing.csv`; detailed CUDA synchronization remains
disabled by default. `LOG_INTERVAL=1` is the shared default. Authoritative
JSONL (`phase=target_train`) and CSV metrics always contain one row per distinct
inherited GRPO `step`, regardless of `LOG_INTERVAL` (which controls progress
display). FastGRPO can perform several target optimizer updates with the same
step label; these are accumulated, and the row is written when the label
advances or the run finishes. Skipped labels are not fabricated.

Each row has `step_*` and `cumulative_*` wall/generation/target-training/
draft-training seconds, rollout-token counts, generation throughput, AAL and
draft acceptance rate. Step AAL is the exact difference in accepted-length
and sequence-verification-round counters, **not** a mean of batch ratios.
It retains FastGRPO's accepted length including the mandatory root/bonus token;
draft acceptance rate excludes that token and divides accepted draft tokens by
proposed draft tokens. Responses filtered out of the reward buffer still count
as rollout work. Step throughput divides step rollout tokens by generation
seconds, not wall seconds. Legacy `tokens_per_s` remains wall-clock throughput.

Wall/generation clocks are monotonic. Generation ends after the required output
ID transfer to CPU. Target/draft training uses CUDA stream events, including
optimizer work, read after the existing aggregate-metric transfer; CPU fallback
uses monotonic time. These GPU intervals include enqueue gaps and do not add
global synchronization or measure side-branch analysis as target training.
Across ranks, counters are summed and cumulative phase times use the maximum
rank; step times are differences of those job counters. Per-step memory fields
`gpu_allocated_gb`, `gpu_reserved_gb`, `gpu_peak_allocated_gb`, `gpu_free_gb` use
GiB, with max allocated/reserved/peak and minimum free across ranks. Peak is
since process start, not reset each step; logging never flushes the allocator.

Histories use owned per-response append buffers, initially reserving at most
256 extra slots, with geometric growth only when necessary. Each round copies
only the new chunk in the ordinary case; completed rows are cloned and their
buffer released without copying active histories at compaction. Padded token
order and online EAGLE3 inputs remain unchanged. Checkpoints include metric
baselines/pending step state; resume rewinds later step rows with backup files,
so continuing a repeated label does not duplicate it. Old CSV schemas are
upgraded with a backup; historical missing per-step values are left blank.

`RESPONSES_PER_PROMPT=8` (alias `REPEATED_GENERATE_NUMS=8`) is shared by both
methods. If both are set, `RESPONSES_PER_PROMPT` takes precedence.

Plot selected runs by editing `RUN_DIRS` in `scripts/plot_training_time.sh` or:

```bash
bash scripts/plot_training_time.sh /absolute/run1 /absolute/run2
```

For the opt-in Reflex parallel kernels, GPU path bookkeeping, side-stream
overlap, and B200 benchmark/selection commands, see
[REFLEX_B200_OPTIMIZATION.md](REFLEX_B200_OPTIMIZATION.md). Production now
requests the streamed-root Triton path; measure on the actual B200 before
attributing a throughput change to it.
