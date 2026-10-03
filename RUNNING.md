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

## Train FastGRPO

The matching `train_<model>.sh` wrappers expose `MODEL`, `DATASET`,
`DRAFT_CHECKPOINT`, `TARGET_LR`, `DRAFT_LR`, `BATCH_SIZE`,
`ACCUMULATION_STEPS`, `GEN_MAX_LENGTH`, `MAX_PROMPT_LENGTH`, `NUM_EPOCHS`,
`NPROC_PER_NODE`, `METHOD`, `REFLEX_FEATURE_DIM`, `REFLEX_LR`, and
`REFLEX_WEIGHT_DECAY`, and `REFLEX_BACKEND=auto|torch|triton`. `METHOD=fastgrpo` disables Reflex and
`METHOD=specnaacl` enables it; `DATASET` accepts `gsm8k`, `simplelr`, or `dapo`.

`REFLEX_FEEDBACK_SCOPE=root|visited_path` defaults to root. Example:
`METHOD=specnaacl REFLEX_BACKEND=torch REFLEX_FEEDBACK_SCOPE=visited_path bash train_qwen25_3b.sh`.
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
disabled by default. `LOG_INTERVAL=100` controls lightweight metric writes.

Plot selected runs by editing `RUN_DIRS` in `scripts/plot_training_time.sh` or:

```bash
bash scripts/plot_training_time.sh /absolute/run1 /absolute/run2
```

For the opt-in Reflex parallel kernels, GPU path bookkeeping, side-stream
overlap, and B200 benchmark/selection commands, see
[REFLEX_B200_OPTIMIZATION.md](REFLEX_B200_OPTIMIZATION.md). Production defaults
keep unmeasured kernel/stream candidates disabled.
