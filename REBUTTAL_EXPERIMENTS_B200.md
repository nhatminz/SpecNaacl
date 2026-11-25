# NAACL rebuttal experiments on one B200

Run each command in its own shell from the transferred checkout. Use the same B200 software environment, model files, data files and pretrained checkpoint for both methods in each pair. Set `PYTHON_BIN` to the Python in that environment if it differs from `python3`.

```bash
cd /workspace/storage-shared/nlp/users/minhpn19/SpecNaacl
export PROJECT_ROOT="$PWD"
export DATA_ROOT=/workspace/storage-shared/nlp/users/minhpn19/data
export MODEL_ROOT=/workspace/storage-shared/models
export CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1
export GSM8K_TRAIN="$DATA_ROOT/gsm8k/main/train-00000-of-00001.parquet"
export GSM8K_TEST="$DATA_ROOT/gsm8k/main/test-00000-of-00001.parquet"
```

The launchers resolve the project root from their location. Model defaults are `$MODEL_ROOT/Qwen2.5-3B-Instruct` and `$MODEL_ROOT/Qwen2.5-7B-Instruct`. Set `MODEL` or `DATASET_PATH` to use an alternate local input. `DATASET=simplelr` resolves to `$DATA_ROOT/simplelr_abel_level3to5/train.parquet`. Training defaults to the original pretraining layout, `$PROJECT_ROOT/outputs/pretrain/<model_key>/latest_checkpoint/draft.pth`. The pretraining script writes the `latest_checkpoint` link. Set `DRAFT_CHECKPOINT` to a specific immutable `draft.pth` for paired runs if the link might change. `PRETRAIN_OUTPUT_ROOT` can point to another original pretraining output tree. `REBUTTAL_OUTPUT_ROOT` only changes the destination for experiment results. Set `TARGET_ADAPTER` when the historical runs started from a saved target adapter. Otherwise both methods start from the same base model and initialize fresh target LoRA with the same training seed within a pair.

## Eight training launchers

The following are independent commands. Launch only the selected run in a given shell.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal/train_qwen25_7b_gsm8k_fastgrpo_seed123.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal/train_qwen25_7b_gsm8k_spark_seed123.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal/train_qwen25_7b_gsm8k_fastgrpo_seed456.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal/train_qwen25_7b_gsm8k_spark_seed456.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal/train_qwen25_3b_gsm8k_fastgrpo_g16.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal/train_qwen25_3b_gsm8k_spark_g16.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal/train_qwen25_3b_gsm8k_fastgrpo_response4096.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal/train_qwen25_3b_gsm8k_spark_response4096.sh
```

| Experiment | Batch | Responses per prompt | Length setting | SPARK |
|---|---:|---:|---|---|
| 7B multi-seed, seeds 123 and 456 | 8 | 8 | Legacy total-context ceiling 2048 | Set to verified seed-42 configuration when available |
| 3B concurrency, seed 42 | 8 | 16 | Legacy total-context ceiling 2048 | Per-prompt-group B |
| 3B long context, seed 42 | 8 | 8 | 4096 new tokens per row | Per-prompt-group B |

Both new 3B pairs use learned A, rollout reset of B and feedback every verification round. The G16 pair retains verification capacity 512 and expands from 64 to 128 initial responses. The existing scheduler reduces the initial verification budget from 8 to 4 nodes per response, including the root. It adapts as responses finish. Flags remain at the existing launcher settings for LR, proposal budgets, data duration and GRPO settings unless explicitly overridden. Multi-seed defaults are one complete epoch and no target-update or prompt limit, as in the checked-in launcher. Training seeds control Python, NumPy and CPU/CUDA PyTorch. The subset seed is 42 and the launcher fixes the default sampler/DataLoader seed at 42 in each pair. `OPD_PROJECTOR_SEED` and `EVALUATION_SEED` are separate.

### Historical settings and paper discrepancy

The checked-in SPARK launcher defaults to `rollout_batch`. The submitted paper describes `per_prompt_group`. The available ablation documentation describes shared B for the 3B/SimpleLR reference. The actual seed-42 7B/GSM8K run manifests and the B200 mounts are unavailable in this checkout. Therefore the checked-in multi-seed fallback is explicitly logged as unverified. Before using the new runs as repeats of the paper result, compare the historical run's config, summary and console, including epoch/update limits and the effective proposal profile on the B200. The 3B/SimpleLR ablation budget of 1000 updates is from a different experiment.

Set `HISTORICAL_CONFIG` to a trusted shell file containing the verified settings for the 7B pair. Use the same file for all four seed launchers. Include the original `NUM_EPOCHS`, `MAX_TARGET_OPTIMIZER_STEPS`, `MAX_ROLLOUT_PROMPTS`, `DRAFT_CHECKPOINT`, `TARGET_ADAPTER`, `OPD_B_SHARING_SCOPE`, `OPD_B_RESET_MODE`, `OPD_TRAIN_PROJECTOR`, `OPD_PROJECTOR_INIT`, `OPD_PROJECTOR_SEED`, `OPD_UPDATE_INTERVAL_ROUNDS`, and any nondefault schedule or sampler settings supported by the launcher. The multi-seed launcher keeps the chosen seed, batch size, response count and the seed-42 subset. If the historical scope was per-prompt group, set `OPD_B_SHARING_SCOPE=per_prompt_group` in the verified settings file. Prior proposal profiles carry a kernel fingerprint. An explicit profile with a mismatched fingerprint fails validation. If no compatible profile is present, the existing auto selector reports its uncalibrated fallback. Record the actual profile and dispatch settings before comparing timings.

### Dry runs and small checks

```bash
DRY_RUN=true CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal/train_qwen25_7b_gsm8k_fastgrpo_seed123.sh
DRY_RUN=true bash scripts/rebuttal/benchmark_concurrency.sh
DRY_RUN=true bash scripts/rebuttal/benchmark_long_context.sh
DRY_RUN=true bash scripts/rebuttal/evaluate_gsm8k.sh
DRY_RUN=true bash scripts/rebuttal/summarize_results.sh

python -m pytest -q tests/test_rebuttal.py
```

The focused tests include a synthetic CUDA training job and both real speculative entrypoints with a tiny model. For a distinct B200 integration check using the local production assets, keep smoke results under a separate destination:

```bash
REBUTTAL_OUTPUT_ROOT="$PROJECT_ROOT/outputs/rebuttal_smoke" \
MAX_TRAIN_SAMPLES=16 MAX_TARGET_OPTIMIZER_STEPS=1 EVAL_INTERVAL_UPDATES=1 \
GEN_MAX_LENGTH=128 MAX_PROMPT_LENGTH=96 CUDA_VISIBLE_DEVICES=0 \
bash scripts/rebuttal/train_qwen25_3b_gsm8k_spark_g16.sh
```

This is a small integration check. A smoke batch can have all zero-variance groups and finish with zero policy updates. Full 4096-token generation, G16 peak memory, B200-specific Triton execution/proposal calibration and steady-state comparisons require B200 validation. OOMs are surfaced at the requested configuration.

## Held-out evaluation

Run this after training on each selected run directory containing `config_resolved.yaml`. Evaluation uses the target alone with the saved LoRA, the training chat template, ordered GSM8K test examples, the same boxed-answer parser and correctness verifier as training. The defaults are greedy generation, 2048 new tokens, batch 8, all test examples and the training run's recorded evaluation seed.

```bash
export RUN_DIR=/absolute/path/to/the/training-run
CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal/evaluate_gsm8k.sh "$RUN_DIR"
```

Use `EVAL_SAMPLE=true` to enable sampling and keep temperature, top-p, seed, batch size and response cap common across comparisons. Use `EVALUATION_OUTPUT_DIR` for a fresh alternate evaluation. The summary script reads the canonical `logs/` directory by default. For a historical run with target adapters but no evaluation snapshots, provide `TARGET_MODEL` and `EVAL_CHECKPOINT_MANIFEST`. That manifest is a JSON array. Each entry has `path` relative to the manifest or absolute, `kind`, `target_optimizer_steps`, and `cumulative_training_wall_time_s` from the original telemetry. Initial snapshots use update 0 and time 0. Use real optimizer-call indices, including for intermediate and final snapshots.

## Frozen checkpoint benchmarks

Choose a frozen target, draft backbone and trained projector for the entire sweep. Both engines share the target and draft weights, prompts, sampling settings, hardware and repetition seeds. `DRAFT_CHECKPOINT` may itself contain the learned `opd_projector`. Otherwise set `PROJECTOR_CHECKPOINT` to the trained SPARK `draft.pth` containing A, or a serialized FP32 A tensor. The benchmark fails if learned A is missing. SPARK updates B on each verification round and resets B per rollout.

```bash
export DRAFT_CHECKPOINT=/absolute/path/to/frozen/draft.pth
export PROJECTOR_CHECKPOINT=/absolute/path/to/trained/spark/draft.pth
# Set TARGET_ADAPTER to the chosen frozen target LoRA when using one.
CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal/benchmark_concurrency.sh
CUDA_VISIBLE_DEVICES=0 PROMPT_GROUP_SWEEP=true bash scripts/rebuttal/benchmark_concurrency.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal/benchmark_long_context.sh
CUDA_VISIBLE_DEVICES=0 MODEL_KEY=qwen25_7b DATASET=simplelr \
bash scripts/rebuttal/benchmark_concurrency.sh
```

Use checkpoints for the selected model, including in the 7B command. Concurrency uses B8 × G4/G8/G16. The alternate prompt-group sweep uses B1/B2/B4/B8/B16 × G8. The long-context sweep compares 2048 and 4096 new tokens. Each configuration warms the exact measured prompt/seed schedule, then performs five timed repetitions by default. Change `BENCHMARK_WARMUP` or `BENCHMARK_REPETITIONS` to adjust this. A separate diagnostic replay collects event and memory measurements and verifies generated-token equality against the clean run. `GENERATION_PROFILE=clean` runs the timed repetitions alone. Benchmark summaries include sample standard deviations across repetitions and paired ratios. These repetitions measure a frozen system and are distinct from training seed uncertainty.

## Files and metric definitions

Runs live in `outputs/rebuttal/{multiseed,concurrency,long_context,frozen_benchmarks,summaries}`. Training run names encode the experiment, model, dataset, method, seed, batch, response count and response cap. Each training launch uses a fresh directory. Optional rolling resume state under `checkpoints/resume/` preserves the existing checkpoint format. Rebuttal launchers start fresh and reject resume arguments. Historical launchers still support their existing resume behavior.

- `config_resolved.yaml` contains a YAML-compatible JSON object with resolved flags, seeds, input SHA-256 identifiers and subset/prompt-order provenance. `environment.json` records software versions actually loaded, Git state, driver, visible devices, world size and GPU memory. The hashes are collected outside the training clock. `summary.json` records completion, actual counts, checkpoint I/O and elapsed times.
- `logs/rollout_metrics.csv` contains every generated rollout. Reward is correctness plus 0.2 times format reward, measured across all sampled responses. It also reports eligible-only reward and group counts. `response_metrics.jsonl` retains per-response lengths, rewards, eligibility and EOS. Lengths exclude prompts and padding and include an emitted EOS. Historical total-context termination uses the original batch-wide stop, so some rows can end early without hitting the local limit. The new per-row cap clips verified paths before committing them and retires each row independently. Length-cap termination and EOS are mutually exclusive in the rate statistic. The fractions over 2048 and reaching 4096 use actual lengths.
- AAL is committed tokens from verification, including the bonus/target token for each active response-round, divided by active response verification rounds. Prefill's first token is excluded from AAL and included in generation throughput. Draft acceptance rate divides accepted draft tokens by actual proposed draft nodes. Pass@1 is the fraction of held-out responses that pass the same training correctness verifier.
- `optimizer_updates.csv` is indexed by actual target optimizer calls. It records reward for intervening rollouts, cumulative reward, losses, LRs, tokens and elapsed times. Evaluation snapshots under `checkpoints/evaluation/` are captured initially, at `EVAL_INTERVAL_UPDATES` actual updates (50 by default, 0 for initial/final only), and at completion. These target adapters are independent of rolling resume retention. `checkpoints/target/` and `checkpoints/draft/` retain the existing trained model outputs.
- `generation_profile.csv` includes absolute generation wall time. Detailed mode also records CUDA events on the stream that executed each operation. Event intervals can overlap across streams and nested scopes. SPARK's corrected proposal is inclusive. Its exclusive correction application is blank for the fused path. `other_generation_overhead_time_ms` measures postprocessing. Clean mode leaves unmeasured phases blank. FastGRPO SPARK fields are zero. The `verification_rounds.csv` detail uses `ROUND_LOG_FREQUENCY` (default 20, 0 disables rows). Rollout aggregate counters always include every verification round.
- `memory_profile.csv` reports allocated correction storage, persistent correction workspace, A bytes, updates and allocator peaks. KV storage peaks at instrumented cache boundaries are available in detailed mode. Storage accounting deduplicates aliases. The dense FP32 B comparison is allocated matrices × vocabulary × rank × 4 bytes. G8 and G16 have the same 8 matrices for grouped B. Flash/SDPA scratch, allocator fragmentation, transient temporaries and retained caches can affect allocator peaks independently of B.
- The training wall clock begins after model/data setup and excludes measured checkpoint writes. Draft and target timings use CUDA events. Evaluation runs in a separate process. `evaluation_metrics.csv`, `evaluation_predictions.jsonl` and `evaluation_summary.json` record evaluation latency, adapter loading overhead, sample counts and accuracy indexed by training time and update. The training summary leaves evaluation overhead empty until measured separately. Older console/step/rollout logs are retained under `logs/` and may use a different clock or index.

## Paired aggregation

```bash
bash scripts/rebuttal/summarize_results.sh
HISTORICAL_MANIFEST=/absolute/path/to/history.json bash scripts/rebuttal/summarize_results.sh
# Explicit run selection if a seed/method was launched more than once.
bash scripts/rebuttal/summarize_results.sh --run-dir /absolute/path/to/fastgrpo-run \
  --run-dir /absolute/path/to/spark-run
```

The summarizer checks model, adapter, draft and dataset content IDs, the subset/order protocol, resolved training and decoding settings, profiling mode, GPU/runtime and SPARK settings before pooling. It produces per-seed results, means and sample standard deviations across independent seeds, paired SPARK speedups at the largest common observed actual policy update, and PDF curves. Reward and evaluation time curves use the last available observation within a common measured GPU-hour range. Curves report their available seed counts. The equivalent policy-update accuracy table uses a shared available checkpoint. A single optional fixed AR run can be imported with method `ar` when shared settings match. AR-relative ratios are labeled conditional on that fixed run.

For legacy inputs, `history.json` is a JSON array of entries. Each entry contains `run_dir`, `source` describing the evidence, and `config` in the resolved-config schema. Use the historical config, environment, input IDs and original run settings for that object, including `identifiers`, `args`, `inputs`, `dataset`, `environment`, `initialization`, `proposal_env` and `metric_protocol`. Confirm equivalent definitions before setting `metric_protocol` to `rebuttal_v1`. The entry may set `summary` (default `summary.json`) and `evaluation_dir` (default `logs`). Paths can be absolute or relative to the run. Supply either `optimizer_updates` with real update-indexed telemetry in the new CSV schema or `legacy_rollout_timing` pointing to the original `logs/rollout_timing.csv`. The legacy importer joins that CSV to the original summary's `optimizer_step_cadence`. It recovers token and speculative work counters. Unconditional reward is unavailable from those logs. Set `comparable_generation_time: true` only if the legacy generation timing and synchronization protocol is equivalent. The opt-in rebuttal profiling paths remove FastGRPO's inherited per-expansion device synchronization and use an explicit generation stream boundary, so old absolute timings may belong in a separate comparison. Use `comparable_wall_clock_csv` with actual update and `cumulative_training_wall_time_s` only if the original data permits an equivalent training clock. Keep incompatible or unmeasured historical metrics empty. A raw legacy `total_wall_time_s` containing startup or checkpoint I/O has a different definition. The summarizer records excluded runs and reasons in `summary.json`.

No seed-42 7B/GSM8K or fixed AR artifacts were available for the implementation check. Measurements in the generated summaries come from supplied runs.

## Implementation and development validation

- Training integration and exact response budgeting are in `grpo_speculative.py`, `helper/{fastgrpo_generate,opd_generate,fastgrpo_training,fastgrpo_model}.py`, `helper/{tree_kernels,opd_scheduling,opd_reflex_kernels}.py`, and `helper/{generation_telemetry,rebuttal_metrics,train_prompts}.py`.
- Entry points are in `scripts/rebuttal/`, with shared training in `scripts/launch/train_model.sh`, evaluation and aggregation in `scripts/{evaluate_rebuttal,summarize_rebuttal}.py`, and the frozen sweep in `scripts/{benchmark_opd_reflex,benchmark_rebuttal}.py`.
- The B200 path audit updated `configs/_shared/b200_common.env`, the per-model B200 profiles, the root model training and pretraining wrappers, the two policy-lag launchers, `helper/get_QAs.py`, `scripts/plot_training_time.sh`, and `SPARK_ABLATIONS.md`.
- Development checks on RTX 3090 passed shell syntax for 61 first-party shell/config files, dry runs of 12 launchers, the Python compile check, source integrity and the 488-test suite. The final focused rebuttal and legacy launcher set passed 34 tests. The synthetic integration exercised both speculative engines in training, checkpoint evaluation, paired aggregation, and the frozen benchmark code. No actual GSM8K training or B200 benchmark measurements were collected.
