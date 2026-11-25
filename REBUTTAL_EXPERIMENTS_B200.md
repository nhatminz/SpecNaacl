# B200 rebuttal runs from d431cf2

This branch starts from `d431cf247ebce2abbbbcb7025f8a331321f324e2`. The four model launchers below are the ones from that revision. The four extra 7B scripts set a seed, select GSM8K by default, and call those launchers. `opd_reflex` is the SPARK method.

## Each workload

```bash
cd /workspace/storage-shared/nlp/users/minhpn19/SpecNaacl
unset RESUME RUN_DIR RUN_NAME MAX_RESPONSE_TOKENS
export NPROC_PER_NODE=1 DATASET=gsm8k TRAIN_SUBSET_SEED=42
```

The common data default is `/workspace/storage-shared/nlp/users/minhpn19/data`. The launcher uses pretraining weights from `outputs/pretrain/<model>/latest_checkpoint/draft.pth`. Set `DRAFT_CHECKPOINT` if the original weights are elsewhere. The run starts with fresh optimizer state. For a separate random-init experiment set `DRAFT_INITIALIZATION_MODE=random` for both methods of a pair.

## Workload 1 · 7B seeds 123 and 456

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal/train_qwen25_7b_gsm8k_fastgrpo_seed123.sh &
CUDA_VISIBLE_DEVICES=1 bash scripts/rebuttal/train_qwen25_7b_gsm8k_opd_reflex_seed123.sh &
CUDA_VISIBLE_DEVICES=2 bash scripts/rebuttal/train_qwen25_7b_gsm8k_fastgrpo_seed456.sh &
CUDA_VISIBLE_DEVICES=3 bash scripts/rebuttal/train_qwen25_7b_gsm8k_opd_reflex_seed456.sh &
```

`SEED` selects training randomness. `TRAIN_SUBSET_SEED=42` still selects the data subset, and the rebuttal launcher uses it as the default prompt-order seed. The base SPARK sharing default for these commands is `rollout_batch`. Compare the archived seed-42 settings before pooling, including the B sharing scope and training budget.

## Workload 2 · 3B G16 and 4096 new tokens

```bash
export SEED=42 OPD_B_SHARING_SCOPE=per_prompt_group

CUDA_VISIBLE_DEVICES=0 REBUTTAL_EXPERIMENT=concurrency GENERATION_PROFILE=detailed RESPONSES_PER_PROMPT=16 bash train_qwen25_3b_fastgrpo.sh &
CUDA_VISIBLE_DEVICES=1 REBUTTAL_EXPERIMENT=concurrency GENERATION_PROFILE=detailed RESPONSES_PER_PROMPT=16 bash train_qwen25_3b.sh &
CUDA_VISIBLE_DEVICES=2 REBUTTAL_EXPERIMENT=long_context MAX_RESPONSE_TOKENS=4096 bash train_qwen25_3b_fastgrpo.sh &
CUDA_VISIBLE_DEVICES=3 REBUTTAL_EXPERIMENT=long_context MAX_RESPONSE_TOKENS=4096 bash train_qwen25_3b.sh &
```

The original batch size remains 8. Set `GENERATION_PROFILE=clean` for G16 when collecting throughput without detailed events. `MAX_RESPONSE_TOKENS` counts new tokens for each response. `GEN_MAX_LENGTH` retains its original total-context semantics for runs using the legacy limit. Long runs continue using the original prompt limit and optimize the complete committed response with chunked vocabulary projections.

Add `DRY_RUN=true` before any command to inspect the resolved arguments. The commands above run in an interactive Bash session. A workload entrypoint script should remain alive through its child runs.

## Metrics and optional offline analysis

`REBUTTAL_EXPERIMENT` enables the additional logs under `outputs/rebuttal/<experiment>/<run>/`. The original logs and checkpoint format are still written. The additional files include:

- `rollout_metrics.csv` and `response_metrics.jsonl` for reward before group filtering, eligibility, measured token lengths, EOS/cap termination, speculative counters, and wall/GPU times.
- `optimizer_updates.csv` indexed by actual target optimizer updates, with draft updates and cumulative counters. Lightweight target evaluation snapshots are saved initially, at `EVAL_INTERVAL_UPDATES` (default 50), and at completion.
- `generation_profile.csv`, `memory_profile.csv` and, in detailed mode, sampled `verification_rounds.csv`. CUDA event intervals and generation wall time have different meanings and can overlap. Fused correction execution has inclusive proposal timing. The dense FP32 B estimate uses matrix count × vocabulary size × rank × 4 bytes.
- `config_resolved.yaml`, `environment.json` and `summary.json` for reproducibility and final totals.

The offline commands use the existing checkout and the artifacts from saved runs. The original benchmark and model launchers retain their own entrypoints.

```bash
data_root="${DATA_ROOT:-/workspace/storage-shared/nlp/users/minhpn19/data}"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_rebuttal.py \
  --run-dir /path/to/run \
  --dataset-path "$data_root/gsm8k/main/test-00000-of-00001.parquet"

CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_rebuttal.py \
  --sweep concurrency --model-key qwen25_3b --dataset gsm8k \
  --target-model /workspace/storage-shared/models/Qwen2.5-3B-Instruct \
  --draft-checkpoint /path/to/frozen/draft.pth \
  --projector-checkpoint /path/to/learned/draft.pth \
  --dataset-path "$data_root/gsm8k/main/train-00000-of-00001.parquet" \
  --output outputs/rebuttal/frozen_benchmarks/concurrency-example

python scripts/summarize_rebuttal.py \
  --runs-root outputs/rebuttal/multiseed --output outputs/rebuttal/summaries/multiseed
```

For frozen runs, `--sweep long_context` compares caps of 2048 and 4096. `--sweep prompt_groups` changes the prompt-group count. Supply a trained projector with each chosen frozen draft. To combine a compatible historical seed, the summarizer accepts `--historical-manifest` with the original configuration and update-indexed metrics or an original rollout CSV paired with its optimizer-call cadence.

Development validation covered all eight dry runs and shell syntax for 50 first-party scripts/configs. An RTX 3090 synthetic pair covered the per-response cap, one actual optimizer update, held-out snapshots and paired aggregation. Production throughput and memory at 4096 new tokens require B200 measurements.
