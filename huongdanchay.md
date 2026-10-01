# Hướng dẫn chạy nhanh trên B200

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
export PYTHON_BIN="$(command -v python)"
```

## 1. Pretrain draft Qwen2.5-3B trong 1 epoch

```bash
MODEL=/workspace/storage-shared/models/Qwen2.5-3B-Instruct \
PRETRAIN_DATASET=sharegpt \
PRETRAIN_DATASET_PATH=/workspace/storage-shared/nlp/minhpn19/data/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json \
PRETRAIN_EPOCHS=1 \
PRETRAIN_LR=5e-5 \
PRETRAIN_BATCH_SIZE=8 \
PRETRAIN_MAX_LENGTH=2048 \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
bash pretrain_qwen25_3b.sh
```

Draft mới nhất được publish tự động tại:

```text
outputs/pretrain/qwen25_3b/latest_checkpoint
outputs/pretrain/qwen25_3b/latest_draft_config.json
outputs/pretrain/qwen25_3b/latest_vocab_mapping.pt
```

Run đầy đủ nằm trong
`outputs/pretrain/qwen25_3b/<run_name>/`; terminal cũng in chính xác `Run dir`.

## 2. Train FastGRPO + Reflex active bằng DAPO

Launcher tự lấy draft mới nhất ở bước 1:

```bash
MODEL=/workspace/storage-shared/models/Qwen2.5-3B-Instruct \
DATASET=dapo \
DATASET_PATH=/workspace/storage-shared/nlp/minhpn19/data/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet \
REFLEX_MODE=active \
REFLEX_FEATURE_DIM=8 \
REFLEX_LR=0.05 \
REFLEX_WEIGHT_DECAY=0.0 \
TARGET_LR=1e-6 \
DRAFT_LR=1e-6 \
BATCH_SIZE=8 \
ACCUMULATION_STEPS=4 \
GEN_MAX_LENGTH=2048 \
MAX_PROMPT_LENGTH=2048 \
NUM_EPOCHS=1 \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
bash train_qwen25_3b.sh
```

Để chạy baseline công bằng, giữ nguyên mọi biến và đổi duy nhất:

```bash
REFLEX_MODE=off bash train_qwen25_3b.sh
```

Muốn chỉ rõ draft thay vì dùng link mới nhất:

```bash
DRAFT_CHECKPOINT=/absolute/pretrain/run/checkpoints/<run>-latest \
DRAFT_CONFIG=/absolute/pretrain/run/config/eagle3.json \
VOCAB_MAPPING=/absolute/pretrain/run/features/vocab_mapping/vocab_mapping.pt \
REFLEX_MODE=active bash train_qwen25_3b.sh
```

Resume run train cũ:

```bash
RUN_DIR=/absolute/path/to/the/same/run RESUME=auto bash train_qwen25_3b.sh
```

## 3. Đổi dataset hoặc model

Các dataset train hợp lệ:

```bash
DATASET=gsm8k bash train_qwen25_3b.sh
DATASET=simplelr bash train_qwen25_3b.sh
DATASET=dapo bash train_qwen25_3b.sh
```

Đổi model chỉ cần dùng wrapper tương ứng: `qwen25_1p5b`, `qwen25_3b`,
`qwen25_7b`, `qwen25_14b`, `qwen3_1p7b`, `qwen3_4b`, hoặc `llama31_8b`.
Nếu đường dẫn model trên máy khác tên mặc định, truyền `MODEL=/đường/dẫn/thật`.

## 4. Output và plot

Train output nằm tại:

```text
outputs/train/qwen25_3b/<run_name>/
  checkpoints/
  logs/metrics.jsonl
  logs/timing.csv
  config_resolved.yaml
  summary.json
  summary.txt
```

Vẽ nhiều run:

```bash
bash scripts/plot_training_time.sh \
  /workspace/storage-shared/nlp/minhpn19/SpecNaacl/outputs/train/qwen25_3b/<baseline_run> \
  /workspace/storage-shared/nlp/minhpn19/SpecNaacl/outputs/train/qwen25_3b/<reflex_run>
```
