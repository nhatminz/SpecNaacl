# Chạy FastGRPO và OPD Reflex trên B200

Dùng venv SpecNaacl đã được kiểm tra; KHÔNG dùng venv TLT (khác Torch/Transformers).
Paths model/data/pretrained draft giữ nguyên. Không cần pretrain lại. Pipeline
không tải model/dataset từ Internet; giữ nguyên dependencies trong ENVIRONMENT.md.

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0
export DATASET=simplelr
export TARGET_LR=1e-5
export DRAFT_LR=1e-5
export OPD_FAST_LR=0.01   # cũng hỗ trợ FAST_LR nếu OPD_FAST_LR chưa được set
export BATCH_SIZE=8
export ACCUMULATION_STEPS=4
export RESPONSES_PER_PROMPT=8
export OPD_RANK=8
export OPD_TOPK=16
export OPD_VISITED_WEIGHT=1.0
export OPD_FRONTIER_WEIGHT=1.0
export OPD_UPDATE_STREAM=0   # UNTUNED: đo cả0/1, không giả định async nhanh hơn
export OPD_PROFILE=0
export OPD_DIAGNOSTICS=0
```

Dataset default:
`/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/train.parquet`.
Đổi bằng `DATASET=dapo`/`gsm8k`, hoặc export `DATASET_PATH` tới file thật.
Draft/config/mapping default: `outputs/pretrain/<model_key>/latest_*`.
Nếu khác, export DRAFT_CHECKPOINT, DRAFT_CONFIG, VOCAB_MAPPING của cùng pretrained run.

## Chạy từng cặp model

```bash
# OPD Reflex                       # FastGRPO/OFF
bash train_qwen25_3b.sh             # bash train_qwen25_3b_fastgrpo.sh
bash train_qwen3_1p7b.sh             # bash train_qwen3_1p7b_fastgrpo.sh
bash train_qwen3_4b.sh               # bash train_qwen3_4b_fastgrpo.sh
bash train_qwen25_1p5b.sh            # bash train_qwen25_1p5b_fastgrpo.sh
bash train_qwen25_7b.sh              # bash train_qwen25_7b_fastgrpo.sh
bash train_qwen25_14b.sh             # bash train_qwen25_14b_fastgrpo.sh
bash train_llama31_8b.sh             # bash train_llama31_8b_fastgrpo.sh
```

Mỗi lệnh là một run riêng. Run name tự có method/seed/timestamp/UUID.
Không gọi tất cả model nếu chỉ cần một model.

Dry-run và smoke thực với weights thật:

```bash
DRY_RUN=true bash train_qwen25_3b.sh
MAX_TRAIN_SAMPLES=128 bash train_qwen25_3b.sh --max_grpo_steps 2
MAX_TRAIN_SAMPLES=128 bash train_qwen25_3b_fastgrpo.sh --max_grpo_steps 2
```

## Sweep nhanh LR/stream: FROZEN model, không train/checkpoint

```bash
MODEL_KEY=qwen25_3b OPD_FAST_LRS=0.001,0.01,0.05,0.1 OPD_STREAMS=0,1 \
BENCH_SEEDS=42,43 BENCH_ITERATIONS=2 BENCH_WARMUP=1 \
bash scripts/sweep_opd_reflex.sh
```

Quick defaults: prompts8, responses8, total max_length512 (bao gồm prompt),
max_prompt_length256, verification_capacity512, K8, depth5. Warmup chạy đúng
seed/prompt schedule để tránh đổ JIT vào timing. Muốn workload train:

```bash
MODEL_KEY=qwen25_3b BENCH_MAX_LENGTH=2048 BENCH_MAX_PROMPT_LENGTH=2048 \
BENCH_ITERATIONS=3 BENCH_SEEDS=11,29,47 bash scripts/sweep_opd_reflex.sh
```

Prompt đã dài bằng max_length bị reject; tăng max_length hoặc giảm max_prompt_length.
Dùng BENCH_OUTPUT mới nếu muốn chỉ định output. Script xuất report.json,
summary.csv, responses.jsonl và fastest_observed.env. recommendation=null nếu
không config nào đồng thời AAL tăng và throughput>=baseline. fastest_observed.env
KHÔNG được coi là recommendation mặc định; kiểm tra delta/overhead, sau đó validate
trên held-out prompts trước khi tự source nó.

OPD_PROFILE=1 chạy diagnostic replay TÁCH RIÊNG khỏi wall/throughput. Các profile
timers không phải authoritative speedup. OPD_DIAGNOSTICS=1 thêm B norm/max ở cuối
rollout, không thêm target/draft transformer forward.

## Outputs và plot AAL đúng từng step

Train outputs: `outputs/train/<model_key>/<run_name>/`.
Mỗi completed inherited GRPO step có một row trong logs/timing.csv và một
target_train row trong logs/metrics.jsonl (các phase khác có row riêng).
Checkpoint target/draft/resume nằm trong checkpoints/ của chính run.
Benchmark outputs: `outputs/benchmarks/opd_<model_key>_<timestamp>/`.

Sau khi có hai run, truyền đường dẫn logs thật của chúng:

```bash
python scripts/plot_opd_aal.py --fastgrpo "$FASTGRPO_TIMING_CSV" \
  --opd-reflex "$OPD_TIMING_CSV" --output "$NEW_AAL_PLOT_PATH"
```

Các biến trên phải trỏ tới CSV/output bạn thực sự chọn. Plot dùng step counters,
không dùng cumulative AAL/moving average. Các pretrain launchers vẫn giữ nguyên.
Không resume optimizer trajectory của method cũ vào OPD; giữ cùng method khi
resume. Có thể dùng lại pretrained draft/target adapter như initialization.
