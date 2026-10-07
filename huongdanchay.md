# Chạy FastGRPO / ReflexOPD

Chạy các lệnh dưới đây từ `SpecNaacl`, sau khi chuẩn bị
[môi trường](ENVIRONMENT.md). Model/data mặc định vẫn lấy từ
`configs/<model_key>/b200.env` và `configs/_shared/b200_common.env`.

## Pretrain

```bash
CUDA_VISIBLE_DEVICES=0 bash pretrain_qwen25_3b.sh
```

Sáu model key có đầy đủ launcher `pretrain_<model_key>.sh`:
`qwen25_1p5b`, `qwen25_3b`, `qwen25_7b`, `qwen25_14b`,
`qwen3_1p7b`, `qwen3_4b`.
Mặc định ShareGPT, 5 epochs. Ví dụ override/smoke:

```bash
PRETRAIN_MAX_SAMPLES=16 PRETRAIN_BATCH_SIZE=2 PRETRAIN_DATALOADER_WORKERS=0 \
  bash pretrain_qwen25_3b.sh --max_steps 2
RESUME=auto bash pretrain_qwen25_3b.sh
```

Khi resume phải giữ cùng model/data, số epoch, batch size, accumulation và seed
của run ban đầu. Các override smoke trên cần được giữ nếu resume chính run đó.
`RESUME=auto` chọn active run chưa hoàn tất; cũng có thể đặt đường dẫn run,
checkpoint directory hoặc `training_state.pt` cụ thể.

Output vẫn là `outputs/pretrain/<model_key>/<run_name>/`, với `logs/`,
`checkpoints/stepN.pth` và `checkpoints/<run_name>-latest/`.
`outputs/pretrain/<model_key>/latest_checkpoint` trỏ tới checkpoint mới nhất;
`latest_target_config.json` dùng cho autotuner. Chuỗi `eagle3-pretrain` trong
run name cũ được giữ để không đổi convention tên, không biểu thị architecture.
Checkpoint cũ từ SpecForge không tương thích; chạy pretrain mới trước GRPO.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_3b_fastgrpo.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_3b.sh
```

Thay model key bằng một trong sáu key phía trên. Bản không có hậu tố
`_fastgrpo` chọn `opd_reflex`; mỗi cặp có cùng cấu hình ngoài method/OPD flags.
Cả hai mặc định đọc `outputs/pretrain/<model_key>/latest_checkpoint`.
Có thể đặt `DRAFT_CHECKPOINT` về cùng một checkpoint cố định cho hai run.

Data mặc định: `$DATA_ROOT/simplelr_abel_level3to5/train.parquet`.
Target LR `1e-6`, draft LR `1e-4`, draft accumulation `1`.
ReflexOPD mặc định rank `8`, TopK `16`, fast LR `0.01`, update stream `1`.

```bash
MAX_TRAIN_SAMPLES=128 bash train_qwen25_3b_fastgrpo.sh --max_grpo_steps 2
MAX_TRAIN_SAMPLES=128 bash train_qwen25_3b.sh --max_grpo_steps 2
RESUME=auto bash train_qwen25_3b.sh
DRY_RUN=true bash train_qwen3_4b.sh
```

Output giữ `outputs/train/<model_key>/<run_name>/`:
`logs/metrics.jsonl`, `logs/timing.csv`, `logs/rollout_timing.csv`,
`summary.json`, `checkpoints/` và `statistics/`.
`timing.csv` có một row mỗi GRPO step, `rollout_timing.csv` một row mỗi
DataLoader iteration. Resume dùng đúng method và cấu hình của run ban đầu.

`MAX_VERIFICATION_NUM` mặc định 160, như FastGRPO gốc: giá trị cũ 512 vượt
số node tối đa khi batch co về 1 với cây K=8/depth=5 (264 node).
`VERIFICATION_CAPACITY=512` vẫn giữ. Nếu override K/depth/capacity cần chọn
số node hợp lệ; baseline không thay pruning upstream để sửa cấu hình sai.

## Tune và đo thực tế

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/tune_opd_proposals.sh
MODEL_KEY=qwen25_3b OPD_FAST_LRS=0.01 OPD_STREAMS=0,1 \
  bash scripts/sweep_opd_reflex.sh
```

Tuner chỉ dùng config target và checkpoint FastGRPO, không dùng vocab mapping.
Profile phụ thuộc GPU, full V, rank, dtype, TopK và kernel/compiler fingerprint.
Không dùng profile RTX 3090 cho B200. Chỉ nhánh OPD lookup/tune proposal.
`OPD_AUTO_TUNE_IF_MISSING=1` cho phép tune lúc startup; mặc định không bật.

Sweep lưu report/summary/responses trong `outputs/benchmarks/`, so baseline
nguyên bản với OPD cùng model/sampling. Đây là frozen-rollout benchmark,
không train A/target/draft. Muốn đánh giá A đã học, đặt `DRAFT_CHECKPOINT`
về draft checkpoint của run OPD. `OPD_PROFILE=1` tạo replay chẩn đoán riêng,
không cộng replay vào generation wall time.
Chỉ kết luận về AAL/throughput sau khi đo trên GPU, weights và workload thật.
