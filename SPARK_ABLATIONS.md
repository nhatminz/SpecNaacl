# SPARK ablations: Qwen2.5-3B-Instruct / SimpleLR

Các cấu hình dùng `grpo_speculative.py` và launcher train hiện tại. Không có
pipeline training riêng. Defaults của training chính vẫn là head-aligned A,
train A và interval 1. FastGRPO, GRPO objective, sampler, verifier và Triton
kernels không đổi.

| Configuration | A init | Train A | Update interval |
|---|---|---:|---:|
| learned_random | random_orthogonal | 1 | 1 |
| frozen_random | random_orthogonal | 0 | 1 |
| interval5 | random_orthogonal | 1 | 5 |
| interval10 | random_orthogonal | 1 | 10 |
| interval15 | random_orthogonal | 1 | 15 |
| interval0 | random_orthogonal | 1 (inactive) | 0 |

`learned_random` là reference chung của cả hai studies. A có rank 8, QR trên
FP32 Gaussian với CPU generator cục bộ, seed 42. Learned/frozen có A ban đầu
bitwise identical và không tiêu thụ global CPU/CUDA sampling RNG. Frozen A
không có analytical gradient hay optimizer membership/weight decay.

Verification round ở đây là **một batch verification / một lần target verify**,
vì B được dùng chung cho các responses. Không dùng response-weighted round
counter hay GRPO step để điều khiển lịch. Round 1 reset tại mỗi rollout:
interval 5 cập nhật 1,6,11,...; interval 10 cập nhật 1,11,21,...; interval 15
cập nhật 1,16,31,... . Các round khác giữ B hiện tại, không extract OPD teacher,
không feedback/KL, không tích lũy gradient A và không catch up. Interval 0
dùng proposal engine với adapter disabled (B hiệu dụng bằng zero, không cấp
phát B/correction workspace), không feedback hay learning A; draft và target
vẫn train. A learned chỉ step ở draft optimizer boundary; B reset mỗi rollout.

Online trajectories của learned/frozen có thể phân kỳ sau optimizer updates.
Test fixed verification trace kiểm tra selection và teacher parity; không
tuyên bố verified states của các run online luôn giống nhau.

Chạy từ thư mục `SpecNaacl`, trong môi trường training đã cài:

```bash
# Tạo một experiment root mới, giữ nguyên đường dẫn này khi resume.
export TRAIN_MODEL_ROOT="$PWD/outputs/ablations/spark_qwen25_3b_simplelr_$(date -u +%Y%m%dT%H%M%S)_$RANDOM"
mkdir -p "$TRAIN_MODEL_ROOT"

# Snapshot đúng một draft cho cả sáu runs. latest_checkpoint có thể được
# pretraining cập nhật tiếp nên không để sáu processes tự đọc symlink riêng.
cp "$PWD/outputs/pretrain/qwen25_3b/latest_checkpoint/draft.pth" \
   "$TRAIN_MODEL_ROOT/initial_draft.pth"
export DRAFT_CHECKPOINT="$TRAIN_MODEL_ROOT/initial_draft.pth"
export MODEL=/workspace/storage-shared/models/Qwen2.5-3B-Instruct
export DATASET_PATH=/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/train.parquet
export TARGET_ADAPTER="" RESUME=""
export TRAIN_SUBSET_SEED=42 OPD_PROJECTOR_SEED=42
export MAX_TARGET_OPTIMIZER_STEPS=1000 NUM_EPOCHS=100
export TARGET_LR=1e-6 DRAFT_LR=1e-4 OPD_PROJECTOR_LR=1e-4
export OPD_FAST_LR=0.01 OPD_TOPK=16
export OPD_VISITED_WEIGHT=1.0 OPD_FRONTIER_WEIGHT=1.0
export OPD_UPDATE_STREAM=1

CUDA_VISIBLE_DEVICES=0 bash scripts/run_spark_ablation.sh learned_random &
CUDA_VISIBLE_DEVICES=1 bash scripts/run_spark_ablation.sh frozen_random &
CUDA_VISIBLE_DEVICES=2 bash scripts/run_spark_ablation.sh interval5 &
CUDA_VISIBLE_DEVICES=3 bash scripts/run_spark_ablation.sh interval10 &
CUDA_VISIBLE_DEVICES=4 bash scripts/run_spark_ablation.sh interval15 &
CUDA_VISIBLE_DEVICES=5 bash scripts/run_spark_ablation.sh interval0 &
wait
```

Budget 1000 là ví dụ và được override qua `MAX_TARGET_OPTIMIZER_STEPS` (đếm
actual target optimizer steps). Ablation launcher dùng tối đa 100 epochs để
không hết một epoch trước budget. Kiểm tra `summary.json.target_optimizer_steps`
để xác nhận từng run hoàn thành budget; không thay bằng inherited GRPO labels.
Batch/accumulation, draft training, model, data, seed, optimizer LRs, selection
weights và decoding settings dùng chung qua wrapper Qwen2.5-3B. Nếu muốn dùng
checkpoint pretrain cụ thể đã bất biến, export thẳng file `checkpoints/stepN.pth`
thay cho bước copy. `DRY_RUN=true` in đầy đủ command mà không chạy training.

Mỗi cấu hình có root riêng:
`$TRAIN_MODEL_ROOT/ablations/<name>__init-...__seed42__train...__interval.../`.
Mỗi run có timestamp/UUID và các con trỏ `active_run`/`latest_run` chỉ nằm trong
root cấu hình đó. Để resume, giữ common exports và snapshot draft cũ, rồi:

```bash
RESUME=auto CUDA_VISIBLE_DEVICES=2 bash scripts/run_spark_ablation.sh interval5
```

Checkpoint lưu/verify init, seed, train/frozen, interval, name, rank, topK,
LRs, visited/frontier weights và selection policy **trước khi load weights hay
optimizer state**. Lưu cả A ban đầu, A hiện tại, pending A gradients rank-local,
optimizers, iterator, RNG và telemetry state; B là rollout-local, không resume.
Checkpoint legacy không có ablation metadata chỉ được chấp nhận cho cấu hình
head-aligned learned interval 1, seed 42.

Toàn bộ `metrics.jsonl`, `timing.csv`, `rollout_timing.csv`, `summary.json`,
statistics, console.log và tqdm tiếp tục qua writers hiện tại. Cột cũ không
bị xóa; CSV append/resume giữ schema migration và iterator rewind hiện có.
Metadata init/mode/seed/interval/name và initial A SHA256 có trên step/iteration
logs. Counter feedback/skipped dùng Python integers; actual B updates dùng
counter GPU đã có trong packet cuối rollout. Actual update frequency là
`actual B updates / verification_batches`; feedback frequency đếm cả feedback
attempts không có valid selected states. Tất cả zero denominators trả zero.

Selected/visited/frontier, AAL, acceptance và generation throughput giữ định
nghĩa hiện tại. Bổ sung generation/E2E throughput theo step/iteration/cumulative.
Initial/final A SHA256 và change norm tính trên CPU ở startup/termination;
final fields ghi ở final completed-step log và summary, các row trước đó để
trống final fields. Không copy/hash A từng round hay từng optimizer step chỉ
để logging. Snapshot A ban đầu trong checkpoint giúp norm vẫn tính từ đầu
experiment khi resume, không reset ở đầu session mới.

Kiểm thử nằm ở `tests/test_spark_ablations.py`, bao gồm CPU oracle, CUDA/Triton,
async streams, fixed trace, short/EOS rollouts, interval=1 đối chiếu snapshot
engine trước sửa, optimizer exclusion, actual training CLI, logs và resume.
Tokens/RNG parity kiểm tra bitwise; gradient GPU dùng tolerance FP32 do atomic
feedback accumulation có thứ tự không xác định như implementation hiện tại.
Không thêm CUDA synchronization/events/kernels/transfers trong verification
round để logging. Chưa chạy sáu training studies đầy đủ trên B200 tại workspace
này; validation CUDA sử dụng RTX 3090.

Kết quả validation: toàn bộ active suite **446 passed, 0 failed** trên CUDA
(Torch 2.5.1/cu124, Transformers 5.12.1); focused CPU Torch 2.13: **74 passed,
27 skipped** do cần CUDA; focused Transformers 4.51.3: **20 passed**. Compileall,
37 shell syntax checks và dry-run đủ sáu launcher configurations đều pass.
Artifacts và smoke summaries: [validation/spark_ablations_20261010/results.json](validation/spark_ablations_20261010/results.json).
