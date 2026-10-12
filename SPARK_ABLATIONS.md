# SPARK: năm ablation trên Qwen2.5-3B-Instruct / SimpleLR

Cả năm dùng training pipeline `grpo_speculative.py` hiện tại. Reference là
`learned_random` của study trước: A `random_orthogonal`, local seed 42, train A,
rank 8, top-k 16, fast LR 0.01, visited/frontier weights 1/1, interval 1,
shared B toàn rollout batch và reset B mỗi rollout. Reference được mô tả để
so sánh; launcher study này chỉ nhận đúng năm tên bên dưới.

| Tên | Interval | B sharing | B lifecycle | Visited/frontier |
|---|---:|---|---|---|
| interval5 | 5 | rollout_batch | reset mỗi rollout | 1/1 |
| interval10 | 10 | rollout_batch | reset mỗi rollout | 1/1 |
| per_prompt_group | 1 | original prompt group | reset mỗi rollout | 1/1 |
| carry_b | 1 | rollout_batch | carry qua rollout/optimizer boundary | 1/1 |
| accepted_only | 1 | rollout_batch | reset mỗi rollout | 1/0 |

Mỗi nhánh chỉ thay biến trong bảng. Không đổi GRPO, teacher sampling/verifier,
backbone training, precision hoặc decoding budgets. Defaults của training chính
vẫn head-aligned A, interval 1, shared/reset B; FastGRPO không đổi. Các flags
interval/frozen cũ vẫn được hỗ trợ bởi CLI chính để resume, nhưng không tạo thêm
configuration trong launcher năm ablation này.

Interval dùng verification **batch** rounds, reset về 1 mỗi rollout: 5 cập nhật
1,6,11,...; 10 cập nhật 1,11,21,... . Round bỏ qua vẫn verify/propose và dùng B
hiện tại; không teacher extraction, feedback, A gradients hay catch-up update.
Accepted-only tái sử dụng selection hiện tại với frontier weight bằng zero:
chỉ visited states có proposal context hợp lệ được chọn.

Per-prompt B có shape `[original_prompt_count,V,rank]`; G responses dùng chung
B của prompt gốc. Owner IDs bám theo swap-remove/residual response identity,
không tính lại bằng physical row / G. Canonical teacher rows được ánh xạ về
physical cache row rồi mới tra owner. CUDA sparse/fused/gemm proposal, selected
head, B update và gradient A đều đọc đúng group. Mỗi B update chia cho tổng
valid selected-state weight **của group đó**; group không có valid feedback giữ
nguyên B. A vẫn dùng tổng gradient/weight toàn bộ selected states như trước và
chỉ optimizer-step ở draft boundary. Không thêm target forward/chạy generation
riêng từng group.

Carry-B giữ B/bitmap/active IDs/count/ever-updated/dispatch bound trên model,
riêng từng process/rank. Chỉ những buffers này đi qua rollout boundary; teacher,
hidden states và proposal contexts cũ không được đưa vào persistent state.
Runtime scratch vẫn tái sử dụng như trước và ghi lại trước khi đọc. Existing
update-done event bảo vệ reuse/checkpoint; không thêm host wait mỗi round.
Checkpoint lưu fast state trong `rank_states[rank]`, cùng A/pending A gradients,
optimizers, iterator, RNG và telemetry. Load kiểm tra configuration và world-size
trước khi load weights. Không suy diễn carry state từ checkpoint legacy thiếu
state. Fresh model/run bắt đầu B=0, kể cả dùng cùng draft initialization file.

## Chuẩn bị chung, chạy một lần

Chạy từ `SpecNaacl` trong venv training trên máy B200. Snapshot một draft cho
cả năm để tránh pretraining tiếp tục đổi `latest_checkpoint`. Giữ study root
và file `study.env` này khi resume; không tạo root/snapshot mới khi resume.

```bash
export TRAIN_MODEL_ROOT="$PWD/outputs/ablations/spark_3b_simplelr_five_$(date -u +%Y%m%dT%H%M%S)_$RANDOM"
mkdir -p "$TRAIN_MODEL_ROOT"
cp outputs/pretrain/qwen25_3b/latest_checkpoint/draft.pth "$TRAIN_MODEL_ROOT/initial_draft.pth"
export DRAFT_CHECKPOINT="$TRAIN_MODEL_ROOT/initial_draft.pth"
export MODEL=/workspace/storage-shared/models/Qwen2.5-3B-Instruct
export DATASET_PATH=/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/train.parquet
export TARGET_ADAPTER="" RESUME=""
export MAX_TARGET_OPTIMIZER_STEPS=1000 NUM_EPOCHS=100
export TARGET_LR=1e-6 DRAFT_LR=1e-4 OPD_PROJECTOR_LR=1e-4
export BATCH_SIZE=8 ACCUMULATION_STEPS=4 DRAFT_ACCUMULATION_STEPS=1
export RESPONSES_PER_PROMPT=8 GRPO_ITERATION_NUM=1 BETA=0.04 EPSILON=0.1
export TEMPERATURE=1.0 TOP_P=0.95 GEN_MAX_LENGTH=2048 MAX_PROMPT_LENGTH=2048
export VERIFICATION_CAPACITY=512 MAX_VERIFICATION_NUM=160
export MAX_DRAFT_K=8 MAX_DRAFT_TOKEN_LENGTH=5 MIN_DRAFT_TOKEN_LENGTH=3 DRAFT_TOKEN_LENGTH_C=0.75
export MODEL_DTYPE=bf16 ATTENTION_IMPLEMENTATION=sdpa IS_TRAIN_DRAFT=True
export OPD_UPDATE_STREAM=1 OPD_BACKEND=triton OPD_SAMPLER_MODE=finite
export OPD_PROPOSAL_MODE=auto OPD_DENSE_IMPLEMENTATION=auto OPD_AUTO_TUNE_IF_MISSING=0
export OPD_PROPOSAL_PROFILE=""
export OPD_PROFILE=0 OPD_DIAGNOSTICS=0 STATISTICAL_TIME=False
unset RUN_DIR RUN_NAME
# Lưu các exports của study, không lưu các biến môi trường khác.
declare -p TRAIN_MODEL_ROOT DRAFT_CHECKPOINT MODEL DATASET_PATH TARGET_ADAPTER \
  MAX_TARGET_OPTIMIZER_STEPS NUM_EPOCHS TARGET_LR DRAFT_LR OPD_PROJECTOR_LR \
  BATCH_SIZE ACCUMULATION_STEPS DRAFT_ACCUMULATION_STEPS RESPONSES_PER_PROMPT \
  GRPO_ITERATION_NUM BETA EPSILON TEMPERATURE TOP_P GEN_MAX_LENGTH MAX_PROMPT_LENGTH \
  VERIFICATION_CAPACITY MAX_VERIFICATION_NUM MAX_DRAFT_K MAX_DRAFT_TOKEN_LENGTH \
  MIN_DRAFT_TOKEN_LENGTH DRAFT_TOKEN_LENGTH_C MODEL_DTYPE ATTENTION_IMPLEMENTATION \
  IS_TRAIN_DRAFT OPD_UPDATE_STREAM OPD_BACKEND OPD_SAMPLER_MODE OPD_PROPOSAL_MODE \
  OPD_DENSE_IMPLEMENTATION OPD_AUTO_TUNE_IF_MISSING OPD_PROPOSAL_PROFILE OPD_PROFILE OPD_DIAGNOSTICS \
  STATISTICAL_TIME > "$TRAIN_MODEL_ROOT/study.env"
echo "$TRAIN_MODEL_ROOT/study.env"
```

Seed 42, A initialization/seed, rank/top-k/fast LR, interval/sharing/reset/weights
được launcher thiết lập tường minh. Budget 1000 là actual target optimizer steps;
100 epochs là ceiling, không bảo đảm budget nếu dataset hết trước. Kiểm tra
actual counter trong summary. Nếu reference dùng budget khác, đổi chung budget
trước khi bắt đầu cả năm. Không retune riêng từng nhánh. Kernel hash mới làm
profile cũ không còn tương thích; cùng cơ chế fallback chính xác hiện tại áp
dụng nếu không có profile phù hợp, không tự tune trong các lệnh này.

## Dry-run cả năm

In command mà không load model hoặc chạy training:

```bash
for name in interval5 interval10 per_prompt_group carry_b accepted_only; do
  DRY_RUN=true RESUME= bash scripts/run_spark_ablation.sh "$name"
done
```

## Đúng năm lệnh training

Dấu `&` chạy đồng thời, mỗi process một GPU; `wait` chờ cả năm. Muốn chạy tuần
tự, bỏ `&` và đổi tất cả `CUDA_VISIBLE_DEVICES` về GPU muốn dùng.

```bash
RESUME= CUDA_VISIBLE_DEVICES=0 bash scripts/run_spark_ablation.sh interval5 &
RESUME= CUDA_VISIBLE_DEVICES=1 bash scripts/run_spark_ablation.sh interval10 &
RESUME= CUDA_VISIBLE_DEVICES=2 bash scripts/run_spark_ablation.sh per_prompt_group &
RESUME= CUDA_VISIBLE_DEVICES=3 bash scripts/run_spark_ablation.sh carry_b &
RESUME= CUDA_VISIBLE_DEVICES=4 bash scripts/run_spark_ablation.sh accepted_only &
wait
```

Mỗi configuration có root riêng dưới `$TRAIN_MODEL_ROOT/ablations/`, chứa
`active_run`/`latest_run` riêng. Mỗi run có timestamp/UUID. Không dùng chung
`RUN_DIR` hoặc chạy hai process của cùng configuration/root cùng lúc.

## Resume từng configuration

Trong terminal mới, source **đường dẫn study.env đã in ở bước chuẩn bị**, kích
hoạt cùng venv, unset RUN_DIR/RUN_NAME. Không chạy lại bước copy draft.

```bash
source /DUONG_DAN_STUDY_CU/study.env
unset RUN_DIR RUN_NAME
RESUME=auto CUDA_VISIBLE_DEVICES=0 bash scripts/run_spark_ablation.sh interval5
RESUME=auto CUDA_VISIBLE_DEVICES=1 bash scripts/run_spark_ablation.sh interval10
RESUME=auto CUDA_VISIBLE_DEVICES=2 bash scripts/run_spark_ablation.sh per_prompt_group
RESUME=auto CUDA_VISIBLE_DEVICES=3 bash scripts/run_spark_ablation.sh carry_b
RESUME=auto CUDA_VISIBLE_DEVICES=4 bash scripts/run_spark_ablation.sh accepted_only
```

Chọn dòng tương ứng run bị ngắt. Thêm `&` nếu muốn resume nhiều GPU đồng thời.
`RESUME=auto` tìm checkpoint trong đúng configuration root, giữ logs/resume
rewind hiện tại và báo lỗi nếu không có checkpoint, không âm thầm bắt đầu mới.
Có thể dùng `RUN_DIR=/path/to/original/run RESUME=auto` để chọn một run cũ cụ thể.
Các flags OPD/A/sharing/reset/selection/LRs không khớp checkpoint sẽ bị từ chối.

## Logs và completion

Toàn bộ `metrics.jsonl`, `timing.csv`, `rollout_timing.csv`, `summary.json`,
statistics, console logs/tqdm và CSV migration/append được giữ. Config identifier,
init/seed/mode, interval, B scope/lifecycle và feedback selection có trong
metadata, iteration/step logs và summary. Counter lấy từ packet aggregated có
sẵn; không có sync hoặc copy GPU→CPU mới mỗi round để logging.

`opd_updates` đếm số **ma trận B được update với valid feedback**. Trong grouped
mode một verification batch có thể update nhiều B; actual update frequency =
B updates / verification batch rounds có thể >1. Feedback frequency = feedback
rounds / verification batch rounds; không nhầm hai tỷ lệ này. Các counter
selected/visited/frontier, AAL, acceptance, generation/E2E throughput giữ định
nghĩa hiện tại. Hash/change norm A chỉ đọc ở startup/checkpoint/termination.

```bash
# Chọn một cấu hình, ví dụ carry_b:
config_root=$(find "$TRAIN_MODEL_ROOT/ablations" -mindepth 1 -maxdepth 1 -type d -name 'carry_b__*' -print -quit)
run_dir=$(readlink -f "$config_root/active_run")
tail -f "$run_dir/logs/console.log"
# Các lệnh độc lập để xem CSV/JSONL hoặc checkpoint đã lưu:
tail -n 3 "$run_dir/logs/rollout_timing.csv"
tail -n 3 "$run_dir/logs/timing.csv"
tail -n 2 "$run_dir/logs/metrics.jsonl"
ls -lh "$run_dir/checkpoints/resume/latest.pt"
```

Kiểm tra cả năm bằng CPU, không load tensors/model:

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
root=Path(os.environ['TRAIN_MODEL_ROOT'])/'ablations'
budget=int(os.environ['MAX_TARGET_OPTIMIZER_STEPS'])
for name in ('interval5','interval10','per_prompt_group','carry_b','accepted_only'):
    folders=list(root.glob(name+'__*'))
    if len(folders)!=1:
        print(name, 'MISSING/AMBIGUOUS ROOT');continue
    path=folders[0]/'active_run'/'summary.json'
    if not path.exists():
        print(name, 'RUNNING/INTERRUPTED (no final summary)');continue
    s=json.loads(path.read_text())
    print(name, 'BUDGET REACHED' if s['target_optimizer_steps']>=budget else 'BELOW BUDGET',
          {k:s.get(k) for k in ('target_optimizer_steps','average_accept_length',
           'generation_tokens_per_s','cumulative_end_to_end_tokens_per_s',
           'total_verification_rounds','opd_feedback_rounds','opd_skipped_rounds',
           'opd_actual_b_updates','opd_actual_update_frequency','opd_b_sharing_scope',
           'opd_b_reset_mode','opd_feedback_selection','opd_a_change_norm')})
PY
```

Summary của session cũ có thể vẫn còn trong lúc resume đang chạy; đối chiếu
console/checkpoint timestamp và actual steps. Summary dưới budget không được
coi là đã hoàn thành experimental budget.

## Validation

Tests mới ở `tests/test_spark_five.py`; tests scheduler/default parity/checkpoint/
logging/launcher hiện có được mở rộng, không chạy full training study. CPU oracle
và CUDA/Triton smoke dùng tiny Qwen, gồm grouped sparse/fused/gemm, invalid group,
canonical↔physical mapping, EOS compaction, async streams, lifecycle carry,
rank-local checkpoint, bitwise restore và train/resume/logs của cả năm variants.
Kết quả cụ thể của lần sửa này xem `validation/spark_five_20261012/results.json`.
Không suy diễn throughput B200 từ GPU khác; chưa chạy full Qwen3B/SimpleLR hoặc
full study trên B200 tại workspace này.
