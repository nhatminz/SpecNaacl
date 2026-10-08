# DataLoader iteration logging

Áp dụng chung cho `fastgrpo` và `opd_reflex`. Chín cột mới được append cuối
`logs/rollout_timing.csv`; giữ nguyên tên/thứ tự các cột cũ, một row mỗi
iteration kể cả prompt/answer/reward-filtered skips. Resume replay các batch
đã log không tạo thêm row. Header cũ được migrate bằng cơ chế resume hiện có;
cột mới của historical rows để trống, không đoán lại thời gian.

| Cột | Dữ liệu / định nghĩa |
|---|---|
| `iter_wall_time_s` | `perf_counter()` ở writer begin → finish tại finally; bao gồm generation, training và CPU/checkpoint work trong iteration, không có startup/model loading |
| `iter_generation_tokens_per_s` | `iter_rollout_tokens / iter_generation_time_s`; denominator 0 trả 0 |
| `iter_end_to_end_tokens_per_s` | `iter_rollout_tokens / iter_wall_time_s`; denominator 0 trả 0 |
| `iter_draft_train_time_s` | Host duration đã có trong draft phase timer, gồm backward và draft optimizer boundary |
| `iter_target_train_time_s` | Tổng các host duration đã có trong target phase timers của iteration, gồm target optimizer boundary |
| `iter_draft_sparse_kl`, `iter_draft_sparse_tv` | Reuse Python scalar từ rollout log; không tính KL/TV mới |
| `iter_draft_update_committed` | Reuse bool đã có của iteration; early skips là False |
| `iter_draft_updates_cumulative` | Counter `draft_step` đã checkpoint/restore, tiếp tục qua resume; không reset theo continuation trace |

Các phase time là CPU wall intervals, không phải CUDA kernel durations.
Generation time tiếp tục theo counter sampler đã có. KL/TV trong core
FastGRPO hiện tại là placeholder 0 từ trainer; CSV phản ánh đúng scalar này,
không coi nó là KL/TV vừa đo. Nếu diagnostic chưa có Python scalar hoặc chỉ
có tensor, để trống thay vì đọc tensor để logging.

Không thêm GPU sync, CUDA events, kernels/reductions, tensor-to-CPU transfer,
model forward, per-token/round logging, file open/flush hoặc terminal printing.
Buffer/flush interval và checkpoint/append semantics giữ nguyên.

Validation: 35 metrics/logging tests pass; 2 real CUDA training/resume cases
pass cho hai methods, bao gồm ratio/time fields, cumulative draft updates,
RNG/weights/cadence và metrics rows. Source audit giữ nguyên mọi tensor/GPU/
event call và các objective functions; generation/sampler/step timer modules
không thay đổi. Raw logs: `validation/iteration_logging_20261008/`.
