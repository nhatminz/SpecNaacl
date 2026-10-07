# FastGRPO core rewrite — 2026-10-08

## Source và kiến trúc

`sources/FastGRPO/` là bản sao nguyên byte từ `../FastGRPO-main/`.
`SOURCE_MANIFEST.json` ghi SHA256; test đối chiếu cả bản sao và checkout gốc.
Không sửa `FastGRPO-main`.

| Thành phần | Runtime hiện tại | Nguồn / thay đổi |
|---|---|---|
| Draft architecture | `helper/modeling_draft.py` | Nguyên byte upstream: EagleFS, DraftDecoderLayer, states/logits MLP và norm |
| Wrapper checkpoint | `helper/fastgrpo_model.py` | Shared target embedding/lm_head; thêm A và cache storage chỉ khi bật OPD |
| Baseline rollout | `helper/fastgrpo_generate.py` | Source upstream; chỉ thêm host metric counters tại boundary có sẵn |
| Dispatch | `helper/specualtive_generate.py` | Hai method; baseline không khởi tạo OPD/kernel/tuner |
| Pretrain | `train_draft.py`, `helper/pretrain_data.py` | Loss/collation upstream, thêm launcher, distributed coordination và resume |
| Online draft / GRPO loss | `helper/fastgrpo_training.py` | Hàm lấy từ source gốc, truyền globals thành tham số |
| OPD rollout | `helper/opd_generate.py` | Cùng FastGRPO architecture/tree/sampler, port KV/tree/history tối ưu |
| OPD teacher | `helper/opd_sampling.py` | Sampler upstream, callback trích metadata trước khi release probabilities |
| OPD correction | `helper/opd_reflex*.py` | Full V; Triton sparse/fused/GEMM, update B và accumulate gradient A |

Không còn import runtime SpecForge/EAGLE3/SGLang, compact output head,
vocabulary-mapping file, feature-layer concatenation hay TTT objective.
`third_party/SpecForge` và `legacy_tests/specforge` được giữ làm lịch sử;
không nằm trên runtime PYTHONPATH hay test discovery mặc định.

## Training semantics

Pretrain giữ `2*SmoothL1 + 0.1*soft CE`, đúng slicing next-feature/next-token,
assistant mask, per-sequence normalization, teacher detach, FP32 softmax/log,
AdamW và cosine-with-min-lr. Không flush partial accumulation cuối training,
giống source. Các backward chỉ phục vụ gradient-norm diagnostics của upstream
không được chạy lại. Default ShareGPT / 5 epochs. Có distributed gradient
averaging, thống nhất skip invalid batch giữa ranks và sampler có epoch seed.

Online giữ thứ tự rollout → draft backward → draft optimizer → target GRPO.
Cả hai method dùng chung LR/objective/cadence. Giữ cả đặc điểm upstream:
final online microbatch không chia `draft_accumulation_steps`, trong khi các
microbatch trước có chia; test đối chiếu cả accumulation 1 và 2. Default là 1.
Target GRPO dùng objective full-vocabulary nguyên gốc.

A là parameter FP32 persistent, checkpointed và được cập nhật tại draft
optimizer boundary. B_fast rollout-local reset mỗi rollout, không checkpoint.
Không thêm transformer forward cho OPD teacher/update.

## Paths và interface

Giữ sáu pretrain launcher, mười hai train launcher, model/data overrides,
`outputs/pretrain/<model_key>/<run_name>` và `outputs/train/<model_key>/<run_name>`.
Hai method đọc cùng `latest_checkpoint`. Chuỗi tên pretrain lịch sử được giữ;
weights bên trong đã chuyển sang FastGRPO, không tương thích checkpoint cũ.

Pretrain checkpoint có `draft_model`, training state chứa optimizer, scheduler,
pending gradients, RNG từng rank và iterator position. Train checkpoint chứa
LoRA target, full draft/A, hai optimizer, pending gradients/A feedback,
step/epoch/iteration, RNG và metrics state. Iterator dùng generator riêng để
resume không tiêu hao model RNG. Target base phải được load từ cùng model path.

Giữ `logs/metrics.jsonl`, `logs/timing.csv` (một row / GRPO step),
`logs/rollout_timing.csv` (một row / DataLoader iteration), `summary.json`.
Metric AAL/time/throughput/acceptance và OPD counters tiếp tục được ghi.
Một số tên metric lịch sử có chữ `compact` được giữ để schema không vỡ,
nhưng vocabulary runtime luôn là full target vocabulary.

`MAX_VERIFICATION_NUM` mặc định chuyển từ 512 sang 160 như upstream; với
K=8/depth=5, cây chỉ có 264 node khi batch còn 1. Giữ 512 sẽ khiến native
`torch.topk` gọi với K vượt số node. `VERIFICATION_CAPACITY=512` vẫn giữ.

## Optimization và giới hạn numerical

Port các kernel Triton, sparse/fused/GEMM dispatch, async feedback, GPU state
selection/union/KL/grad, persistent growable KV, swap-remove compaction,
attention/position/tree buffers và contiguous rollout history.
Không copy toàn bộ KV history mỗi round; vẫn phải copy khi pool grow hoặc
một live row được chuyển chỗ. Benchmark counters ghi các copy này thực tế.

Teacher reuse đúng sampler probabilities sau temperature/top-p/top-k và
sorted/topk intermediates. Không thêm full-vocab softmax/sort hay giữ `[N,V]`
qua async update. Sửa lỗi Triton BF16 bitcast khi lấy Top16 từ sorted metadata.
Async dispatch dùng active-count snapshot một round cũ, không thêm device
scalar read chỉ để chọn proposal backend.

Baseline giữ native TopK(K). OPD dùng Top16[:K], FP32 correction/normalization;
IDs khi tie và rounding có thể khác baseline. B=0 bảo toàn raw logits, không
đồng nghĩa hai method luôn có cùng trajectory. Confidence pruning vẫn dùng
native torch.topk như source, không thay bằng luật tie khác.

CUDA atomic reduction cho gradient A không đảm bảo bitwise deterministic.
Integration resume kiểm tra bit-for-bit toàn bộ target/draft/A, optimizer và
RNG ngay sau load, trước rollout tiếp theo. Khi so hai trajectory chạy riêng,
OPD cho phép 2 FP32 epsilon relative ở A và 8 FP32 epsilon theo tensor scale
ở momentum A được tính mới; các weights/optimizer khác và RNG vẫn phải khớp
bit-for-bit. Đã quan sát A lệch một phần tử, 3.73e-9 (một ULP). Tolerance này
không áp cho việc load checkpoint.

## Validation

Kết quả cuối: **173 passed**, 44 warnings, 53.94 giây. Warnings là thông báo
thiếu profile OPD tương thích trong fixture, dùng fallback chưa hiệu chỉnh.
Kiểm tra AST 58 file Python, `bash -n` 36 shell scripts và source checkout
check cũng pass. Raw logs/reports: [validation artifacts](validation/fastgrpo_rewrite_20261008/README.md).

Lệnh cho môi trường đã cài dependencies:

```bash
python -m pytest -q tests
python scripts/check_training_sources.py --backend fastgrpo
```

Các kiểm thử gồm:

- Qwen2/Qwen3 transformer thật kích thước nhỏ so với source upstream: tokens,
  RNG, accepted lengths, masks/tree, verification counts, forward counts,
  draft-input history.
- Pretrain loss/gradients/update và ShareGPT mask so source; online loss/gradient
  và optimizer update với accumulation 1 và 2.
- Sampler FP32/BF16/FP16, top-p/top-k và invalid rows giữ tokens/RNG upstream.
- B=0; state selection; union/tail KL; q-p; B update; A gradient CPU/GPU oracle.
- Full V=151936, BF16/FP16, active rows 0/16/1024: sparse/fused/GEMM bitwise
  proposal parity, probabilities đối chiếu dense oracle.
- Async/sync feedback, persistent pool reuse, KV swap/remove, history,
  no-extra-forward và teacher metadata không giữ full probabilities.
- Profile fingerprint mismatch, full-vocab checkpoint inspection, model dedup,
  host-only interpolation, ba backend dispatch và tuner CUDA thật.
- Sáu cặp launcher cùng config/checkpoint; ShareGPT defaults và SimpleLR parquet.
- Subprocess pretrain liên tục / dừng-resume và GRPO liên tục / dừng-resume cho
  cả hai method: model, optimizer, scheduler khi có, RNG, metrics rows và cadence.

Môi trường thực tế: RTX 3090 24GB, Python 3.10, Torch 2.5.1+cu124,
Triton 3.1.0, Transformers 4.51.3, PEFT 0.17.1. Pin deployment là Torch 2.8.0 /
Triton 3.4.0; chưa chạy bộ test với đúng hai pin này trên B200.

Tiny frozen-rollout sweep đã chạy cả baseline, OPD sync/async và profiling
replay. Đây là kiểm tra pipeline, không phải benchmark model đã pretrain đủ.
Trong fixture V=97, OPD chậm hơn baseline khoảng 11–12%; workload nhỏ này
không chứng minh hay phủ định throughput B200/full-model. Không tuyên bố
“không regression throughput” khi chưa có phép đo phù hợp. Cần pretrain thật,
tune full-V trên B200 rồi chạy `scripts/sweep_opd_reflex.sh` cùng workload.

Các tài liệu/số đo trước rewrite ở `legacy_tests/specforge/docs/` không được
coi là bằng chứng performance cho architecture mới.

## Smoke với weights và dữ liệu local thật

Đã chạy Qwen2.5-1.5B-Instruct / ShareGPT pretrain 2 optimizer steps,
batch 1, length 128, tối đa 8 records. Loss lần lượt 4.708617 và 4.075339;
checkpoint draft/optimizer/scheduler/RNG và latest links được ghi.
Đây là smoke test, không phải checkpoint đã pretrain đủ 5 epochs.

Dùng cùng checkpoint đó và một prompt SimpleLR, hai responses, max length
128, K=2/depth=3, một seed, một warmup và một measured rollout trên RTX 3090:

| Method | Stream | AAL | Token/s | Target / draft forwards |
|---|---:|---:|---:|---:|
| FastGRPO source | — | 1.092593 | 72.18 | 28 / 81 |
| ReflexOPD | 0 | 1.092593 | 71.87 | 28 / 81 |
| ReflexOPD | 1 | 1.092593 | 72.11 | 28 / 81 |

Không có extra transformer forward trong phép chạy này. Số đo chỉ một rollout,
chưa profile-tune, draft mới học 2 bước; chênh lệch 0.09–0.42% wall time không
đủ để kết luận speedup/regression. Chưa đo workload mặc định dài/batch lớn,
B200 hay distributed nhiều GPU.
