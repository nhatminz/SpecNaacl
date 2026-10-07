# Autotuning proposal full vocabulary

```bash
bash scripts/tune_opd_proposals.sh
OPD_TUNE_MODELS=qwen3_1p7b bash scripts/tune_opd_proposals.sh
```

Mặc định tune cả sáu model. Input là
`outputs/pretrain/<key>/latest_checkpoint` và `latest_target_config.json`.
Một custom model có thể override `TARGET_CONFIG` và `DRAFT_CHECKPOINT` khi
`OPD_TUNE_MODELS` chỉ chứa một key. Tuner kiểm tra shape FastGRPO draft/head,
không load target transformer. Không có compact-vocab hay mapping input.

Execution key gồm GPU/compute capability, full target V, rank, dtype, TopK,
kernel/compiler fingerprint. Profile lưu tại
`outputs/benchmarks/opd_proposals`; không key theo tên model hay hidden size vì
phần projection h@A không thuộc proposal benchmark. Profile tương thích có
thể dùng chung giữa model cùng execution key. Missing model warning+skip;
profile sai fingerprint bị từ chối.

Runtime `OPD_PROPOSAL_MODE=auto`, `OPD_DENSE_IMPLEMENTATION=auto` dùng measured
cost theo contexts/active rows để chọn sparse/fused/GEMM. Thiếu profile thì
warning và fallback chưa hiệu chỉnh; không tự báo speedup. Có thể opt-in
`OPD_AUTO_TUNE_IF_MISSING=1`. Baseline không lookup hay tune OPD.

Async stream dùng snapshot một round cũ cho dispatch và bound workspace an toàn,
không thêm host wait để lấy active count mới. Retune sau khi đổi kernel hoặc
GPU. Kiểm chứng lựa chọn bằng frozen-rollout sweep trên checkpoint thật;
chi phí kernel riêng không chứng minh throughput end-to-end.
