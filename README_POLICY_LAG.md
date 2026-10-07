# Policy-lag entrypoints

Policy-lag scripts hiện dùng core và checkpoint FastGRPO, không có SpecForge
config/mapping. Đặt `TARGET_MODEL_PATH`, `DRAFT_CHECKPOINT` và dataset local
rồi chạy `run_policy_lag_analysis.sh`; xem `--help` của `policy_lag_analysis.py`
cho các chế độ phân tích. `run_policy_lag_analysis_b200.sh` chuẩn bị DAPO split
trước khi gọi wrapper. Đây là công cụ phụ, chưa kiểm thử production policy-lag
sweep trong lần rewrite này.

Môi trường: [ENVIRONMENT.md](ENVIRONMENT.md). Pretrain/train chính:
[huongdanchay.md](huongdanchay.md). Tài liệu framework cũ được giữ tại
`legacy_tests/specforge/docs/` để tham khảo lịch sử.
