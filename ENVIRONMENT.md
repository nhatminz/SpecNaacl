# Môi trường FastGRPO

Runtime kiểm tra Python >=3.10 và các phiên bản trong `requirements.txt`.
Các pin chính: Torch 2.8.0, Triton 3.4.0, Transformers 4.51.3, PEFT 0.17.1.
Transformers 4.51.3 cung cấp Qwen3 và giữ API decoder/cache mà FastGRPO dùng.
Không cài editable SpecForge hay SGLang cho core này.

Trong virtual environment riêng, từ thư mục `SpecNaacl`:

```bash
python -m pip install -r requirements.txt
export PYTHON_BIN="$(command -v python)"
python -m pip check
python scripts/validate_environment.py --require-cuda
python scripts/check_training_sources.py --backend fastgrpo
python -m pytest -q tests
```

Các launcher đọc model/data local và bật Hugging Face offline mode. Chúng không
tự tải weights. `MODEL`, `DATA_ROOT`, `DRAFT_CHECKPOINT`, `OUTPUT_ROOT` và
`PYTHON_BIN` vẫn có thể override. Không dùng môi trường Transformers 5.x từ
core cũ vì API decoder/cache khác source FastGRPO.

Kiểm thử rewrite tại máy phát triển dùng RTX 3090, Python 3.10,
Torch 2.5.1+cu124, Triton 3.1.0, Transformers 4.51.3 và PEFT 0.17.1.
Đó là môi trường kiểm thử thực tế, khác pin triển khai Torch/Triton phía trên.
Chưa xác nhận toàn bộ pin hoặc throughput trên B200; xem
[phạm vi kiểm thử](FASTGRPO_REWRITE.md).
