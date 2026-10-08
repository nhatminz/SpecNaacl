# Giữ môi trường thư viện hiện có — 2026-10-08

Validator chấp nhận stack server đã báo trong các API family được hỗ trợ,
kiểm tra import/API rồi chạy model nhỏ để kiểm tra execution thực tế.
`requirements.txt` giữ pin cho môi trường cài mới; dùng `--strict-versions`
khi muốn bắt buộc đúng pin. Launcher không tự cài hay thay đổi package.

Transformers 5.12.1 cần adapter cho decoder `past_key_values`/tensor output và
cache `layers[].keys/values`. Adapter chỉ chuyển tiếp API. Baseline generation
chỉ đổi import DynamicCache; loss/tree/verifier/sampling và DraftModel nguồn
không đổi. Checkpoint vẫn dùng cùng parameter names. HF 4.51.3 dùng native API.

Kiểm chứng:

| Môi trường | Kết quả |
|---|---|
| Python 3.12.12, Torch 2.13.0+cpu, các package đúng version server đã báo | Validator pass; toàn suite CPU 116 pass, 207 GPU cases skip |
| RTX 3090 / Torch 2.5.1+cu124, HF 5.12.1 / PEFT 0.21.1 / Datasets 5.0.1 / Accelerate 1.15.0 | Validator GPU pass; 350 tests pass |
| HF 4.51.3 / PEFT 0.17.1 | Validator GPU pass; 59 source-parity/API tests pass |
| Qwen2.5-3B-Instruct thật / ShareGPT, HF 5.12.1 trên RTX 3090 | Pretrain 2 steps, finite loss 4.760236 → 4.283566; checkpoint/optimizer/scheduler/RNG được lưu |

Integration resume kiểm tra target/draft/optimizer/RNG bitwise ngay sau load,
cadence và metrics rows. Synthetic fixture dùng ByteLevel/BPE tương thích
Qwen2 và math SDPA cho comparison cần bitwise; production giữ SDPA dispatch.
OPD atomic gradient A có tolerance FP32 đã được khai báo trong test trước đó.

Chưa chạy CUDA build Torch 2.13.0 / Triton 3.7.1 trên B200 tại máy phát triển.
Validator sẽ probe trên chính GPU của server trước khi khởi động training.
Đồng bộ toàn bộ code SpecNaacl mới, giữ `.venv`, rồi chạy:

```bash
python scripts/validate_environment.py --require-cuda
bash pretrain_qwen25_3b.sh
```

Các run/model/data/output conventions giữ nguyên. Validation logs và metrics
nằm ở [validation/environment_compat_20261008](validation/environment_compat_20261008/).
