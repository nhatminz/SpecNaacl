# Môi trường FastGRPO

Có thể giữ môi trường hiện có nếu các import/API và smoke test của validator
pass. `requirements.txt` là bộ pin để cài môi trường mới có thể tái lập; pin
không còn là allowlist tuyệt đối bắt buộc cho mọi môi trường đang dùng.

## Môi trường server đã có thư viện

Stack được khai báo hỗ trợ gồm Torch 2.13.0, Transformers 5.12.1, PEFT 0.21.1,
Datasets 5.0.1, Accelerate 1.15.0, Safetensors 0.8.0, NumPy 2.3.5,
Pandas 3.0.6, tqdm 4.70.1, math-verify 0.9.0, latex2sympy2-extended 1.11.0,
Triton 3.7.1, Matplotlib 3.11.2 và Packaging 26.3.

Đồng bộ code mới, giữ nguyên `.venv`, rồi chạy từ `SpecNaacl`:

```bash
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" scripts/validate_environment.py --require-cuda
bash pretrain_qwen25_3b.sh
```

Validator kiểm tra Python >=3.10, version trong API family được hỗ trợ,
import thật và các API mà runtime dùng. Sau đó chạy model Qwen2 nhỏ trên GPU
để kiểm tra SDPA, pretrain loss/backward, optimizer/scheduler, decoder/cache,
PEFT LoRA và checkpoint roundtrip. Check thất bại vẫn chặn launcher với lỗi
cụ thể. Không có bước tự cài, nâng/hạ hay thay đổi thư viện trong launcher.

`helper/transformers_compat.py` chuyển tiếp API Transformers 5.x: singular
`past_key_value`/tuple output của speculative decoder sang plural API/tensor,
và cung cấp legacy key/value lists trên cache mới. Normal target forward vẫn
nhận tensor theo API native. Transformers 4.x dùng decoder/cache gốc.
Architecture, loss, sampling, verifier và parameter/checkpoint names giữ nguyên.

Môi trường kiểm thử cùng các phiên bản server phía trên đã pass validator và
CPU forward/backward/checkpoint với Python 3.12.12 / Torch 2.13.0+cpu.
Kiểm thử CUDA riêng dùng RTX 3090 / Torch 2.5.1+cu124, Transformers 5.12.1,
PEFT 0.21.1, Datasets 5.0.1 và Accelerate 1.15.0. CUDA build Torch 2.13.0 /
Triton 3.7.1 trên B200 cần pass GPU probe tại server; máy phát triển không chạy
được CUDA 13 với driver hiện có. Không dùng CPU test để tuyên bố B200 throughput.

## Môi trường mới hoặc cần đúng pin tái lập

```bash
python -m pip install -r requirements.txt
python scripts/validate_environment.py --strict-versions --require-cuda
python scripts/check_training_sources.py --backend fastgrpo
python -m pytest -q tests
```

`--strict-versions` yêu cầu đúng pin cài đặt. Chế độ mặc định cho phép phiên bản
khác trong các family được hỗ trợ và vẫn chạy probe. Không cài SpecForge/SGLang
cho core này. Các launcher dùng model/data local, bật Hugging Face offline mode
và tiếp tục cho override `MODEL`, `DATA_ROOT`, `DRAFT_CHECKPOINT`, `OUTPUT_ROOT`
và `PYTHON_BIN`.
