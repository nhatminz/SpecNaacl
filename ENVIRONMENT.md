# Reproducible B200 environment

The supported environment is Python 3.12.12, PyTorch 2.13.0 with the CUDA 13.0
wheel, Transformers 5.12.1, and SGLang 0.5.18. CUDA 13.x requires an NVIDIA
driver from the R580 branch or newer. The PyTorch wheel carries its CUDA runtime
libraries; a system CUDA toolkit is not required unless building an optional
CUDA extension.

## Online installation

Install the exact Python patch with `uv`, then create the environment from that
managed interpreter:

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
nvidia-smi
uv python install 3.12.12
uv venv --python 3.12.12 .venv
source .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install --no-deps -e third_party/SpecForge --no-build-isolation

python scripts/validate_environment.py --require-cuda
python -m pip check
python -m compileall -q .
pytest -q
```

Confirm that `torch.__version__` is `2.13.0+cu130` and `torch.version.cuda` is
`13.0`:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
```

## Preparing an offline wheelhouse

Run this on an Internet-connected Linux x86-64 machine with Python 3.12.12, then
copy both the project and `wheelhouse/` to the B200 machine. `pip wheel` is used
instead of `pip download` so source distributions are built before transfer.

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
uv python install 3.12.12
uv venv --python 3.12.12 .wheel-builder
source .wheel-builder/bin/activate
python -m pip install --upgrade pip wheel setuptools
mkdir -p wheelhouse

python -m pip wheel --wheel-dir wheelhouse torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip wheel --wheel-dir wheelhouse --find-links wheelhouse \
  -r requirements.txt
```

On the offline B200 machine:

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
python3.12 -c 'import sys; assert sys.version_info[:3] == (3, 12, 12), sys.version'
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links wheelhouse torch==2.13.0
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
python -m pip install --no-deps -e third_party/SpecForge --no-build-isolation

python scripts/validate_environment.py --require-cuda
python -m pip check
python -m compileall -q .
pytest -q
```

No `requirements-optional.txt` is needed for this pipeline. SpecForge's EAGLE
implementation catches the optional `flash_attn` v2 import and falls back to
PyTorch flex attention. SGLang 0.5.18 has its own mandatory `flash-attn-4`
dependency; it remains governed by SGLang's package metadata and must be present
for `pip check` to pass.
