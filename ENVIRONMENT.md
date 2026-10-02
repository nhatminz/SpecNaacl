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
uv venv --python 3.12.12 --seed .venv
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
python scripts/validate_environment.py --python-only
python -m pip install --upgrade pip setuptools wheel

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
uv venv --python 3.12.12 --seed .wheel-builder
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

## Fixing `Python 3.12.12 is required; found 3.12.13`

The launchers require the exact patch pinned in `.python-version`. Activating a
virtual environment created with Python 3.12.13 still uses 3.12.13; installing
3.12.12 does not change that existing environment. Create a separate environment
so the previous `.venv` is preserved:

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
uv python install 3.12.12
uv venv --python 3.12.12 --seed .venv-py31212
source .venv-py31212/bin/activate
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" --version
"$PYTHON_BIN" scripts/validate_environment.py --python-only

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install --no-deps -e third_party/SpecForge --no-build-isolation
python scripts/validate_environment.py --require-cuda
python -m pip check
```

Then rerun the original pretrain command in the same shell. `PYTHON_BIN` is
exported explicitly because an older value may still point to the previous
environment. `--seed` supplies pip for the `python -m pip` commands (see the
[uv venv reference](https://docs.astral.sh/uv/reference/cli/#uv-venv)). If `uv`
is unavailable, install it using the
[official instructions](https://docs.astral.sh/uv/getting-started/installation/).
On an offline server, provision the 3.12.12 interpreter first and use the
wheelhouse installation commands above for the new environment; package wheels
alone do not install Python.
