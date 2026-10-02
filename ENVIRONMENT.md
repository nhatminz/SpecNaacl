# Reproducible B200 environment

The supported Python versions are 3.12.12 and 3.12.13. New environments use
3.12.13, pinned in `.python-version`. The library stack is PyTorch 2.13.0 with the CUDA 13.0
wheel, Transformers 5.12.1, and SGLang 0.5.18. CUDA 13.x requires an NVIDIA
driver from the R580 branch or newer. The PyTorch wheel carries its CUDA runtime
libraries; a system CUDA toolkit is not required unless building an optional
CUDA extension.

## Online installation

For a new environment, install Python 3.12.13 with `uv`, then create the
environment from that managed interpreter. An existing 3.12.12 or 3.12.13
environment can be kept; use the validation commands below:

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
nvidia-smi
uv python install 3.12.13
uv venv --python 3.12.13 --seed .venv
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

Run this on an Internet-connected Linux x86-64 machine with Python 3.12.13, then
copy both the project and `wheelhouse/` to the B200 machine. `pip wheel` is used
instead of `pip download` so source distributions are built before transfer.

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
uv python install 3.12.13
uv venv --python 3.12.13 --seed .wheel-builder
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
python3.12 scripts/validate_environment.py --python-only
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
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

## Using an existing Python 3.12.13 environment

Both launchers accept 3.12.12 and 3.12.13. `.python-version` selects 3.12.13
for new uv environments; it does not change the interpreter in an existing venv.
CPython documents ABI compatibility across patch releases within the same minor
release when builds match (see [C API stability](https://docs.python.org/3/c-api/stable.html)).
The project still checks dependency versions, imports, and required runtime APIs.

If an older checkout reports `Python 3.12.12 is required; found 3.12.13`, update
the project code and keep the existing environment:

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" --version
"$PYTHON_BIN" scripts/validate_environment.py --python-only
"$PYTHON_BIN" scripts/validate_environment.py --require-cuda
"$PYTHON_BIN" -m pip check
```

Then rerun the original pretrain command in the same shell. Export `PYTHON_BIN`
explicitly because an older value may still point to a different environment.
If dependency validation reports missing or incompatible packages, install the
pinned dependencies using the online or offline commands above.
