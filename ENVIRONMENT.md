# Reproducible B200 environment

The supported Python versions are 3.12.12 and 3.12.13. New environments use
3.12.13. The optional `.python-version` file helps uv select that version;
runtime validation does not read it, so it can be omitted on the server.
The library stack is PyTorch 2.13.0 with the CUDA 13.0
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

No `requirements-optional.txt` is needed for this pipeline. New EAGLE pretraining
runs request the optional standard `flash_attn` v2 backend and explicitly fail
if its CUDA forward/backward interface is unavailable; there is no silent
fallback. Install a compatible build or set `PRETRAIN_ATTENTION_BACKEND=sdpa`
(or `flex_attention`) explicitly. Resumed runs retain their saved backend.
SGLang 0.5.18 has its own mandatory `flash-attn-4`
dependency; it remains governed by SGLang's package metadata and must be present
for `pip check` to pass.

### Recovery: `cannot import name 'flash_attn_varlen_func'`

This means the selected EAGLE `fa` backend cannot import the standard varlen
interface it uses. An importable `flash_attn` namespace or a successful SGLang
dependency check is not proof that this interface is available. The launcher
stops before feature capture/training; this particular error is not an OOM.
Do not remove SGLang dependencies or change the pinned Torch stack to bypass it.

Choose the existing backend that does not need the external FA interface:

```bash
export PRETRAIN_ATTENTION_BACKEND=sdpa
# Rerun the original pretrain command with its model/dataset/batch settings.
bash pretrain_qwen25_3b.sh  # include your original environment assignments
```

To check the choice independently of dataset preparation:

```bash
python scripts/check_pretrain_attention.py --backend sdpa
# If you want to use standard FA, validate its real CUDA forward/backward:
CUDA_VISIBLE_DEVICES=0 python scripts/check_pretrain_attention.py --backend fa --probe
```

SDPA is an explicit choice, not a silent fallback, and its existing EAGLE cached
TTT implementation is unchanged. Checking `sdpa` only validates the selection;
it is not a CUDA execution/throughput benchmark. On backend-check failure the
launcher now prints the SDPA override and the prepared run path. A model wrapper
can reuse that directory with `RESUME=/absolute/run/path`. If a real checkpoint
exists, normal resume validation still applies; never change its saved backend
implicitly. Batch size 64 in the reported command was not changed by this fix;
whether it fits at length 2048/TTT 7 must be established on the GPU separately.

## Using an existing Python 3.12.13 environment

Both launchers accept 3.12.12 and 3.12.13 without a `.python-version` file.
The optional file selects 3.12.13 for new uv environments; it does not change
the interpreter in an existing venv. The explicit `--python 3.12.13` option
in the installation commands also works without this file (see
[uv Python version files](https://docs.astral.sh/uv/concepts/python-versions/#python-version-files)).
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
