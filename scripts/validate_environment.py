#!/usr/bin/env python3
"""Fail-fast validation for the pinned SpecNaacl B200 environment."""

from __future__ import annotations

import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import sys


IMPORT_NAMES = {
    "huggingface-hub": "huggingface_hub",
    "latex2sympy2-extended": "latex2sympy2_extended",
    "math-verify": "math_verify",
    "openai-harmony": "openai_harmony",
    "pyyaml": "yaml",
    "typing-extensions": "typing_extensions",
}
SUPPORTED_PYTHON_VERSIONS = ((3, 12, 12), (3, 12, 13))


def validate_python_version():
    if sys.version_info[:3] in SUPPORTED_PYTHON_VERSIONS:
        return
    required = ".".join(map(str, SUPPORTED_PYTHON_VERSIONS[-1]))
    supported = " or ".join(".".join(map(str, item)) for item in SUPPORTED_PYTHON_VERSIONS)
    environment = f"venv-py{required.replace('.', '')}"
    raise RuntimeError(
        f"Python {supported} is required; found {sys.version.split()[0]}\n"
        f"Interpreter: {sys.executable}\n"
        "Create a supported environment from the project directory:\n"
        f"  uv python install {required}\n"
        f"  uv venv --python {required} --seed {environment}\n"
        f"  source {environment}/bin/activate\n"
        '  export PYTHON_BIN="$(command -v python)"\n'
        "Install the project dependencies in this environment; see ENVIRONMENT.md."
    )


def pinned_requirements(path: Path):
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match is None:
            raise RuntimeError(f"requirement is not exactly pinned: {line}")
        yield match.group(1), match.group(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", type=Path, default=Path("requirements.txt"))
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--python-only", action="store_true",
                        help="check the supported interpreter without importing dependencies")
    args = parser.parse_args()
    validate_python_version()
    if args.python_only:
        return

    failures = []
    for distribution, expected in pinned_requirements(args.requirements):
        try:
            actual = version(distribution)
        except PackageNotFoundError:
            failures.append(f"{distribution}: not installed")
            continue
        if actual.split("+", 1)[0] != expected:
            failures.append(f"{distribution}: expected {expected}, found {actual}")
            continue
        module_name = IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(
                f"{distribution}: import {module_name} failed: "
                f"{type(exc).__name__}: {exc}"
            )
    if failures:
        raise RuntimeError("environment validation failed:\n- " + "\n- ".join(failures))

    torch = importlib.import_module("torch")
    if args.require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) < (10, 0):
            raise RuntimeError(
                f"B200-class compute capability >=10.0 required; found {major}.{minor}"
            )

    backend = importlib.import_module("specforge.offline_capture.sglang_backend")
    if not hasattr(backend, "OfflineSGLangCaptureBackend"):
        raise RuntimeError("SpecForge SGLang offline capture API is unavailable")
    print(
        "environment validation passed: "
        f"python={sys.version.split()[0]} torch={torch.__version__} "
        f"cuda={torch.version.cuda}"
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
