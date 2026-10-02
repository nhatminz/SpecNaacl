import sys

import pytest

from scripts import validate_environment


def set_python_version(monkeypatch, version):
    monkeypatch.setattr(sys, "version_info", (*version, "final", 0))
    monkeypatch.setattr(sys, "version", ".".join(map(str, version)))


def test_python_only_accepts_pinned_version_without_dependencies(monkeypatch):
    set_python_version(monkeypatch, (3, 12, 12))
    monkeypatch.setattr(sys, "argv", ["validate_environment.py", "--python-only",
                                      "--requirements", "missing-requirements.txt"])

    def unexpected_dependency_check(*args):
        pytest.fail("python-only validation must not inspect or import dependencies")

    monkeypatch.setattr(validate_environment, "version", unexpected_dependency_check)
    monkeypatch.setattr(validate_environment.importlib, "import_module",
                        unexpected_dependency_check)
    validate_environment.main()


@pytest.mark.parametrize("version", [(3, 11, 12), (3, 12, 11), (3, 12, 13), (3, 13, 0)])
def test_wrong_python_version_reports_interpreter_and_recovery(monkeypatch, version):
    set_python_version(monkeypatch, version)
    with pytest.raises(RuntimeError) as error:
        validate_environment.validate_python_version()
    message = str(error.value)
    assert f"Python 3.12.12 is required; found {sys.version}" in message
    assert sys.executable in message
    assert "uv venv --python 3.12.12 --seed .venv-py31212" in message
    assert "export PYTHON_BIN=" in message
    assert "ENVIRONMENT.md" in message
