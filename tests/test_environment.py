import sys

import pytest

from scripts import validate_environment


def set_python_version(monkeypatch, version):
    monkeypatch.setattr(sys, "version_info", (*version, "final", 0))
    monkeypatch.setattr(sys, "version", ".".join(map(str, version)))


@pytest.mark.parametrize("version", [(3, 12, 12), (3, 12, 13)])
def test_python_only_accepts_supported_versions_without_dependencies(monkeypatch, tmp_path, version):
    monkeypatch.setattr(validate_environment, "__file__", str(tmp_path / "scripts/validate_environment.py"))
    assert not (tmp_path / ".python-version").exists()
    set_python_version(monkeypatch, version)
    monkeypatch.setattr(sys, "argv", ["validate_environment.py", "--python-only",
                                      "--requirements", "missing-requirements.txt"])

    def unexpected_dependency_check(*args):
        pytest.fail("python-only validation must not inspect or import dependencies")

    monkeypatch.setattr(validate_environment, "version", unexpected_dependency_check)
    monkeypatch.setattr(validate_environment.importlib, "import_module",
                        unexpected_dependency_check)
    validate_environment.main()


@pytest.mark.parametrize("version", [(3, 11, 12), (3, 12, 11), (3, 12, 14), (3, 13, 0)])
def test_wrong_python_version_reports_interpreter_and_recovery(monkeypatch, tmp_path, version):
    monkeypatch.setattr(validate_environment, "__file__", str(tmp_path / "scripts/validate_environment.py"))
    assert not (tmp_path / ".python-version").exists()
    set_python_version(monkeypatch, version)
    with pytest.raises(RuntimeError) as error:
        validate_environment.validate_python_version()
    message = str(error.value)
    assert f"Python 3.12.12 or 3.12.13 is required; found {sys.version}" in message
    assert sys.executable in message
    assert "uv venv --python 3.12.13 --seed venv-py31213" in message
    assert "export PYTHON_BIN=" in message
    assert "ENVIRONMENT.md" in message


def test_python_31213_still_rejects_incompatible_dependencies(monkeypatch):
    set_python_version(monkeypatch, (3, 12, 13))
    monkeypatch.setattr(sys, "argv", ["validate_environment.py", "--require-cuda"])
    monkeypatch.setattr(validate_environment, "pinned_requirements",
                        lambda path: iter([("torch", "2.13.0")]))
    monkeypatch.setattr(validate_environment, "version", lambda name: "0.0.0")
    with pytest.raises(RuntimeError, match="torch: expected 2.13.0, found 0.0.0"):
        validate_environment.main()
