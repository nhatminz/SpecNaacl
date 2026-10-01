from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def _requirement_names(path):
    names = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "http://", "https://")):
            continue
        match = re.match(r"[A-Za-z0-9_.-]+", line)
        if match:
            names.add(match.group(0).lower().replace("_", "-"))
    return names


def test_requirements_are_exact_and_contain_no_stdlib_packages():
    requirement_path = ROOT / "requirements.txt"
    names = _requirement_names(requirement_path)
    stdlib_names = getattr(
        sys,
        "stdlib_module_names",
        {"asyncio", "datetime", "pathlib", "statistics", "typing", "json", "os"},
    )
    stdlib = {name.lower().replace("_", "-") for name in stdlib_names}
    assert not names.intersection(stdlib)
    for raw_line in requirement_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            assert "==" in line, f"direct dependency is not exactly pinned: {line}"


def test_specforge_is_installed_without_dependency_resolution():
    environment = (ROOT / "ENVIRONMENT.md").read_text(encoding="utf-8")
    assert "pip install --no-deps -e third_party/SpecForge" in environment
