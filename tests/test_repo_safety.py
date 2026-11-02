import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KEY_PATTERN = re.compile("sk-or-" + "v1-" + r"[A-Za-z0-9_\-]{20,}")
SCANNED_DIRS = ("src", "tests", "examples", "docs", ".github")
SCANNED_FILES = ("README.md", ".env.example", "pyproject.toml", ".gitignore")
SKIPPED_PARTS = {".venv", "__pycache__", ".git", ".ruff_cache", ".pytest_cache", ".mypy_cache"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".html", ".js", ".css", ".yml", ".yaml", ".toml", ".txt"}


def scanned_files() -> list[Path]:
    files = [ROOT / name for name in SCANNED_FILES if (ROOT / name).is_file()]
    for directory in SCANNED_DIRS:
        base = ROOT / directory
        if base.is_dir():
            files += [
                path
                for path in base.rglob("*")
                if path.is_file()
                and path.suffix in TEXT_SUFFIXES
                and not SKIPPED_PARTS & set(path.relative_to(ROOT).parts)
            ]
    return files


def test_env_file_is_gitignored() -> None:
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text("utf-8").splitlines()}

    assert ".env" in lines


def test_env_variants_are_ignored_but_the_example_is_not() -> None:
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text("utf-8").splitlines()}

    assert ".env.*" in lines
    assert "!.env.example" in lines


def test_env_example_has_an_empty_api_key() -> None:
    lines = (ROOT / ".env.example").read_text("utf-8").splitlines()

    key_lines = [line for line in lines if line.startswith("OPENROUTER_API_KEY")]
    assert key_lines == ["OPENROUTER_API_KEY="]


def test_env_example_has_no_other_credentials() -> None:
    text = (ROOT / ".env.example").read_text("utf-8")

    assert not KEY_PATTERN.search(text)
    assert not re.search(r"(?im)^[A-Z_]*(KEY|TOKEN|SECRET|PASSWORD)[A-Z_]*=\S", text)


def test_scan_covers_the_expected_files() -> None:
    names = {path.relative_to(ROOT).as_posix() for path in scanned_files()}

    assert {"README.md", ".env.example", "tests/test_repo_safety.py"} <= names
    assert any(name.startswith("src/product_card/") for name in names)
    assert not any(".venv" in name for name in names)


def test_key_pattern_detects_a_realistic_key_and_ignores_short_text() -> None:
    assert KEY_PATTERN.search("sk-or-" + "v1-" + "a" * 64)
    assert not KEY_PATTERN.search("sk-or-" + "v1-" + "short")


def test_no_file_contains_an_openrouter_key() -> None:
    leaked = [
        path.relative_to(ROOT).as_posix()
        for path in scanned_files()
        if KEY_PATTERN.search(path.read_text("utf-8", errors="ignore"))
    ]

    assert leaked == []


@pytest.mark.parametrize("name", [".env"])
def test_no_real_env_file_is_inside_scanned_source_directories(name: str) -> None:
    found = [
        path
        for directory in SCANNED_DIRS
        if (ROOT / directory).is_dir()
        for path in (ROOT / directory).rglob(name)
    ]

    assert found == []
