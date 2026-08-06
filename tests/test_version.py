from __future__ import annotations

import tomllib
from pathlib import Path

from bot import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_synchronized_with_metadata_and_readme() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert metadata["project"]["version"] == __version__
    assert f"Текущая версия: {__version__}" in readme
    assert f"Версия {__version__}" in readme
