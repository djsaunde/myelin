from __future__ import annotations

import py_compile
from pathlib import Path


def test_modal_launchers_compile_without_modal_installed() -> None:
    modal_dir = Path(__file__).resolve().parents[1] / "modal"

    for path in sorted(modal_dir.glob("*.py")):
        py_compile.compile(str(path), doraise=True)
