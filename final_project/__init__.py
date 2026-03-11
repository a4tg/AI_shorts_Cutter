"""Root package shim for src-layout local execution.

This allows commands like `python -m final_project.gui` to work from the
repository root without requiring an editable install first.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC_PACKAGE = _ROOT / "src" / "final_project"

if not _SRC_PACKAGE.exists():
    raise ImportError(f"Expected source package at {_SRC_PACKAGE}")

__path__ = [str(_SRC_PACKAGE)]
