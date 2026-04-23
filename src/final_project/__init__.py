"""Top level package for the unified AI-driven video cutter."""

from __future__ import annotations

import os
from pathlib import Path

from .moviepy_compat import apply_moviepy_compatibility_fixes


def _load_local_env() -> None:
    """Load project-local .env values once without overriding explicit env vars."""
    env_candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    loaded_paths: set[Path] = set()
    for env_path in env_candidates:
        if env_path in loaded_paths or not env_path.exists():
            continue
        loaded_paths.add(env_path)
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip('"').strip("'")


_load_local_env()
apply_moviepy_compatibility_fixes()

__all__ = [
    "core",
    "segmentation",
    "decorators",
    "generator",
    "gpu",
    "models",
    "gui",
]
