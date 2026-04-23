"""Runtime logging helpers for GUI/CLI sessions."""

from __future__ import annotations

import faulthandler
import logging
from datetime import datetime
from pathlib import Path


def configure_runtime_logging(log_name: str = "session") -> Path:
    """Attach a file logger and faulthandler dump for the current process."""

    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = logs_dir / f"{log_name}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    has_file_handler = any(
        isinstance(handler, logging.FileHandler) and Path(getattr(handler, "baseFilename", "")) == log_path
        for handler in root_logger.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(file_handler)

    if not any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(stream_handler)

    try:
        fault_file = open(log_path, "a", encoding="utf-8")
        faulthandler.enable(file=fault_file, all_threads=True)
    except Exception:
        pass

    logging.getLogger(__name__).info("Runtime log file: %s", log_path)
    return log_path
