"""Compatibility helpers for working around flaky MoviePy resource cleanup on Windows."""

from __future__ import annotations

from typing import Any, Callable


def _is_invalid_handle_error(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and getattr(exc, "winerror", None) == 6


def _close_process_handles(proc: Any) -> None:
    if proc is None:
        return
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(proc, stream_name, None)
        if stream is None:
            continue
        try:
            stream.close()
        except Exception:
            pass


def make_reader_close_safe(original_close: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap MoviePy reader close to suppress harmless WinError 6 cleanup failures."""

    def safe_close(self: Any, delete_lastread: bool = True) -> Any:
        try:
            return original_close(self, delete_lastread=delete_lastread)
        except Exception as exc:
            if not _is_invalid_handle_error(exc):
                raise
            _close_process_handles(getattr(self, "proc", None))
            if hasattr(self, "proc"):
                self.proc = None
            if delete_lastread and hasattr(self, "last_read"):
                self.last_read = None
            return None

    return safe_close


def apply_moviepy_compatibility_fixes() -> None:
    """Install idempotent MoviePy cleanup guards for Windows invalid-handle races."""

    try:
        from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader  # type: ignore
    except Exception:
        return

    if getattr(FFMPEG_VideoReader, "_final_project_safe_close", False):
        return

    original_close = FFMPEG_VideoReader.close
    FFMPEG_VideoReader.close = make_reader_close_safe(original_close)  # type: ignore[assignment]
    FFMPEG_VideoReader._final_project_safe_close = True  # type: ignore[attr-defined]


def safe_close_video_clip(clip: Any) -> None:
    """Close a clip and its common sub-resources without surfacing benign WinError 6."""

    if clip is None:
        return

    for attr_name in ("audio", "mask", "reader"):
        resource = getattr(clip, attr_name, None)
        if resource is None:
            continue
        try:
            close_method = getattr(resource, "close", None)
            if callable(close_method):
                close_method()
        except Exception as exc:
            if not _is_invalid_handle_error(exc):
                raise

    close_method = getattr(clip, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception as exc:
            if not _is_invalid_handle_error(exc):
                raise
