"""Shared request and styling models for the video pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .core.constants import MAX_CLIP_DURATION, MIN_CLIP_DURATION


@dataclass(frozen=True)
class SubtitleStyle:
    enabled: bool = True
    font: str = ""
    fontsize: int = 42
    min_fontsize: int = 24
    auto_fit: bool = True
    color: str = "white"
    highlight_color: str = "yellow"
    stroke_color: str = "black"
    stroke_width: float = 2.0
    position: Tuple[str, str] | str = ("center", "below_foreground")
    background_enabled: bool = True
    background_color: Tuple[int, int, int] = (0, 0, 0)
    background_opacity: float = 0.55
    background_padding: Tuple[int, int] = (40, 24)
    max_width_ratio: float = 0.82
    max_height_ratio: float = 0.22


@dataclass(frozen=True)
class ProcessingRequest:
    input_path: str
    output_dir: str
    mode: str = "speech"
    clip_count: int = 10
    min_clip_duration: float = MIN_CLIP_DURATION
    max_clip_duration: float = MAX_CLIP_DURATION
    coords: Optional[Tuple[int, int, int, int]] = None
    blur_radius: float = 0.0
    sticker_path: Optional[str] = None
    sticker_paths: Tuple[str, ...] = ()
    sticker_clips_count: Optional[int] = None
    sticker_size: Optional[Tuple[int, int]] = None
    sticker_position: str = "bottom"
    sound_path: Optional[str] = None
    subtitle_style: SubtitleStyle = SubtitleStyle()
