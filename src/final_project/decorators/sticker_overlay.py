"""Decorator for superimposing an image or video sticker onto a clip.

The sticker decorator allows static images (PNG, JPG) or
animated images/videos (GIF, MP4) to be rendered on top of an edited
fragment.  Stickers can be resized, positioned and made semi
transparent or rotated.  When a sticker clip is shorter than the
target clip it will be looped automatically.
"""

from pathlib import Path
import math
from typing import Optional, Tuple, Union

from moviepy import VideoFileClip, VideoClip, ImageClip, CompositeVideoClip, vfx  # type: ignore

from ..core.decorator_interface import DecoratorInterface

StickerPositionType = Union[Tuple[int, int], Tuple[str, str], Tuple[str, int], Tuple[int, str], str]


class StickerOverlay(DecoratorInterface):
    def __init__(
        self,
        priority_index: int = 50,
        sticker_path: Optional[Union[str, Path]] = None,
        size: Optional[Tuple[int, int]] = None,
        position: str = "center",
        opacity: float = 1.0,
        rotation: float = 0.0,
    ) -> None:
        super().__init__(priority_index)
        self._sticker_path = str(sticker_path) if sticker_path else ""
        self._size = size
        self._position_str = position
        self._opacity = opacity
        self._rotation = rotation

    @staticmethod
    def _parse_position(
        position_str: str,
        video_size: Tuple[int, int],
        sticker_size: Tuple[int, int],
    ) -> StickerPositionType:
        margin = 48
        sticker_width, sticker_height = sticker_size
        positions = {
            "center": ("center", "center"),
            "top": ("center", 50),
            "bottom": ("center", video_size[1] - sticker_height - margin),
            "left": (50, "center"),
            "right": (video_size[0] - sticker_width - margin, "center"),
            "top_left": (50, 50),
            "top_right": (video_size[0] - sticker_width - margin, 50),
            "bottom_left": (50, video_size[1] - sticker_height - margin),
            "bottom_right": (video_size[0] - sticker_width - margin, video_size[1] - sticker_height - margin),
        }
        return positions.get(position_str, ("center", "center"))

    @staticmethod
    def _load_sticker(file_path: str) -> Optional[Union[VideoClip, ImageClip]]:
        if not file_path:
            return None
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Sticker file not found: {file_path}")
        ext = path.suffix.lower()
        try:
            if ext in [".gif", ".mp4", ".mov", ".avi"]:
                clip = VideoFileClip(file_path, has_mask=ext != ".gif", audio=False)
            elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                clip = ImageClip(file_path)
            else:
                raise ValueError(f"Unsupported sticker format: {ext}")
            return clip
        except Exception as e:
            raise RuntimeError(f"Sticker loading error {file_path}: {e}")

    def _resize_sticker(self, clip: Union[VideoClip, ImageClip]) -> Union[VideoClip, ImageClip]:
        return clip.resized(self._size) if self._size else clip

    def _fit_to_quadrant(
        self,
        clip: Union[VideoClip, ImageClip],
        video_size: Tuple[int, int],
    ) -> Union[VideoClip, ImageClip]:
        if self._size:
            return self._resize_sticker(clip)
        if self._position_str not in {"bottom_right", "top_right"}:
            return clip
        max_width = int(video_size[0] * 0.42)
        max_height = int(video_size[1] * 0.42)
        clip_width, clip_height = clip.size
        scale = min(max_width / clip_width, max_height / clip_height, 1.0)
        return clip.resized(scale)

    @staticmethod
    def _fit_sticker_duration(
        clip: Union[VideoClip, ImageClip], target_duration: float
    ) -> Union[VideoClip, ImageClip]:
        clip_duration = getattr(clip, "duration", 0) or 0
        if clip_duration <= 0:
            return clip.with_duration(target_duration) if hasattr(clip, "with_duration") else clip
        if clip_duration < target_duration and hasattr(clip, "with_effects"):
            loops = max(1, int(math.ceil(target_duration / clip_duration)))
            try:
                clip = clip.with_effects([vfx.Loop(n=loops)])
            except TypeError:
                clip = clip.with_effects([vfx.Loop(duration=clip_duration * loops)])
        if clip_duration > target_duration and hasattr(clip, "subclipped"):
            clip = clip.subclipped(0, max(0.0, target_duration - 1e-3))
        elif clip_duration <= target_duration and hasattr(clip, "subclipped"):
            clip = clip.subclipped(0, max(0.0, target_duration - 1e-3))
        return clip.with_duration(target_duration) if hasattr(clip, "with_duration") else clip

    def _apply_effects(self, clip: Union[VideoClip, ImageClip]) -> Union[VideoClip, ImageClip]:
        if self._opacity < 1.0 and hasattr(clip, "with_opacity"):
            clip = clip.with_opacity(self._opacity)
        if self._rotation != 0:
            clip = clip.rotated(self._rotation)
        return clip

    def get_processed_fragment(self, edited_fragment: VideoClip) -> VideoClip:
        if not self._sticker_path:
            return edited_fragment
        sticker_clip = self._load_sticker(self._sticker_path)
        if sticker_clip is None:
            return edited_fragment
        video_size = edited_fragment.size
        sticker_clip = self._fit_to_quadrant(sticker_clip, video_size)
        sticker_clip = self._apply_effects(sticker_clip)
        sticker_clip = self._fit_sticker_duration(sticker_clip, edited_fragment.duration)
        position = self._parse_position(self._position_str, video_size, sticker_clip.size)
        if hasattr(sticker_clip, "with_position"):
            sticker_clip = sticker_clip.with_position(position)
        # Compose the clips
        return CompositeVideoClip([edited_fragment, sticker_clip])
