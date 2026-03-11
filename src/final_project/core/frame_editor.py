"""Frame editors used to crop and resize video segments."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

from moviepy import ColorClip, CompositeVideoClip, ImageClip, VideoClip  # type: ignore

from .constants import (
    EXPECTED_SHORT_SOURCE_SIZE,
    STANDARD_BACKGROUND_BLUR_KERNEL,
    STANDARD_FOREGROUND_SCALE,
    TARGET_SHORT_SIZE,
    TARGET_SHORT_SIZE_NEW,
)


class EditorInterface(ABC):
    @abstractmethod
    def get_short_video(self, fragment: VideoClip) -> VideoClip:
        raise NotImplementedError


class EditorStandard(EditorInterface):
    """Convert horizontal footage into a vertical short.

    The background is created from a blurred duplicate of the same clip,
    scaled to fill the 9:16 canvas. The original content stays horizontal
    in the foreground, is enlarged slightly, cropped a bit on the sides,
    and centered over the blurred background.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = TARGET_SHORT_SIZE,
        foreground_scale: float = STANDARD_FOREGROUND_SCALE,
        blur_kernel: int = STANDARD_BACKGROUND_BLUR_KERNEL,
    ) -> None:
        self.target_size = target_size
        self.foreground_scale = foreground_scale
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    def _resize_to_fill(self, clip: VideoClip, target_size: Tuple[int, int]) -> VideoClip:
        target_width, target_height = target_size
        current_width, current_height = clip.size
        fill_scale = max(target_width / current_width, target_height / current_height)
        return clip.resized(fill_scale)

    @staticmethod
    def _without_audio(clip: VideoClip) -> VideoClip:
        if hasattr(clip, "without_audio"):
            return clip.without_audio()
        if hasattr(clip, "with_audio"):
            return clip.with_audio(None)
        return clip

    def _crop_center(self, clip: VideoClip, target_size: Tuple[int, int]) -> VideoClip:
        target_width, target_height = target_size
        current_width, current_height = clip.size
        crop_width = min(target_width, current_width)
        crop_height = min(target_height, current_height)
        x1 = max(0, int((current_width - crop_width) / 2))
        y1 = max(0, int((current_height - crop_height) / 2))
        return clip.cropped(
            x1=x1,
            y1=y1,
            x2=x1 + int(crop_width),
            y2=y1 + int(crop_height),
        )

    def _blur_background(self, clip: VideoClip) -> VideoClip:
        try:
            import cv2  # type: ignore
        except ImportError:
            return clip
        if not hasattr(clip, "image_transform"):
            return clip

        def blur_frame(frame):
            return cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)

        return clip.image_transform(blur_frame)

    def _build_background(self, fragment: VideoClip) -> VideoClip:
        background = self._resize_to_fill(fragment, self.target_size)
        background = self._crop_center(background, self.target_size)
        background = self._blur_background(background)
        background = self._without_audio(background)
        return background.with_position(("center", "center"))

    def _build_foreground(self, fragment: VideoClip) -> VideoClip:
        target_width, _ = self.target_size
        scale_to_width = target_width / fragment.size[0]
        foreground = fragment.resized(scale_to_width * self.foreground_scale)
        if foreground.size[0] > target_width:
            foreground = self._crop_center(foreground, (target_width, foreground.size[1]))
        return foreground.with_position(("center", "center"))

    def get_short_video(self, fragment: VideoClip) -> VideoClip:
        if (
            fragment.size != EXPECTED_SHORT_SOURCE_SIZE
            and fragment.size != TARGET_SHORT_SIZE
        ):
            fragment = fragment.resized(EXPECTED_SHORT_SOURCE_SIZE)

        background = self._build_background(fragment)
        foreground = self._build_foreground(fragment)
        composite = CompositeVideoClip([background, foreground], size=self.target_size)
        if hasattr(composite, "with_audio"):
            composite = composite.with_audio(getattr(foreground, "audio", None))
        return composite


class EditorPoint(EditorInterface):
    def __init__(self, coordinates: List[Dict[str, int]]) -> None:
        if len(coordinates) != 2:
            raise ValueError("EditorPoint expects a list of two coordinate sets")
        self.coordinates = coordinates

    def get_short_video(self, fragment: VideoClip) -> VideoClip:
        first_section = self.coordinates[0]
        second_section = self.coordinates[1]

        first_subclip = fragment.cropped(
            x1=int(first_section["x1"]),
            y1=int(first_section["y1"]),
            x2=int(first_section["x2"]),
            y2=int(first_section["y2"]),
        )

        second_subclip = fragment.cropped(
            x1=int(second_section["x1"]),
            y1=int(second_section["y1"]),
            x2=int(second_section["x2"]),
            y2=int(second_section["y2"]),
        )

        first_resized = first_subclip.resized(TARGET_SHORT_SIZE_NEW)
        second_resized = second_subclip.resized(TARGET_SHORT_SIZE_NEW)

        composite = CompositeVideoClip(
            [
                self._without_audio(first_resized).with_position(("center", "top")),
                self._without_audio(second_resized).with_position(("center", "bottom")),
            ],
            size=TARGET_SHORT_SIZE,
        )
        if hasattr(composite, "with_audio"):
            composite = composite.with_audio(getattr(fragment, "audio", None))
        return composite


class FrameEditor(EditorInterface):
    def __init__(
        self,
        coordinates: Dict[str, int],
        target_size: Tuple[int, int] = TARGET_SHORT_SIZE,
    ) -> None:
        required = {"x1", "y1", "x2", "y2"}
        if not required.issubset(coordinates.keys()):
            raise ValueError(f"Coordinates must contain keys {required}")
        self.coordinates = coordinates
        self.target_size = target_size

    def get_short_video(self, fragment: Union[VideoClip, ImageClip]) -> VideoClip:
        x1 = int(self.coordinates["x1"])
        y1 = int(self.coordinates["y1"])
        x2 = int(self.coordinates["x2"])
        y2 = int(self.coordinates["y2"])
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid coordinates: x1={x1}, x2={x2}, y1={y1}, y2={y2}. x2 must be > x1 and y2 > y1"
            )

        cropped_fragment = fragment.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
        resized_fragment = self._resize_with_padding(cropped_fragment, self.target_size)
        return resized_fragment.with_position(("center", "center"))

    @staticmethod
    def _without_audio(clip: Union[VideoClip, ImageClip]) -> VideoClip:
        if hasattr(clip, "without_audio"):
            return clip.without_audio()
        if hasattr(clip, "with_audio"):
            return clip.with_audio(None)
        return clip

    def _resize_with_padding(
        self, clip: Union[VideoClip, ImageClip], target_size: Tuple[int, int]
    ) -> VideoClip:
        target_width, target_height = target_size
        current_width, current_height = clip.size

        width_ratio = target_width / current_width
        height_ratio = target_height / current_height
        scale_factor = min(width_ratio, height_ratio)

        new_width = int(current_width * scale_factor)
        new_height = int(current_height * scale_factor)

        resized_clip = clip.resized((new_width, new_height))

        if new_width != target_width or new_height != target_height:
            background = ColorClip(
                size=target_size, color=(0, 0, 0), duration=clip.duration
            )
            x_pos = (target_width - new_width) // 2
            y_pos = (target_height - new_height) // 2
            composite = CompositeVideoClip(
                [self._without_audio(background), resized_clip.with_position((x_pos, y_pos))],
                size=target_size,
            )
            if hasattr(composite, "with_audio"):
                composite = composite.with_audio(getattr(resized_clip, "audio", None))
            return composite

        return resized_clip
