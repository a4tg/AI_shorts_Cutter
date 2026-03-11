"""Decorator applying a moving blur effect to a video fragment.

This class is ported from CutterPy and allows an optional moving blur
effect where the blur centre can vary over time.  The blur is
implemented via MoviePy's ``HeadBlur`` effect.  The default blur
position is the centre of a vertical 1080×1920 frame.  A higher
``priority_index`` causes this decorator to be applied later in the
decorator chain.
"""

from typing import Optional, Callable, cast

from moviepy import VideoClip  # type: ignore
from moviepy.video.fx.HeadBlur import HeadBlur  # type: ignore

from ..core.decorator_interface import DecoratorInterface


class BlurOverlay(DecoratorInterface):
    def __init__(
        self,
        priority_index: int = 200,
        fx: Optional[Callable[[float], float]] = None,
        fy: Optional[Callable[[float], float]] = None,
        radius: float = 100.0,
        intensity: Optional[float] = None,
    ) -> None:
        super().__init__(priority_index)
        # Default static position functions (centre of a 1080×1920 frame)
        self._fx: Callable[[float], float] = fx if fx else lambda t: 540.0
        self._fy: Callable[[float], float] = fy if fy else lambda t: 960.0
        self._radius: float = radius
        self._intensity: Optional[float] = intensity

    def get_processed_fragment(self, edited_fragment: VideoClip) -> VideoClip:
        if self._radius <= 0:
            return edited_fragment
        blur_effect = HeadBlur(
            fx=self._fx, fy=self._fy, radius=self._radius, intensity=self._intensity
        )
        result = blur_effect.apply(edited_fragment)
        return cast(VideoClip, result)