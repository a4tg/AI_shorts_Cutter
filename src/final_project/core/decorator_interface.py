"""Base class for video decorators.

Decorators are applied to edited video fragments to add extra layers
such as blur overlays, stickers, soundtracks or subtitles.  Each
decorator specifies a processing priority via ``priority_index`` to
determine the order in which multiple decorators are applied.  Subclasses
must implement ``get_processed_fragment`` returning a new
``VideoClip``.
"""

from abc import ABC, abstractmethod
from moviepy import VideoClip  # type: ignore


class DecoratorInterface(ABC):
    """Abstract base class for all video decorators."""

    def __init__(self, priority_index: int = 0) -> None:
        self._priority_index = priority_index

    def get_priority_index(self) -> int:
        return self._priority_index

    @abstractmethod
    def get_processed_fragment(self, edited_fragment: VideoClip) -> VideoClip:
        """Return a new clip with the decoration applied."""
        raise NotImplementedError