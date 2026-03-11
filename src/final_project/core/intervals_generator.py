"""Time interval generators for video segmentation.

This module provides two generators:

* ``DefaultGenerator`` splits the input video into contiguous
  segments of variable length between ``MIN_CLIP_DURATION`` and
  ``MAX_CLIP_DURATION`` seconds.  It yields a sequence of time
  slots that gradually cover the video.
* ``PredeterminedGenerator`` yields a single user supplied time slot.

These classes are used by the high level ``ShortsGenerator`` to
determine which parts of a video are processed.  They encapsulate
generation logic separate from frame editing and decoration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Iterable

from moviepy import VideoClip  # type: ignore

from .constants import MIN_CLIP_DURATION, MAX_CLIP_DURATION


class IntervalGenerator(Enum):
    """Enumeration of available time slot generation strategies."""

    default_generator = "default_generator"
    predetermined_generator = "predetermined_generator"


@dataclass
class TimeSlotData:
    """Simple container describing the start and end of a time slot."""

    start: float
    end: float


@dataclass
class SegmentData:
    """Container for a video segment and its ordinal index."""

    video_segment: VideoClip
    segment_counter: int


class GeneratorInterface(ABC):
    """Abstract base class for all interval generators."""

    @staticmethod
    @abstractmethod
    def get_time(time_slot: TimeSlotData) -> Generator[TimeSlotData, None, None]:
        """Yield individual TimeSlotData objects from an input slot."""
        raise NotImplementedError

    @abstractmethod
    def get_fragment(self, video: VideoClip) -> Generator[SegmentData, None, None]:
        """Yield segments of ``video`` according to the generator's logic."""
        raise NotImplementedError


class DefaultGenerator(GeneratorInterface):
    """Split a video into sequential segments of manageable length.

    The generator walks through the video duration and yields
    ``TimeSlotData`` objects whose lengths are chosen to not exceed
    ``MAX_CLIP_DURATION``.  The first clip is skipped (starting at
    zero) to avoid extremely short segments at the beginning.  The
    algorithm gradually reduces the step size towards the end of the
    video to ensure the final segments are at least
    ``MIN_CLIP_DURATION`` long.
    """

    @staticmethod
    def get_time(time_slot: TimeSlotData) -> Generator[TimeSlotData, None, None]:
        duration = int(time_slot.end)
        current_time = 0
        if duration < MIN_CLIP_DURATION:
            raise RuntimeError("Video is too short to generate any clips")

        def get_step(current_step: int, delta: int) -> int:
            # choose a step no larger than the remaining duration and
            # bounded by MAX_CLIP_DURATION
            temporary_step = min(current_step, delta) - 1
            return (
                temporary_step
                if temporary_step <= MAX_CLIP_DURATION
                else MAX_CLIP_DURATION
            )

        step = get_step(MAX_CLIP_DURATION, duration)
        for segment in range(current_time, duration, step):
            if segment == 0:
                continue
            yield TimeSlotData(start=current_time, end=segment)
            current_time = segment
            step = get_step(step, duration - segment)
            if step <= MIN_CLIP_DURATION:
                break

    def get_fragment(self, video: VideoClip) -> Generator[SegmentData, None, None]:
        segment_counter = 0
        for segment in self.get_time(TimeSlotData(0, end=float(video.duration))):
            video_segment: VideoClip = video.subclipped(segment.start, segment.end)
            yield SegmentData(
                video_segment=video_segment, segment_counter=segment_counter
            )
            segment_counter += 1


class PredeterminedGenerator(GeneratorInterface):
    """Yield a predetermined time slot for a video fragment.

    This generator is useful when the time boundaries for the desired
    clip are already known – for example from a speech analysis step.
    It validates that the time slot satisfies the min/max duration
    constraints before yielding it.  If the slot does not meet the
    criteria a ``RuntimeError`` is raised.
    """

    def __init__(self, time_slot: TimeSlotData) -> None:
        self._time_slot = time_slot

    @staticmethod
    def get_time(time_slot: TimeSlotData) -> Generator[TimeSlotData, None, None]:
        if time_slot.start > time_slot.end:
            raise ValueError("'start' value must not exceed 'end'")
        duration = time_slot.end - time_slot.start
        if duration < MIN_CLIP_DURATION or duration > MAX_CLIP_DURATION:
            raise RuntimeError("Duration of fragment is out of the allowed range")
        yield time_slot

    def get_fragment(self, video: VideoClip) -> Generator[SegmentData, None, None]:
        segment_counter = 0
        for segment in self.get_time(self._time_slot):
            video_segment = video.subclipped(segment.start, segment.end)
            yield SegmentData(video_segment, segment_counter)
            segment_counter += 1