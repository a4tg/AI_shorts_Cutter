"""Decorator for overlaying external audio onto a video fragment.

The sound overlay decorator allows you to add an additional audio
track on top of the original audio of a video clip.  You can
specify which portion of the audio file to use, where in the video
timeline it should begin and whether it should loop to match the
length of the clip.  If ``volume_scale`` is different from 1.0 the
original audio is scaled accordingly before mixing.
"""

from typing import Optional, Union
from pathlib import Path

from moviepy import VideoClip, AudioClip, AudioFileClip, CompositeAudioClip, afx  # type: ignore

from ..core.decorator_interface import DecoratorInterface


class SoundOverlay(DecoratorInterface):
    def __init__(
        self,
        priority_index: int = 100,
        audio_path: Optional[Union[str, Path]] = None,
        start_in_video: float = 0.0,
        end_in_video: float = 0.0,
        segment_audio_start: float = 0.0,
        segment_audio_end: float = 0.0,
        volume_scale: float = 1.0,
        loop_audio: bool = True,
    ) -> None:
        super().__init__(priority_index)
        self._audio_path = str(audio_path) if audio_path else ""
        self._start_in_video: float = start_in_video
        self._end_in_video: float = end_in_video
        self._segment_audio_start: float = segment_audio_start
        self._segment_audio_end: float = segment_audio_end
        self._volume_scale: float = volume_scale
        self._loop_audio: bool = loop_audio

    @staticmethod
    def _load_audio(file_path: str) -> Optional[AudioClip]:
        if not file_path:
            return None
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        try:
            return AudioFileClip(file_path)
        except Exception as e:
            raise RuntimeError(f"Audio loading error {file_path}: {e}")

    def _prepare_audio_clip(self, audio_clip: AudioClip, video_duration: float) -> AudioClip:
        # Crop the audio
        if self._segment_audio_end > 0:
            audio_clip = audio_clip.subclipped(self._segment_audio_start, self._segment_audio_end)
        elif self._segment_audio_start > 0:
            audio_clip = audio_clip.subclipped(self._segment_audio_start)
        target_duration = video_duration if self._end_in_video <= 0 else min(self._end_in_video, video_duration)
        # Loop or trim audio to match the target duration
        if audio_clip.duration < target_duration and self._loop_audio:
            audio_clip = audio_clip.with_effects([afx.AudioLoop(duration=target_duration)])
        elif audio_clip.duration > target_duration:
            audio_clip = audio_clip.subclipped(0, target_duration)
        return audio_clip

    def get_processed_fragment(self, edited_fragment: VideoClip) -> VideoClip:
        if not self._audio_path:
            return edited_fragment
        audio_clip = self._load_audio(self._audio_path)
        if audio_clip is None:
            return edited_fragment
        audio_clip = self._prepare_audio_clip(audio_clip, edited_fragment.duration)
        if self._start_in_video > 0:
            audio_clip = audio_clip.with_start(self._start_in_video)
        original_audio = edited_fragment.audio
        if original_audio and self._volume_scale != 1.0:
            original_audio = original_audio.with_volume_scaled(self._volume_scale)
        if original_audio:
            final_audio = CompositeAudioClip([original_audio, audio_clip])
        else:
            final_audio = audio_clip
        return edited_fragment.with_audio(final_audio)