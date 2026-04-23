"""High level generator orchestrating the unified video cutting pipeline."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import random
import re
import tempfile
import threading
import time
from typing import Callable, List, Optional, Tuple

import ffmpeg
from moviepy import VideoFileClip  # type: ignore

try:
    from proglog import ProgressBarLogger
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight test envs
    class ProgressBarLogger:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.bars = {}

from .core.constants import MAX_CLIP_DURATION, MIN_CLIP_DURATION, TARGET_SHORT_SIZE
from .core.decorator_interface import DecoratorInterface
from .core.frame_editor import EditorInterface
from .decorators.subtitle_overlay import SubtitlesOverlay
from .decorators.sticker_overlay import StickerOverlay
from .gpu import ffmpeg_nvenc_available
from .models import ProcessingRequest, SubtitleStyle
from .moviepy_compat import safe_close_video_clip
from .segmentation import (
    adjust_clip_boundaries,
    adjust_segment_boundaries,
    build_dynamic_subtitles,
    detect_music_beats,
    detect_speech_pauses,
    generate_beats_segments,
    group_segments,
    iterative_candidate_selection,
    transcribe_audio,
    transcribe_precise_clip,
    ENABLE_PRECISE_SUBTITLE_REFINEMENT,
    DEFAULT_PRECISE_SUBTITLE_MAX_CLIPS,
)

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[float, str], None]
VERY_LONG_VIDEO_SECONDS = float(os.environ.get("FINAL_PROJECT_VERY_LONG_VIDEO_SECONDS", str(3 * 3600)))
LONG_VIDEO_SECONDS = float(os.environ.get("FINAL_PROJECT_LONG_VIDEO_SECONDS", str(60 * 60)))


@dataclass(frozen=True)
class RenderJob:
    candidate_index: int
    candidate: "Candidate"
    output_path: str
    subtitles: List[dict[str, object]]
    banner_path: str | None = None


@dataclass(frozen=True)
class BannerRenderJob:
    candidate_index: int
    candidate: "Candidate"
    source_path: str
    output_path: str
    subtitles: List[dict[str, object]]
    banner_path: str


class GuiProgressLogger(ProgressBarLogger):
    """Translate MoviePy export progress into GUI progress updates."""

    def __init__(
        self,
        callback: ProgressCallback,
        clip_label: str,
        stage_start: float,
        stage_span: float,
    ) -> None:
        super().__init__(logged_bars=False, ignored_bars=None)
        self._callback = callback
        self._clip_label = clip_label
        self._stage_start = stage_start
        self._stage_span = stage_span

    def _current_ratio(self) -> float:
        audio_ratio = 0.0
        video_ratio = 0.0

        audio_bar = self.bars.get("chunk")
        if audio_bar and audio_bar.get("total"):
            audio_ratio = min(1.0, float(audio_bar["index"]) / float(audio_bar["total"]))

        video_bar = self.bars.get("t")
        if video_bar and video_bar.get("total"):
            video_ratio = min(1.0, float(video_bar["index"]) / float(video_bar["total"]))

        if video_bar and video_bar.get("total"):
            return (audio_ratio * 0.2) + (video_ratio * 0.8)
        if audio_bar and audio_bar.get("total"):
            return audio_ratio
        return 0.0

    def bars_callback(self, bar, attr, value, old_value=None):
        del old_value
        if attr != "index":
            return
        ratio = self._current_ratio()
        status = f"Encoding {self._clip_label}"
        if bar == "chunk":
            status = f"Encoding audio for {self._clip_label}"
        elif bar == "t":
            status = f"Encoding video for {self._clip_label}"
        self._callback(self._stage_start + (self._stage_span * ratio), status)

    def callback(self, **changes):
        message = str(changes.get("message", "")).strip()
        if not message:
            return
        lowered = message.lower()
        if "writing audio" in lowered:
            self._callback(self._stage_start, f"Encoding audio for {self._clip_label}")
        elif "writing video" in lowered:
            self._callback(self._stage_start + (self._stage_span * 0.2), f"Encoding video for {self._clip_label}")


def extract_audio(video_path: str, audio_path: str = "temp_audio.wav") -> Optional[str]:
    """Extract a mono WAV file from a video using ffmpeg."""
    try:
        ffmpeg.input(video_path).output(
            audio_path, acodec="pcm_s16le", ac=1, ar=16000
        ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        return audio_path
    except ffmpeg.Error as exc:
        logger.error("Audio extraction error: %s", exc.stderr.decode())
        return None


def extract_audio_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    audio_path: str,
) -> Optional[str]:
    try:
        duration = max(0.01, end_time - start_time)
        ffmpeg.input(video_path, ss=max(0.0, start_time), t=duration).output(
            audio_path,
            acodec="pcm_s16le",
            ac=1,
            ar=16000,
        ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        return audio_path
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode() if getattr(exc, "stderr", None) else str(exc)
        logger.error("Segment audio extraction error: %s", stderr)
        return None


class Candidate:
    def __init__(
        self,
        start: float,
        end: float,
        text: str,
        subtitles: Optional[List[dict[str, object]]] = None,
    ) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.subtitles = subtitles or []


class ParallelProgressTracker:
    """Aggregate progress from multiple clip render workers into one GUI stream."""

    def __init__(
        self,
        callback: ProgressCallback | None,
        total_jobs: int,
        stage_start: float,
        stage_span: float,
    ) -> None:
        self._callback = callback
        self._stage_start = stage_start
        self._stage_span = stage_span
        self._lock = threading.Lock()
        self._ratios = [0.0] * max(1, total_jobs)

    def update(self, job_index: int, ratio: float, message: str) -> None:
        if not self._callback:
            return
        bounded_ratio = max(0.0, min(1.0, float(ratio)))
        with self._lock:
            self._ratios[job_index] = bounded_ratio
            average_ratio = sum(self._ratios) / len(self._ratios)
        progress = self._stage_start + (self._stage_span * average_ratio)
        self._callback(progress, message)


class ShortsGenerator:
    def __init__(
        self,
        editor: EditorInterface,
        decorators: Optional[List[DecoratorInterface]] = None,
        target_size: tuple[int, int] = TARGET_SHORT_SIZE,
        subtitle_style: SubtitleStyle | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._editor = editor
        self._decorators = decorators or []
        self._target_size = target_size
        self._subtitle_style = subtitle_style or SubtitleStyle()
        self._progress_callback = progress_callback

    @staticmethod
    def _build_export_settings() -> Tuple[str, List[str]]:
        if ffmpeg_nvenc_available():
            preset = os.environ.get("FINAL_PROJECT_NVENC_PRESET", "p6")
            cq = os.environ.get("FINAL_PROJECT_NVENC_CQ", "19")
            rc = os.environ.get("FINAL_PROJECT_NVENC_RC", "vbr")
            return (
                "h264_nvenc",
                [
                    "-preset",
                    preset,
                    "-rc",
                    rc,
                    "-cq",
                    cq,
                    "-b:v",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                ],
            )
        return ("libx264", ["-crf", os.environ.get("FINAL_PROJECT_X264_CRF", "18"), "-pix_fmt", "yuv420p"])

    def _report_progress(self, value: float, message: str) -> None:
        if not self._progress_callback:
            return
        bounded_value = max(0.0, min(100.0, float(value)))
        self._progress_callback(bounded_value, message)

    @staticmethod
    def _reuse_base_renders_for_banners() -> bool:
        return os.environ.get("FINAL_PROJECT_REUSE_BASE_FOR_BANNERS", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _make_temp_audio_path(output_dir: str) -> str:
        fd, path = tempfile.mkstemp(
            prefix="final_project_audio_",
            suffix=".wav",
            dir=output_dir,
        )
        os.close(fd)
        return path

    @staticmethod
    def _resolve_precise_subtitle_limit(
        clip_count: int,
        video_duration: float,
    ) -> int:
        configured_limit = max(0, int(DEFAULT_PRECISE_SUBTITLE_MAX_CLIPS))
        if configured_limit == 0:
            return 0
        if video_duration >= VERY_LONG_VIDEO_SECONDS or clip_count >= 40:
            return 0
        if video_duration >= LONG_VIDEO_SECONDS or clip_count >= 20:
            return min(configured_limit, 4)
        if clip_count >= 12:
            return min(configured_limit, 8)
        return configured_limit

    @staticmethod
    def _resolve_parallel_export_plan(
        clip_count: int,
        cpu_count: int | None = None,
    ) -> tuple[int, int]:
        safe_clip_count = max(1, int(clip_count))
        total_cpus = max(1, int(cpu_count or os.cpu_count() or 1))
        nvenc_available = ffmpeg_nvenc_available()
        configured_workers = os.environ.get("FINAL_PROJECT_PARALLEL_EXPORTS", "").strip()
        configured_threads = os.environ.get("FINAL_PROJECT_EXPORT_THREADS", "").strip()
        if configured_workers:
            try:
                requested_workers = max(1, int(configured_workers))
            except ValueError:
                requested_workers = 1
        else:
            requested_workers = 2 if nvenc_available else 1
        workers = max(1, min(safe_clip_count, requested_workers, total_cpus))
        if configured_threads:
            try:
                threads_per_worker = max(1, int(configured_threads))
            except ValueError:
                threads_per_worker = 1
        else:
            if nvenc_available:
                threads_per_worker = 1
            elif workers == 1:
                threads_per_worker = max(1, min(total_cpus, total_cpus - 1 or 1))
            else:
                threads_per_worker = max(1, total_cpus // workers)
        return workers, threads_per_worker

    async def _render_jobs_with_worker_count(
        self,
        request: ProcessingRequest,
        jobs: List[RenderJob],
        worker_count: int,
        export_threads: int,
        stage_start: float = 65.0,
        stage_span: float = 33.0,
    ) -> List[str]:
        progress_tracker = ParallelProgressTracker(
            callback=self._progress_callback,
            total_jobs=len(jobs),
            stage_start=stage_start,
            stage_span=stage_span,
        )
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="clip-render") as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    self._render_job,
                    request,
                    job,
                    len(jobs),
                    export_threads,
                    progress_tracker,
                )
                for job in jobs
            ]
            rendered_paths = await asyncio.gather(*tasks)
        return list(rendered_paths)

    @staticmethod
    def _deduplicate_candidates(
        candidates: List[Candidate],
        pauses: List[tuple[float, float]],
        video_duration: float,
        mode: str,
        min_duration: float,
        max_duration: float,
        max_clips: int,
    ) -> List[Candidate]:
        del pauses, video_duration, mode, min_duration, max_duration
        unique: List[Candidate] = []
        seen_ranges: set[tuple[int, int]] = set()
        for candidate in sorted(candidates, key=lambda item: (item.start, item.end)):
            start, end = candidate.start, candidate.end
            key = (round(start * 10), round(end * 10))
            if key in seen_ranges:
                continue
            overlaps_existing = any(
                not (end <= existing.start or start >= existing.end)
                for existing in unique
            )
            if overlaps_existing:
                continue
            seen_ranges.add(key)
            unique.append(
                Candidate(
                    start=start,
                    end=end,
                    text=candidate.text,
                    subtitles=candidate.subtitles,
                )
            )
            if len(unique) >= max_clips:
                break
        return unique

    @staticmethod
    def _collect_overlapping_segments(
        raw_segments: List[dict[str, float]],
        start: float,
        end: float,
    ) -> List[dict[str, float]]:
        overlapping: List[dict[str, float]] = []
        for segment in raw_segments:
            seg_start = float(segment["start"])
            seg_end = float(segment["end"])
            if seg_end <= start or seg_start >= end:
                continue
            overlapping.append(segment)
        return overlapping

    @classmethod
    def _speech_coverage_ratio(
        cls,
        raw_segments: List[dict[str, float]],
        start: float,
        end: float,
    ) -> float:
        duration = max(0.01, end - start)
        covered = 0.0
        for segment in cls._collect_overlapping_segments(raw_segments, start, end):
            covered += max(0.0, min(end, float(segment["end"])) - max(start, float(segment["start"])))
        return covered / duration

    async def _get_speech_candidates(
        self,
        audio_path: str,
        video_duration: float,
        min_clip_duration: float,
        max_clip_duration: float,
        clip_count: int,
        pauses: List[tuple[float, float]] | None = None,
    ) -> List[Candidate]:
        self._report_progress(15.0, "Transcribing speech")
        transcript, raw_segments = await transcribe_audio(audio_path)
        logger.info("Transcription returned %s raw segments", len(raw_segments))
        if not raw_segments:
            raise RuntimeError("Transcription produced no segments")
        if pauses is None:
            self._report_progress(28.0, "Detecting pauses")
            pauses = detect_speech_pauses(audio_path)
        logger.info("Pause detection returned %s intervals", len(pauses))
        self._report_progress(36.0, "Grouping speech segments")
        grouped_segments = group_segments(raw_segments, transcript)
        logger.info("Grouped ASR into %s candidate segments", len(grouped_segments))
        adjusted_segments = adjust_segment_boundaries(
            grouped_segments,
            pauses,
            min_duration=min_clip_duration,
            max_duration=max_clip_duration,
        )
        logger.info("Boundary adjustment produced %s segments", len(adjusted_segments))
        candidate_pool_size = max(clip_count * 5, clip_count + 5)
        candidates_dicts = iterative_candidate_selection(
            adjusted_segments,
            video_duration,
            max_clips=candidate_pool_size,
        )
        logger.info("Interestingness selector picked %s candidates before dedup", len(candidates_dicts))
        self._report_progress(44.0, "Building subtitles")
        candidates: List[Candidate] = []
        for item in candidates_dicts:
            start, end = adjust_clip_boundaries(
                float(item["start"]),
                float(item["end"]),
                pauses,
                video_duration,
                min_duration=min_clip_duration,
                max_duration=max_clip_duration,
            )
            overlapping_segments = self._collect_overlapping_segments(raw_segments, start, end)
            coverage_ratio = self._speech_coverage_ratio(overlapping_segments, start, end)
            subtitles = build_dynamic_subtitles(overlapping_segments, start, end, pauses=pauses)
            if not subtitles or coverage_ratio < 0.18:
                logger.info(
                    "Rejected candidate %.2f-%.2f due to low speech coverage: ratio=%.2f, subtitles=%s",
                    start,
                    end,
                    coverage_ratio,
                    len(subtitles),
                )
                continue
            candidates.append(
                Candidate(
                    start=start,
                    end=end,
                    text=str(item["text"]),
                    subtitles=subtitles,
                )
            )
            logger.info(
                "Accepted candidate %.2f-%.2f: coverage=%.2f, subtitles=%s",
                start,
                end,
                coverage_ratio,
                len(subtitles),
            )
        logger.info("Built %s speech candidates with dynamic subtitles", len(candidates))
        return candidates

    def _get_music_candidates(
        self,
        audio_path: str,
        video_duration: float,
        max_clips: int,
        min_clip_duration: float,
        max_clip_duration: float,
    ) -> List[Candidate]:
        self._report_progress(18.0, "Detecting beats")
        beat_times = detect_music_beats(audio_path)
        if not beat_times:
            raise RuntimeError("Unable to detect beats in the audio")
        self._report_progress(34.0, "Building beat segments")
        segments = generate_beats_segments(
            beat_times, video_duration, min_clip_duration, max_clip_duration
        )
        candidates = [
            Candidate(start=item["start"], end=item["end"], text=item["text"])
            for item in segments
        ]
        return candidates[:max_clips]

    def _write_clip(
        self,
        clip,
        output_path: str,
        fps: int = 24,
        threads: int | None = None,
        progress_callback: ProgressCallback | None = None,
        stage_start: float = 55.0,
        stage_span: float = 40.0,
        clip_label: str = "clip",
    ) -> None:
        """Write a clip to disk, preferring NVENC when supported by FFmpeg."""
        codec, ffmpeg_params = self._build_export_settings()
        export_logger: GuiProgressLogger | str | None = "bar"
        effective_progress_callback = progress_callback or self._progress_callback
        if effective_progress_callback:
            export_logger = GuiProgressLogger(
                callback=effective_progress_callback,
                clip_label=clip_label,
                stage_start=stage_start,
                stage_span=stage_span,
            )
        temp_audiofile = str(Path(output_path).with_name(f"{Path(output_path).stem}_audio_{os.getpid()}_{threading.get_ident()}.mp3"))
        clip.write_videofile(
            output_path,
            codec=codec,
            fps=fps,
            threads=max(1, int(threads or os.cpu_count() or 1)),
            ffmpeg_params=ffmpeg_params,
            temp_audiofile=temp_audiofile,
            remove_temp=True,
            logger=export_logger,
        )

    @staticmethod
    def _resolve_banner_paths(request: ProcessingRequest) -> List[str]:
        configured = [path.strip() for path in request.sticker_paths if path and path.strip()]
        if configured:
            return configured
        if request.sticker_path and request.sticker_path.strip():
            return [request.sticker_path.strip()]
        return []

    @staticmethod
    def _sanitize_banner_folder_name(banner_path: str) -> str:
        invalid_chars_pattern = r'[<>:"/\\|?*]+'
        stem = Path(banner_path).stem.strip() or "banner"
        sanitized = re.sub(invalid_chars_pattern, "_", stem).rstrip(" .")
        return sanitized or "banner"

    @classmethod
    def _resolve_banner_folder_map(cls, banner_paths: List[str]) -> dict[str, str]:
        used_names: dict[str, int] = {}
        resolved: dict[str, str] = {}
        for banner_path in banner_paths:
            base_name = cls._sanitize_banner_folder_name(banner_path)
            occurrence = used_names.get(base_name, 0) + 1
            used_names[base_name] = occurrence
            folder_suffix = base_name if occurrence == 1 else f"{base_name}_{occurrence}"
            resolved[banner_path] = f"with_banner_{folder_suffix}"
        return resolved

    @staticmethod
    def _build_banner_candidate_indices(
        sticker_clips_count: int | None,
        total_candidates: int,
    ) -> set[int]:
        if total_candidates <= 0:
            return set()
        if sticker_clips_count is None:
            clips_with_banner = total_candidates
        else:
            clips_with_banner = max(0, min(total_candidates, int(sticker_clips_count)))
        if clips_with_banner == 0:
            return set()
        selected_indices = random.sample(range(total_candidates), k=clips_with_banner)
        return set(selected_indices)

    @staticmethod
    def _resolve_clip_output_path(
        request: ProcessingRequest,
        candidate_index: int,
        banner_folder: str | None,
    ) -> str:
        subdir_name = banner_folder or "without_banner"
        target_dir = Path(request.output_dir) / subdir_name
        target_dir.mkdir(parents=True, exist_ok=True)
        return str(target_dir / f"clip_{candidate_index}.mp4")

    def _render_job(
        self,
        request: ProcessingRequest,
        job: RenderJob,
        total_candidates: int,
        export_threads: int,
        progress_tracker: ParallelProgressTracker | None,
    ) -> str:
        clip_label = f"clip {job.candidate_index + 1}/{total_candidates}"
        tracker = progress_tracker
        if tracker:
            tracker.update(job.candidate_index, 0.02, f"Opening {clip_label}")

        with VideoFileClip(request.input_path) as video:
            source_fragment = video.subclipped(job.candidate.start, job.candidate.end)
            if tracker:
                tracker.update(job.candidate_index, 0.12, f"Editing {clip_label}")
            edited_clip = self._editor.get_short_video(source_fragment)
            if tracker:
                tracker.update(job.candidate_index, 0.24, f"Compositing {clip_label}")

            subtitle_decorator = SubtitlesOverlay(
                subtitles=job.subtitles
                or [
                    {
                        "start": 0.0,
                        "end": edited_clip.duration,
                        "text": job.candidate.text,
                    }
                ],
                style=self._subtitle_style,
                priority_index=150,
            )

            processed_clip = edited_clip
            decorators = self._decorators.copy()
            if job.banner_path:
                decorators.append(
                    StickerOverlay(
                        sticker_path=job.banner_path,
                        size=request.sticker_size,
                        position=request.sticker_position,
                        opacity=1.0,
                    )
                )
            decorators.append(subtitle_decorator)
            try:
                for decorator in sorted(decorators, key=lambda item: item.get_priority_index()):
                    try:
                        processed_clip = decorator.get_processed_fragment(processed_clip)
                    except Exception as exc:
                        logger.warning(
                            "Decorator %s failed on clip %s: %s",
                            decorator.__class__.__name__,
                            job.candidate_index,
                            exc,
                        )

                if tracker:
                    tracker.update(job.candidate_index, 0.3, f"Encoding {clip_label}")
                self._write_clip(
                    processed_clip,
                    job.output_path,
                    threads=export_threads,
                    progress_callback=(
                        (lambda value, message: tracker.update(job.candidate_index, value, message))
                        if tracker
                        else None
                    ),
                    stage_start=0.3,
                    stage_span=0.68,
                    clip_label=clip_label,
                )
                if tracker:
                    tracker.update(job.candidate_index, 1.0, f"Saved {clip_label}")
                return job.output_path
            finally:
                safe_close_video_clip(processed_clip)
                safe_close_video_clip(edited_clip)
                safe_close_video_clip(source_fragment)

    def _render_banner_job(
        self,
        request: ProcessingRequest,
        job: BannerRenderJob,
        total_jobs: int,
        export_threads: int,
        job_index: int,
        progress_tracker: ParallelProgressTracker | None,
    ) -> str:
        clip_label = f"banner {job_index + 1}/{total_jobs}"
        tracker = progress_tracker
        if tracker:
            tracker.update(job_index, 0.05, f"Opening {clip_label}")
        try:
            with VideoFileClip(job.source_path) as base_clip:
                if tracker:
                    tracker.update(job_index, 0.18, f"Compositing {clip_label}")
                sticker_decorator = StickerOverlay(
                    sticker_path=job.banner_path,
                    size=request.sticker_size,
                    position=request.sticker_position,
                    opacity=1.0,
                )
                processed_clip = base_clip
                try:
                    processed_clip = sticker_decorator.get_processed_fragment(base_clip)
                    if tracker:
                        tracker.update(job_index, 0.3, f"Encoding {clip_label}")
                    self._write_clip(
                        processed_clip,
                        job.output_path,
                        threads=export_threads,
                        progress_callback=(
                            (lambda value, message: tracker.update(job_index, value, message))
                            if tracker
                            else None
                        ),
                        stage_start=0.3,
                        stage_span=0.68,
                        clip_label=clip_label,
                    )
                    if tracker:
                        tracker.update(job_index, 1.0, f"Saved {clip_label}")
                    return job.output_path
                finally:
                    if processed_clip is not base_clip:
                        safe_close_video_clip(processed_clip)
        except Exception as exc:
            logger.warning(
                "Fast banner render failed for clip %s and banner %s. Retrying full render: %s",
                job.candidate_index,
                job.banner_path,
                exc,
            )
            fallback_job = RenderJob(
                candidate_index=job.candidate_index,
                candidate=job.candidate,
                output_path=job.output_path,
                subtitles=job.subtitles,
                banner_path=job.banner_path,
            )
            return self._render_job(
                request=request,
                job=fallback_job,
                total_candidates=max(1, total_jobs),
                export_threads=export_threads,
                progress_tracker=None,
            )

    async def _render_banner_jobs_parallel(
        self,
        request: ProcessingRequest,
        jobs: List[BannerRenderJob],
    ) -> List[str]:
        if not jobs:
            return []
        worker_count, export_threads = self._resolve_parallel_export_plan(len(jobs))
        logger.info(
            "Fast banner export plan: banners=%s workers=%s ffmpeg_threads_per_worker=%s",
            len(jobs),
            worker_count,
            export_threads,
        )
        progress_tracker = ParallelProgressTracker(
            callback=self._progress_callback,
            total_jobs=len(jobs),
            stage_start=88.0,
            stage_span=10.0,
        )
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="banner-render") as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    self._render_banner_job,
                    request,
                    job,
                    len(jobs),
                    export_threads,
                    job_index,
                    progress_tracker,
                )
                for job_index, job in enumerate(jobs)
            ]
            rendered_paths = await asyncio.gather(*tasks)
        return list(rendered_paths)

    async def _render_jobs_parallel(
        self,
        request: ProcessingRequest,
        jobs: List[RenderJob],
        stage_start: float = 65.0,
        stage_span: float = 33.0,
    ) -> List[str]:
        if not jobs:
            return []
        worker_count, export_threads = self._resolve_parallel_export_plan(len(jobs))
        logger.info(
            "Parallel export plan: clips=%s workers=%s ffmpeg_threads_per_worker=%s",
            len(jobs),
            worker_count,
            export_threads,
        )
        try:
            if stage_start == 65.0 and stage_span == 33.0:
                return await self._render_jobs_with_worker_count(
                    request,
                    jobs,
                    worker_count,
                    export_threads,
                )
            return await self._render_jobs_with_worker_count(
                request,
                jobs,
                worker_count,
                export_threads,
                stage_start,
                stage_span,
            )
        except Exception as exc:
            if worker_count <= 1:
                raise
            logger.warning(
                "Parallel clip export failed with workers=%s threads=%s. Retrying sequentially: %s",
                worker_count,
                export_threads,
                exc,
            )
            self._report_progress(65.0, "Parallel export failed, retrying sequentially")
            if stage_start == 65.0 and stage_span == 33.0:
                return await self._render_jobs_with_worker_count(
                    request,
                    jobs,
                    1,
                    1,
                )
            return await self._render_jobs_with_worker_count(
                request,
                jobs,
                1,
                1,
                stage_start,
                stage_span,
            )

    async def _refine_candidate_subtitles(
        self,
        request: ProcessingRequest,
        candidate: Candidate,
        candidate_index: int,
        precise_subtitle_limit: int,
    ) -> List[dict[str, object]]:
        if request.mode != "speech" or not self._subtitle_style.enabled or not ENABLE_PRECISE_SUBTITLE_REFINEMENT:
            return candidate.subtitles
        if precise_subtitle_limit <= 0:
            return candidate.subtitles
        if candidate_index >= precise_subtitle_limit:
            logger.info(
                "Skipping precise subtitle refinement for clip %s: limit reached (%s)",
                candidate_index,
                precise_subtitle_limit,
            )
            return candidate.subtitles
        temp_audio_path = os.path.join(request.output_dir, f"clip_{candidate_index}_subtitle_refine.wav")
        segment_audio = extract_audio_segment(
            request.input_path,
            candidate.start,
            candidate.end,
            temp_audio_path,
        )
        if not segment_audio:
            return candidate.subtitles
        try:
            _transcript, precise_segments = await transcribe_precise_clip(segment_audio)
            if not precise_segments:
                return candidate.subtitles
            clip_pauses = detect_speech_pauses(segment_audio)
            precise_subtitles = build_dynamic_subtitles(
                precise_segments,
                0.0,
                candidate.end - candidate.start,
                pauses=clip_pauses,
            )
            if precise_subtitles:
                logger.info(
                    "Refined subtitles for clip %s with %s timed entries",
                    candidate_index,
                    len(precise_subtitles),
                )
                return precise_subtitles
            return candidate.subtitles
        except Exception as exc:
            logger.warning("Precise subtitle refinement failed for clip %s: %s", candidate_index, exc)
            return candidate.subtitles
        finally:
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass

    async def process(self, request: ProcessingRequest) -> List[str]:
        process_started_at = time.perf_counter()
        self._report_progress(2.0, "Preparing output folder")
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)
        self._report_progress(6.0, "Extracting audio")
        stage_started_at = time.perf_counter()
        temp_audio_path = self._make_temp_audio_path(request.output_dir)
        audio_path = extract_audio(request.input_path, temp_audio_path)
        if not audio_path:
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass
            raise RuntimeError("Failed to extract audio from the video")
        logger.info("Stage timing: audio extraction completed in %.2fs", time.perf_counter() - stage_started_at)

        self._report_progress(10.0, "Reading source video")
        stage_started_at = time.perf_counter()
        probe = ffmpeg.probe(request.input_path)
        video_duration = float(probe["format"]["duration"])
        logger.info("Stage timing: source probe completed in %.2fs", time.perf_counter() - stage_started_at)
        precise_subtitle_limit = self._resolve_precise_subtitle_limit(request.clip_count, video_duration)
        logger.info(
            "Precise subtitle policy: requested_clips=%s video_duration=%.2fs limit=%s model=%s",
            request.clip_count,
            video_duration,
            precise_subtitle_limit,
            os.environ.get("FINAL_PROJECT_REFINE_ASR_MODEL", "medium"),
        )
        stage_started_at = time.perf_counter()
        pauses = detect_speech_pauses(audio_path) if request.mode == "speech" else []
        logger.info("Stage timing: pause detection completed in %.2fs", time.perf_counter() - stage_started_at)

        stage_started_at = time.perf_counter()
        if request.mode == "speech":
            candidates = await self._get_speech_candidates(
                audio_path,
                video_duration,
                request.min_clip_duration,
                request.max_clip_duration,
                request.clip_count,
                pauses=pauses,
            )
        elif request.mode == "beat":
            candidates = self._get_music_candidates(
                audio_path,
                video_duration,
                request.clip_count,
                request.min_clip_duration,
                request.max_clip_duration,
            )
        else:
            raise ValueError("mode must be either 'speech' or 'beat'")
        logger.info("Stage timing: candidate generation completed in %.2fs", time.perf_counter() - stage_started_at)

        if not candidates:
            raise RuntimeError("No candidates were generated")

        output_paths: List[str] = []
        self._report_progress(52.0, "Selecting clips")
        stage_started_at = time.perf_counter()
        candidates = self._deduplicate_candidates(
            candidates,
            pauses,
            video_duration,
            request.mode,
            request.min_clip_duration,
            request.max_clip_duration,
            request.clip_count,
        )
        logger.info(
            "Selected unique candidates: %s",
            [(round(item.start, 2), round(item.end, 2)) for item in candidates],
        )
        logger.info("Stage timing: clip selection completed in %.2fs", time.perf_counter() - stage_started_at)

        stage_started_at = time.perf_counter()
        total_candidates = max(1, len(candidates))
        render_jobs: List[RenderJob] = []
        banner_render_jobs: List[BannerRenderJob] = []
        banner_paths = self._resolve_banner_paths(request)
        banner_candidate_indices = self._build_banner_candidate_indices(
            sticker_clips_count=request.sticker_clips_count,
            total_candidates=total_candidates,
        )
        banner_folder_map = self._resolve_banner_folder_map(banner_paths)
        reuse_base_renders_for_banners = self._reuse_base_renders_for_banners()
        logger.info(
            "Banner preparation: banners=%s clips_with_banner=%s/%s reuse_base_renders=%s",
            len(banner_paths),
            len(banner_candidate_indices),
            total_candidates,
            reuse_base_renders_for_banners,
        )
        for idx, candidate in enumerate(candidates):
            prep_progress = 55.0 + (((idx + 1) / total_candidates) * 10.0)
            clip_label = f"clip {idx + 1}/{total_candidates}"
            self._report_progress(prep_progress - 4.0, f"Preparing {clip_label}")
            refined_subtitles = await self._refine_candidate_subtitles(
                request,
                candidate,
                idx,
                precise_subtitle_limit,
            )
            output_path_without_banner = self._resolve_clip_output_path(
                request=request,
                candidate_index=idx,
                banner_folder=None,
            )
            render_jobs.append(
                RenderJob(
                    candidate_index=idx,
                    candidate=candidate,
                    output_path=output_path_without_banner,
                    subtitles=refined_subtitles,
                    banner_path=None,
                )
            )
            if idx in banner_candidate_indices:
                for banner_path in banner_paths:
                    output_path_with_banner = self._resolve_clip_output_path(
                        request=request,
                        candidate_index=idx,
                        banner_folder=banner_folder_map[banner_path],
                    )
                    if reuse_base_renders_for_banners:
                        banner_render_jobs.append(
                            BannerRenderJob(
                                candidate_index=idx,
                                candidate=candidate,
                                source_path=output_path_without_banner,
                                output_path=output_path_with_banner,
                                subtitles=refined_subtitles,
                                banner_path=banner_path,
                            )
                        )
                    else:
                        render_jobs.append(
                            RenderJob(
                                candidate_index=idx,
                                candidate=candidate,
                                output_path=output_path_with_banner,
                                subtitles=refined_subtitles,
                                banner_path=banner_path,
                            )
                        )
            self._report_progress(prep_progress, f"Queued {clip_label}")
        logger.info("Stage timing: render job preparation completed in %.2fs", time.perf_counter() - stage_started_at)

        stage_started_at = time.perf_counter()
        render_stage_span = 23.0 if banner_render_jobs else 33.0
        output_paths = await self._render_jobs_parallel(
            request,
            render_jobs,
            stage_start=65.0,
            stage_span=render_stage_span,
        )
        logger.info("Stage timing: rendering completed in %.2fs", time.perf_counter() - stage_started_at)
        if banner_render_jobs:
            stage_started_at = time.perf_counter()
            self._report_progress(88.0, "Creating banner variants")
            output_paths.extend(await self._render_banner_jobs_parallel(request, banner_render_jobs))
            logger.info("Stage timing: banner rendering completed in %.2fs", time.perf_counter() - stage_started_at)

        try:
            os.remove(audio_path)
        except OSError:
            pass

        logger.info("Stage timing: full processing completed in %.2fs", time.perf_counter() - process_started_at)
        self._report_progress(100.0, f"Done: {len(output_paths)} clips")
        return output_paths

    async def process_video(
        self,
        video_path: str,
        output_dir: str,
        mode: str = "speech",
        max_clips: int = 10,
    ) -> List[str]:
        request = ProcessingRequest(
            input_path=video_path,
            output_dir=output_dir,
            mode=mode,
            clip_count=max_clips,
            min_clip_duration=15.0,
            max_clip_duration=20.0,
            subtitle_style=self._subtitle_style,
        )
        return await self.process(request)
