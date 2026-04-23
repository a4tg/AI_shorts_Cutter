"""Speech and beat based segmentation utilities."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .core.constants import MAX_CLIP_DURATION, MAX_NUM_CLIPS, MIN_CLIP_DURATION
from .gpu import torch_cuda_available

logger = logging.getLogger(__name__)

DEFAULT_ASR_MODEL_NAME = os.environ.get("FINAL_PROJECT_ASR_MODEL", "medium")
DEFAULT_REFINE_ASR_MODEL_NAME = os.environ.get("FINAL_PROJECT_REFINE_ASR_MODEL", "medium")
DEFAULT_PRECISE_SUBTITLE_MAX_CLIPS = int(os.environ.get("FINAL_PROJECT_PRECISE_SUBTITLE_MAX_CLIPS", "12"))
DEFAULT_ASR_BEAM_SIZE = int(os.environ.get("FINAL_PROJECT_ASR_BEAM_SIZE", "1"))
DEFAULT_ASR_CHUNK_DURATION = float(os.environ.get("FINAL_PROJECT_ASR_CHUNK_DURATION", "1800"))
DEFAULT_ASR_CHUNK_OVERLAP = float(os.environ.get("FINAL_PROJECT_ASR_CHUNK_OVERLAP", "1.0"))
ENABLE_PRECISE_SUBTITLE_REFINEMENT = os.environ.get("FINAL_PROJECT_ENABLE_PRECISE_SUBTITLES", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Heavy post-processing/model downloads are disabled in the default runtime
# path because they slow down the first real generation a lot and do not
# materially help the core clipping workflow.
ENABLE_TEXT_ANALYSIS = False
ENABLE_TEXT_POSTPROCESS = False
_FASTER_WHISPER_MODELS: Dict[Tuple[str, str, str], object] = {}


def _normalize_asr_language(language: str) -> str:
    normalized = language.strip().lower()
    aliases = {
        "russian": "ru",
        "ru-ru": "ru",
        "english": "en",
        "en-us": "en",
        "en-gb": "en",
    }
    return aliases.get(normalized, normalized)


def _get_faster_whisper_model(model_name: str):
    from faster_whisper import WhisperModel  # type: ignore

    device = "cuda" if torch_cuda_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    cache_key = (model_name, device, compute_type)
    if cache_key not in _FASTER_WHISPER_MODELS:
        logger.info(
            "Loading faster-whisper model: model=%s, device=%s, compute_type=%s",
            model_name,
            device,
            compute_type,
        )
        _FASTER_WHISPER_MODELS[cache_key] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
    return _FASTER_WHISPER_MODELS[cache_key], device, compute_type


def _transcribe_with_faster_whisper(
    audio_path: str,
    language: str,
    model_name: str,
    word_timestamps: bool = False,
    vad_filter: bool = True,
) -> Tuple[str, List[Dict[str, float]]]:
    model, device, compute_type = _get_faster_whisper_model(model_name)
    logger.info(
        "Starting ASR with faster-whisper: model=%s, device=%s, compute_type=%s, beam_size=%s, word_timestamps=%s",
        model_name,
        device,
        compute_type,
        DEFAULT_ASR_BEAM_SIZE,
        word_timestamps,
    )
    segments_iter, _info = model.transcribe(
        audio_path,
        language=_normalize_asr_language(language),
        beam_size=DEFAULT_ASR_BEAM_SIZE,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
        word_timestamps=word_timestamps,
    )

    segments: List[Dict[str, float]] = []
    transcript_parts: List[str] = []
    for segment in segments_iter:
        text = segment.text.strip()
        if not text:
            continue
        transcript_parts.append(text)
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "words": [
                    {
                        "start": float(getattr(word, "start", segment.start)),
                        "end": float(getattr(word, "end", segment.end)),
                        "text": str(getattr(word, "word", "")).strip(),
                    }
                    for word in (getattr(segment, "words", None) or [])
                    if str(getattr(word, "word", "")).strip()
                ],
            }
        )
        if len(segments) % 25 == 0:
            logger.info(
                "ASR progress: %s segments collected, last_end=%.2fs",
                len(segments),
                float(segment.end),
            )
    logger.info("ASR completed with %s segments", len(segments))
    return " ".join(transcript_parts).strip(), segments


def _transcribe_with_openai_whisper(
    audio_path: str,
    language: str,
    model_name: str,
    word_timestamps: bool = False,
) -> Tuple[str, List[Dict[str, float]]]:
    import whisper  # type: ignore

    device = "cuda" if torch_cuda_available() else "cpu"
    logger.info("Starting ASR with openai-whisper fallback: model=%s, device=%s", model_name, device)
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(
        audio_path,
        language=_normalize_asr_language(language),
        word_timestamps=word_timestamps,
    )
    transcript = result.get("text", "")
    segments_raw: Iterable[Dict[str, float]] = result.get("segments", [])
    segments: List[Dict[str, float]] = []
    for seg in segments_raw:
        segments.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"],
                "words": [
                    {
                        "start": float(word["start"]),
                        "end": float(word["end"]),
                        "text": str(word["word"]).strip(),
                    }
                    for word in seg.get("words", [])
                    if str(word.get("word", "")).strip()
                ],
            }
        )
    logger.info("Fallback ASR completed with %s segments", len(segments))
    return transcript, segments


async def transcribe_audio(
    audio_path: str,
    language: str = "russian",
    model_name: str = DEFAULT_ASR_MODEL_NAME,
) -> Tuple[str, List[Dict[str, float]]]:
    logger.info("Preparing transcription for %s", audio_path)
    media_duration = _get_media_duration(audio_path)
    if media_duration > DEFAULT_ASR_CHUNK_DURATION:
        return _transcribe_audio_in_chunks(audio_path, language, model_name, media_duration)
    return _transcribe_audio_once(audio_path, language, model_name)


def _transcribe_audio_once(
    audio_path: str,
    language: str,
    model_name: str,
    word_timestamps: bool = False,
    vad_filter: bool = True,
) -> Tuple[str, List[Dict[str, float]]]:
    try:
        transcript, segments = _transcribe_with_faster_whisper(
            audio_path,
            language,
            model_name,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
        )
        if segments:
            return transcript, segments
    except ImportError:
        logger.info("faster-whisper is not installed, falling back to openai-whisper")
    except Exception as exc:
        logger.warning("faster-whisper failed, falling back to openai-whisper: %s", exc)

    try:
        return _transcribe_with_openai_whisper(
            audio_path,
            language,
            model_name,
            word_timestamps=word_timestamps,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ASR backend is unavailable. Install faster-whisper or openai-whisper."
        ) from exc


def _get_media_duration(audio_path: str) -> float:
    try:
        import ffmpeg  # type: ignore

        probe = ffmpeg.probe(audio_path)
        return float(probe["format"]["duration"])
    except Exception:
        return 0.0


def _extract_audio_chunk(
    audio_path: str,
    chunk_path: str,
    start_time: float,
    duration: float,
) -> None:
    import ffmpeg  # type: ignore

    (
        ffmpeg.input(audio_path, ss=max(0.0, start_time), t=max(0.01, duration))
        .output(chunk_path, acodec="pcm_s16le", ac=1, ar=16000)
        .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
    )


def _transcribe_audio_in_chunks(
    audio_path: str,
    language: str,
    model_name: str,
    media_duration: float,
) -> Tuple[str, List[Dict[str, float]]]:
    logger.info(
        "Long audio detected (%.2fs). Running chunked ASR with %.0fs chunks.",
        media_duration,
        DEFAULT_ASR_CHUNK_DURATION,
    )
    temp_dir = Path(tempfile.mkdtemp(prefix="final_project_asr_"))
    all_segments: List[Dict[str, float]] = []
    transcript_parts: List[str] = []

    try:
        chunk_start = 0.0
        chunk_index = 0
        while chunk_start < media_duration:
            window_start = max(0.0, chunk_start - (DEFAULT_ASR_CHUNK_OVERLAP if chunk_start > 0 else 0.0))
            window_end = min(media_duration, chunk_start + DEFAULT_ASR_CHUNK_DURATION)
            chunk_path = temp_dir / f"chunk_{chunk_index:04d}.wav"
            logger.info(
                "ASR chunk %s: %.2fs -> %.2fs",
                chunk_index + 1,
                window_start,
                window_end,
            )
            _extract_audio_chunk(
                audio_path,
                str(chunk_path),
                window_start,
                window_end - window_start,
            )
            _transcript, chunk_segments = _transcribe_audio_once(str(chunk_path), language, model_name)
            kept_segments: List[Dict[str, float]] = []
            for segment in chunk_segments:
                shifted_start = float(segment["start"]) + window_start
                shifted_end = float(segment["end"]) + window_start
                if shifted_end <= chunk_start or shifted_start >= window_end:
                    continue
                shifted = {
                    "start": max(chunk_start, shifted_start),
                    "end": min(window_end, shifted_end),
                    "text": str(segment["text"]),
                }
                kept_segments.append(shifted)
                transcript_parts.append(str(segment["text"]).strip())
            all_segments.extend(kept_segments)
            logger.info(
                "ASR chunk %s complete: kept %s segments, total=%s",
                chunk_index + 1,
                len(kept_segments),
                len(all_segments),
            )
            try:
                chunk_path.unlink(missing_ok=True)
            except OSError:
                pass
            chunk_index += 1
            chunk_start += DEFAULT_ASR_CHUNK_DURATION
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("Chunked ASR completed with %s segments", len(all_segments))
    return " ".join(part for part in transcript_parts if part).strip(), all_segments


async def transcribe_precise_clip(
    audio_path: str,
    language: str = "russian",
    model_name: str = DEFAULT_REFINE_ASR_MODEL_NAME,
) -> Tuple[str, List[Dict[str, float]]]:
    logger.info("Preparing precise clip transcription for %s with model=%s", audio_path, model_name)
    return _transcribe_audio_once(
        audio_path,
        language,
        model_name,
        word_timestamps=True,
        vad_filter=True,
    )


def _select_pause_aligned_end(
    start: float,
    pauses: List[Tuple[float, float]],
    audio_duration: float,
    min_duration: float,
    max_duration: float,
    grace_duration: float = 3.0,
) -> float:
    min_target = start + min_duration
    max_target = start + max_duration
    pause_starts = [pause_start for pause_start, _ in pauses if pause_start >= min_target]
    within_bounds = [pause_start for pause_start in pause_starts if pause_start <= max_target]
    if within_bounds:
        return within_bounds[-1]
    just_after = [pause_start for pause_start in pause_starts if pause_start <= max_target + grace_duration]
    if just_after:
        return just_after[0]
    return min(max_target, audio_duration)


def detect_music_beats(audio_path: str) -> List[float]:
    try:
        import librosa  # type: ignore
    except ImportError as exc:
        raise ImportError("librosa is required for beat detection; please install it via pip") from exc
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    logger.info("Detected tempo: %.2f BPM, beats: %s", float(tempo), len(beat_times))
    return beat_times


def generate_beats_segments(
    beat_times: List[float],
    video_duration: float,
    min_duration: float = MIN_CLIP_DURATION,
    max_duration: float = MAX_CLIP_DURATION,
) -> List[Dict[str, float]]:
    segments: List[Dict[str, float]] = []
    if not beat_times:
        return segments
    start_index = 0
    while start_index < len(beat_times):
        segment_start = beat_times[start_index]
        end_index = start_index + 1
        while end_index < len(beat_times) and beat_times[end_index] - segment_start < min_duration:
            end_index += 1
        if end_index >= len(beat_times):
            break
        chosen_index = end_index
        while chosen_index < len(beat_times) and beat_times[chosen_index] - segment_start <= max_duration:
            chosen_index += 1
        chosen_index -= 1
        segment_end = min(beat_times[chosen_index], video_duration)
        if segment_end - segment_start > max_duration:
            segment_end = segment_start + max_duration
        segments.append(
            {
                "start": float(segment_start),
                "end": float(segment_end),
                "text": "Beat-based segment",
            }
        )
        start_index = chosen_index + 1
    return segments


def detect_speech_pauses(
    audio_path: str,
    top_db: int = 40,
    frame_length: int = 4096,
    hop_length: int = 1024,
) -> List[Tuple[float, float]]:
    logger.info("Detecting speech pauses")
    try:
        import librosa  # type: ignore
    except ImportError as exc:
        raise ImportError("librosa is required for speech pause detection; please install it") from exc
    y, sr = librosa.load(audio_path, sr=None)
    intervals = librosa.effects.split(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    pauses: List[Tuple[float, float]] = []
    previous_end = 0
    for start, end in intervals:
        if start > previous_end:
            pauses.append((previous_end / sr, start / sr))
        previous_end = end
    if previous_end < len(y):
        pauses.append((previous_end / sr, len(y) / sr))
    logger.info("Detected %s pause intervals", len(pauses))
    return pauses


def postprocess_text(text: str) -> str:
    if not ENABLE_TEXT_POSTPROCESS:
        return text
    try:
        from ruaccent import RUAccent  # type: ignore
    except ImportError:
        return text
    try:
        accentizer = RUAccent()
        accentizer.load(omograph_model_size="turbo", use_dictionary=True)
        return accentizer.process_all(text).replace("+", "")
    except Exception as exc:
        logger.warning("Accentisation failed: %s", exc)
        return text


def split_into_sentences(text: str) -> List[str]:
    try:
        from natasha import Doc, Segmenter  # type: ignore
    except ImportError:
        return [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    segmenter = Segmenter()
    doc = Doc(text)
    doc.segment(segmenter)
    return [sent.text for sent in doc.sents]


def group_segments(
    segments: List[Dict[str, float]],
    transcript: str,
    max_group_duration: float = 12.0,
    max_group_words: int = 28,
) -> List[Dict[str, float]]:
    del transcript
    grouped_segments: List[Dict[str, float]] = []
    current_parts: List[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    def flush_current() -> None:
        nonlocal current_parts, current_start, current_end
        if not current_parts or current_start is None or current_end is None:
            return
        text = postprocess_text(" ".join(current_parts).strip())
        if text:
            grouped_segments.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "text": text,
                }
            )
        current_parts = []
        current_start = None
        current_end = None

    for seg in sorted(segments, key=lambda item: item["start"]):
        seg_text = str(seg["text"]).strip()
        if not seg_text:
            continue
        if current_start is None:
            current_start = float(seg["start"])
        current_parts.append(seg_text)
        current_end = float(seg["end"])
        duration = current_end - current_start
        word_count = len(" ".join(current_parts).split())
        ends_sentence = seg_text.endswith((".", "!", "?", "..."))
        if ends_sentence or duration >= max_group_duration or word_count >= max_group_words:
            flush_current()

    flush_current()
    return grouped_segments or segments


def _fallback_word_entries(segment: Dict[str, float]) -> List[Dict[str, float]]:
    text = str(segment.get("text", "")).strip()
    if not text:
        return []
    words = [word for word in text.split() if word.strip()]
    if not words:
        return []
    start = float(segment["start"])
    end = float(segment["end"])
    duration = max(0.01, end - start)
    slice_duration = duration / len(words)
    entries: List[Dict[str, float]] = []
    for index, word in enumerate(words):
        word_start = start + index * slice_duration
        word_end = end if index == len(words) - 1 else start + (index + 1) * slice_duration
        entries.append({"start": word_start, "end": word_end, "text": word})
    return entries


def build_dynamic_subtitles(
    raw_segments: List[Dict[str, object]],
    clip_start: float,
    clip_end: float,
    max_words_per_caption: int = 3,
    max_chars_per_caption: int = 26,
    pauses: Optional[List[Tuple[float, float]]] = None,
) -> List[Dict[str, object]]:
    word_entries: List[Dict[str, float]] = []
    for segment in raw_segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        if seg_end <= clip_start or seg_start >= clip_end:
            continue
        entries = segment.get("words") or _fallback_word_entries(segment)
        for entry in entries:
            word_start = max(clip_start, float(entry["start"]))
            word_end = min(clip_end, float(entry["end"]))
            text = str(entry.get("text", "")).strip()
            if not text or word_end <= word_start:
                continue
            word_entries.append({"start": word_start, "end": word_end, "text": text})

    if not word_entries:
        return []

    captions: List[Dict[str, object]] = []
    current_words: List[Dict[str, float]] = []

    def flush_current() -> None:
        nonlocal current_words
        if not current_words:
            return
        caption_text = " ".join(str(item["text"]).strip() for item in current_words).strip()
        if caption_text:
            relative_words = [
                {
                    "start": max(0.0, float(item["start"]) - clip_start),
                    "end": max(0.0, float(item["end"]) - clip_start),
                    "text": str(item["text"]).strip(),
                }
                for item in current_words
            ]
            for index, word in enumerate(relative_words):
                captions.append(
                    {
                        "start": float(word["start"]),
                        "end": float(word["end"]),
                        "text": caption_text,
                        "words": relative_words,
                        "active_word_index": index,
                    }
                )
        current_words = []

    for entry in word_entries:
        proposed_words = current_words + [entry]
        proposed_text = " ".join(str(item["text"]).strip() for item in proposed_words).strip()
        gap = 0.0
        if current_words:
            gap = float(entry["start"]) - float(current_words[-1]["end"])
        if (
            current_words
            and (
                len(proposed_words) > max_words_per_caption
                or len(proposed_text) > max_chars_per_caption
                or gap > 0.45
            )
        ):
            flush_current()
        current_words.append(entry)
    flush_current()

    normalized: List[Dict[str, object]] = []
    for caption in captions:
        start = max(0.0, float(caption["start"]))
        end = min(max(0.0, clip_end - clip_start), float(caption["end"]))
        absolute_start = clip_start + start
        absolute_end = clip_start + end
        for pause_start, pause_end in pauses or []:
            if pause_start <= absolute_start < pause_end:
                start = max(0.0, pause_end - clip_start)
                absolute_start = clip_start + start
            if absolute_start < pause_start < absolute_end:
                end = max(start, pause_start - clip_start)
                absolute_end = clip_start + end
                break
        if end <= start:
            end = min(max(0.0, clip_end - clip_start), start + 0.35)
        normalized.append(
            {
                "start": start,
                "end": end,
                "text": str(caption["text"]).strip(),
                "words": caption.get("words", []),
                "active_word_index": int(caption.get("active_word_index", 0)),
            }
        )
    return normalized


def adjust_segment_boundaries(
    segments: List[Dict[str, float]],
    pauses: List[Tuple[float, float]],
    min_duration: float = 30.0,
    max_duration: float = 45.0,
    buffer: float = 1.0,
) -> List[Dict[str, float]]:
    adjusted_segments: List[Dict[str, float]] = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        for pause_start, pause_end in pauses:
            if pause_end <= start and (start - pause_end) <= buffer:
                start = pause_end
                break
        for pause_start, _pause_end in pauses:
            if pause_start >= end and (pause_start - end) <= buffer:
                end = pause_start
                break
        duration = end - start
        if duration < min_duration:
            end = _select_pause_aligned_end(
                start,
                pauses,
                start + max_duration + 3.0,
                min_duration,
                max_duration,
            )
        elif duration > max_duration:
            end = _select_pause_aligned_end(start, pauses, end, min_duration, max_duration)
        adjusted_segments.append({"start": start, "end": end, "text": seg["text"]})
    return adjusted_segments


def adjust_clip_boundaries(
    clip_start: float,
    clip_end: float,
    pauses: List[Tuple[float, float]],
    audio_duration: float,
    min_duration: float = MIN_CLIP_DURATION,
    max_duration: float = MAX_CLIP_DURATION,
    buffer: float = 0.5,
) -> Tuple[float, float]:
    del buffer
    prev_pauses = [pause for pause in pauses if pause[1] <= clip_start]
    next_pauses = [pause for pause in pauses if pause[0] >= clip_end]
    new_start = prev_pauses[-1][1] if prev_pauses else clip_start
    new_end = next_pauses[0][0] if next_pauses else clip_end
    if new_end - new_start < min_duration or new_end - new_start > max_duration:
        new_end = _select_pause_aligned_end(
            new_start,
            pauses,
            audio_duration,
            min_duration,
            max_duration,
        )
    if new_end - new_start > max_duration + 3.0:
        new_end = min(new_start + max_duration, audio_duration)
    return max(0.0, new_start), min(new_end, audio_duration)


def analyze_text(text: str) -> Dict[str, object]:
    analysis = {
        "sentiment": "neutral",
        "confidence": 0.0,
        "keywords": [],
        "entities": {"names": [], "locations": []},
    }
    if not ENABLE_TEXT_ANALYSIS:
        return analysis
    try:
        from transformers import pipeline  # type: ignore

        sentiment_analyzer = pipeline(
            "text-classification",
            model="cointegrated/rubert-tiny2-cedr-emotion-detection",
            device=0 if torch_cuda_available() else -1,
        )
        result = sentiment_analyzer(text[:512])[0]
        analysis["sentiment"] = result["label"]
        analysis["confidence"] = float(result["score"])
    except Exception as exc:
        logger.debug("Sentiment analysis unavailable: %s", exc)
    return analysis


def score_candidate(candidate: Dict[str, object]) -> float:
    analysis = analyze_text(candidate["text"])
    weight = float(analysis.get("confidence", 0.0))
    if len(split_into_sentences(candidate["text"])) == 1:
        weight *= 1.5
    if analysis.get("sentiment") != "neutral":
        weight *= 1.5
    duration = float(candidate["end"]) - float(candidate["start"])
    if duration <= MAX_CLIP_DURATION:
        weight *= 1.2
    base = max(weight, 1.0)
    return base * len(str(candidate["text"]).split())


def text_preprocessing(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^\w\s]", "", lowered)
    return re.sub(r"\s+", " ", lowered)


def deduplicate_candidates(
    candidates: List[Dict[str, object]],
    tol: float = 0.1,
) -> List[Dict[str, object]]:
    deduped: List[Dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda item: item["start"]):
        duplicate = False
        for existing in deduped:
            if (
                abs(float(candidate["start"]) - float(existing["start"])) < tol
                and abs(float(candidate["end"]) - float(existing["end"])) < tol
            ):
                duplicate = True
                break
        if not duplicate:
            deduped.append(candidate)
    return deduped


def iterative_candidate_selection(
    segments: List[Dict[str, object]],
    video_duration: float,
    max_clips: int = MAX_NUM_CLIPS,
) -> List[Dict[str, object]]:
    del video_duration
    remaining = deduplicate_candidates(segments)
    remaining = sorted(remaining, key=lambda item: item["start"])
    selected: List[Dict[str, object]] = []
    while remaining and len(selected) < max_clips:
        candidate = max(remaining, key=score_candidate)
        selected.append(candidate)
        cand_start = float(candidate["start"])
        cand_end = float(candidate["end"])
        remaining = [
            seg
            for seg in remaining
            if float(seg["end"]) <= cand_start or float(seg["start"]) >= cand_end
        ]
    return sorted(selected, key=lambda item: item["start"])


def sequential_candidate_selection(
    segments: List[Dict[str, object]],
    video_duration: float,
    max_clips: int = MAX_NUM_CLIPS,
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    current_time = 0.0
    segments_sorted = sorted(segments, key=lambda item: item["start"])
    while current_time < video_duration and len(selected) < max_clips:
        available = [seg for seg in segments_sorted if float(seg["start"]) >= current_time]
        if not available:
            break
        candidate = max(available, key=score_candidate)
        selected.append(candidate)
        current_time = float(candidate["end"]) + 0.5
    return selected
