"""Command line interface for the unified AI video cutter.

This script exposes the functionality of the ``ShortsGenerator``
through a simple CLI.  Users can supply an input video and choose
between speech based or beat based segmentation.  Additional options
allow custom cropping coordinates and various decorations such as
blur overlays, stickers and sound tracks.  Generated clips are
written into the specified output directory.

Examples::

    python -m final_project.main --input input.mp4 --output out --mode speech \
        --coords 0 0 1920 1080 --blur-radius 50 --sticker logo.png --sound music.mp3

    python -m final_project.main -i video.mp4 -o clips -m beat

Run ``python -m final_project.main --help`` to see all available options.
"""

import argparse
import asyncio
import logging
import re
from pathlib import Path
from typing import Callable, List, Optional

from .core.frame_editor import EditorStandard, FrameEditor
from .decorators.blur_overlay import BlurOverlay
from .decorators.sound_overlay import SoundOverlay
from .generator import ShortsGenerator
from .models import ProcessingRequest, SubtitleStyle
from .runtime_logging import configure_runtime_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
configure_runtime_logging("cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI powered video cutter and resizer")
    parser.add_argument("--input", "-i", required=True, nargs="+", help="One or more input video files")
    parser.add_argument("--output", "-o", required=True, help="Directory to store the output clips")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["speech", "beat"],
        default="speech",
        help="Segmentation mode: 'speech' for speech driven or 'beat' for music driven",
    )
    parser.add_argument(
        "--coords",
        nargs=4,
        type=int,
        metavar=("x1", "y1", "x2", "y2"),
        help="Cropping coordinates (x1 y1 x2 y2) for FrameEditor.  If omitted EditorStandard is used.",
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=0.0,
        help="Radius in pixels for the blur overlay.  Set to 0 to disable",
    )
    parser.add_argument(
        "--sticker",
        type=str,
        default=None,
        help="Path to an image or video file to overlay as a sticker",
    )
    parser.add_argument(
        "--sticker-size",
        nargs=2,
        type=int,
        metavar=("width", "height"),
        default=None,
        help="Size of the sticker in pixels.  Only used if --sticker is provided",
    )
    parser.add_argument(
        "--sticker-position",
        type=str,
        default="bottom",
        help="Position of the sticker (e.g. center, top_right, bottom_left)",
    )
    parser.add_argument(
        "--sound",
        type=str,
        default=None,
        help="Path to an audio file to overlay on the video segments",
    )
    parser.add_argument(
        "--count",
        type=int,
        dest="max_clips",
        help="Alias for --max-clips, intended for simpler UI wiring",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=10,
        help="Maximum number of clips to generate",
    )
    parser.add_argument(
        "--clip-counts",
        nargs="*",
        type=int,
        help="Optional per-input clip counts. Provide one value to reuse for all inputs or one value per input.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=15.0,
        help="Preferred minimum clip duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=20.0,
        help="Preferred maximum clip duration in seconds",
    )
    parser.add_argument(
        "--subtitles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable subtitle generation",
    )
    parser.add_argument(
        "--subtitle-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render subtitles with a semi-transparent background box",
    )
    parser.add_argument(
        "--subtitle-fontsize",
        type=int,
        default=42,
        help="Subtitle font size in points",
    )
    return parser.parse_args()


def _sanitize_output_folder_name(file_path: str) -> str:
    raw_name = Path(file_path).stem.strip() or "video"
    sanitized = re.sub(r'[<>:"/\\\\|?*]+', "_", raw_name).rstrip(" .")
    return sanitized or "video"


def _build_output_directories(input_paths: List[str], output_root: str) -> List[str]:
    if len(input_paths) == 1:
        return [output_root]
    used_names: dict[str, int] = {}
    output_dirs: List[str] = []
    for input_path in input_paths:
        base_name = _sanitize_output_folder_name(input_path)
        occurrence = used_names.get(base_name, 0) + 1
        used_names[base_name] = occurrence
        folder_name = base_name if occurrence == 1 else f"{base_name}_{occurrence}"
        output_dirs.append(str(Path(output_root) / folder_name))
    return output_dirs


def _resolve_clip_counts(
    input_paths: List[str],
    default_clip_count: int,
    configured_counts: Optional[List[int]],
) -> List[int]:
    if not configured_counts:
        return [default_clip_count] * len(input_paths)
    positive_counts = [int(item) for item in configured_counts]
    if any(item <= 0 for item in positive_counts):
        raise ValueError("Clip counts must be greater than zero")
    if len(positive_counts) == 1:
        return positive_counts * len(input_paths)
    if len(positive_counts) != len(input_paths):
        raise ValueError("Clip counts must contain either one value or one value per input video")
    return positive_counts


def _parse_sticker_paths(raw_value: str | None) -> List[str]:
    if not raw_value:
        return []
    normalized = raw_value.replace("\n", ";").replace(",", ";")
    paths: List[str] = []
    seen: set[str] = set()
    for item in normalized.split(";"):
        path = item.strip()
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def build_processing_requests(args: argparse.Namespace) -> List[ProcessingRequest]:
    if args.min_duration <= 0 or args.max_duration <= 0:
        raise ValueError("Clip durations must be greater than zero")
    if args.min_duration > args.max_duration:
        raise ValueError("Minimum duration must be less than or equal to maximum duration")
    input_paths = [str(Path(item)) for item in args.input]
    output_dirs = _build_output_directories(input_paths, args.output)
    clip_counts = _resolve_clip_counts(input_paths, args.max_clips, args.clip_counts)
    subtitle_style = SubtitleStyle(
        enabled=args.subtitles,
        fontsize=args.subtitle_fontsize,
        background_enabled=args.subtitle_background,
    )
    sticker_paths = _parse_sticker_paths(args.sticker)
    requests: List[ProcessingRequest] = []
    for input_path, output_dir, clip_count in zip(input_paths, output_dirs, clip_counts):
        requests.append(
            ProcessingRequest(
                input_path=input_path,
                output_dir=output_dir,
                mode=args.mode,
                clip_count=clip_count,
                min_clip_duration=args.min_duration,
                max_clip_duration=args.max_duration,
                coords=tuple(args.coords) if args.coords else None,
                blur_radius=args.blur_radius,
                sticker_path=sticker_paths[0] if sticker_paths else None,
                sticker_paths=tuple(sticker_paths),
                sticker_size=tuple(args.sticker_size) if args.sticker_size else None,
                sticker_position=args.sticker_position,
                sound_path=args.sound,
                subtitle_style=subtitle_style,
            )
        )
    return requests


def build_processing_request(args: argparse.Namespace) -> ProcessingRequest:
    return build_processing_requests(args)[0]


def build_generator(
    request: ProcessingRequest,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> ShortsGenerator:
    if request.coords:
        x1, y1, x2, y2 = request.coords
        editor = FrameEditor({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    else:
        editor = EditorStandard()
    decorators: List = []
    if request.blur_radius > 0:
        decorators.append(BlurOverlay(radius=request.blur_radius))
    if request.sound_path:
        decorators.append(SoundOverlay(audio_path=request.sound_path))
    return ShortsGenerator(
        editor=editor,
        decorators=decorators,
        subtitle_style=request.subtitle_style,
        progress_callback=progress_callback,
    )


async def main_async(args: argparse.Namespace) -> None:
    requests = build_processing_requests(args)
    for index, request in enumerate(requests, start=1):
        logging.info("Processing video %s/%s: %s", index, len(requests), request.input_path)
        generator = build_generator(request)
        output_paths = await generator.process(request)
        logging.info("Generated %s clips for %s: %s", len(output_paths), request.input_path, output_paths)


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except Exception as exc:
        logging.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
