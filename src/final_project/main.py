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
from typing import Callable, List, Optional

from .core.frame_editor import EditorStandard, FrameEditor
from .decorators.blur_overlay import BlurOverlay
from .decorators.sound_overlay import SoundOverlay
from .decorators.sticker_overlay import StickerOverlay
from .generator import ShortsGenerator
from .models import ProcessingRequest, SubtitleStyle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI powered video cutter and resizer")
    parser.add_argument("--input", "-i", required=True, help="Path to the input video file")
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


def build_processing_request(args: argparse.Namespace) -> ProcessingRequest:
    if args.min_duration <= 0 or args.max_duration <= 0:
        raise ValueError("Clip durations must be greater than zero")
    if args.min_duration > args.max_duration:
        raise ValueError("Minimum duration must be less than or equal to maximum duration")
    subtitle_style = SubtitleStyle(
        enabled=args.subtitles,
        fontsize=args.subtitle_fontsize,
        background_enabled=args.subtitle_background,
    )
    return ProcessingRequest(
        input_path=args.input,
        output_dir=args.output,
        mode=args.mode,
        clip_count=args.max_clips,
        min_clip_duration=args.min_duration,
        max_clip_duration=args.max_duration,
        coords=tuple(args.coords) if args.coords else None,
        blur_radius=args.blur_radius,
        sticker_path=args.sticker,
        sticker_size=tuple(args.sticker_size) if args.sticker_size else None,
        sticker_position=args.sticker_position,
        sound_path=args.sound,
        subtitle_style=subtitle_style,
    )


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
    if request.sticker_path:
        decorators.append(
            StickerOverlay(
                sticker_path=request.sticker_path,
                size=request.sticker_size,
                position=request.sticker_position,
                opacity=1.0,
            )
        )
    if request.sound_path:
        decorators.append(SoundOverlay(audio_path=request.sound_path))
    return ShortsGenerator(
        editor=editor,
        decorators=decorators,
        subtitle_style=request.subtitle_style,
        progress_callback=progress_callback,
    )


async def main_async(args: argparse.Namespace) -> None:
    request = build_processing_request(args)
    generator = build_generator(request)
    output_paths = await generator.process(request)
    logging.info(f"Generated {len(output_paths)} clips: {output_paths}")


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except Exception as exc:
        logging.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
