"""Decorator to overlay subtitles onto a video fragment."""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import CompositeVideoClip, ImageClip, VideoClip  # type: ignore

from ..core.decorator_interface import DecoratorInterface
from ..core.constants import STANDARD_FOREGROUND_SCALE
from ..models import SubtitleStyle


class SubtitlesOverlay(DecoratorInterface):
    def __init__(
        self,
        subtitles: List[Dict[str, Union[str, float]]],
        style: SubtitleStyle | None = None,
        priority_index: int = 150,
    ) -> None:
        super().__init__(priority_index)
        self._subtitles = subtitles
        self._style = style or SubtitleStyle()

    @staticmethod
    def _find_system_font() -> str:
        candidates = [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return ""

    def _resolve_font(self, font_size: int | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_path = self._style.font or self._find_system_font()
        resolved_size = max(1, int(font_size if font_size is not None else self._style.fontsize))
        if font_path:
            try:
                return ImageFont.truetype(font_path, resolved_size)
            except Exception:
                pass
        return ImageFont.load_default()

    def _resolve_position(
        self,
        overlay_size: Tuple[int, int],
        video_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        overlay_width, overlay_height = overlay_size
        video_width, video_height = video_size
        margin_x = 48
        margin_y = 72
        if isinstance(self._style.position, tuple):
            x, y = self._style.position
            resolved_x = (video_width - overlay_width) // 2 if x == "center" else int(x)
            if y == "bottom":
                resolved_y = video_height - overlay_height - margin_y
            elif y == "below_foreground":
                foreground_height = int(video_width * (9 / 16) * STANDARD_FOREGROUND_SCALE)
                foreground_bottom = int((video_height + foreground_height) / 2)
                resolved_y = min(
                    video_height - overlay_height - margin_y,
                    foreground_bottom + 24,
                )
            elif y == "center":
                resolved_y = (video_height - overlay_height) // 2
            elif y == "top":
                resolved_y = margin_y
            else:
                resolved_y = int(y)
            return resolved_x, resolved_y
        if self._style.position == "center":
            return (video_width - overlay_width) // 2, (video_height - overlay_height) // 2
        return (video_width - overlay_width) // 2, video_height - overlay_height - margin_y

    @staticmethod
    def _parse_color(color: str | Tuple[int, int, int], alpha: int = 255) -> Tuple[int, int, int, int]:
        if isinstance(color, tuple):
            return color[0], color[1], color[2], alpha
        named = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "yellow": (255, 255, 0),
            "red": (255, 0, 0),
        }
        rgb = named.get(str(color).lower(), (255, 255, 255))
        return rgb[0], rgb[1], rgb[2], alpha

    def _wrap_text(
        self,
        words: List[str],
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        stroke_width: int,
        max_text_width: int | None,
    ) -> List[List[str]]:
        if not max_text_width or max_text_width <= 0:
            return [words] if words else []
        if not words:
            return []
        measure_img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        measure_draw = ImageDraw.Draw(measure_img)
        lines: List[List[str]] = []
        current_line: List[str] = [words[0]]
        for word in words[1:]:
            proposed_words = current_line + [word]
            proposed = " ".join(proposed_words)
            bbox = measure_draw.multiline_textbbox(
                (0, 0),
                proposed,
                font=font,
                stroke_width=stroke_width,
                spacing=6,
                align="center",
            )
            if int(bbox[2] - bbox[0]) <= max_text_width:
                current_line = proposed_words
            else:
                lines.append(current_line)
                current_line = [word]
        lines.append(current_line)
        return lines

    def _measure_lines(
        self,
        lines: List[List[str]],
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        stroke_width: int,
    ) -> Tuple[int, int]:
        draw = ImageDraw.Draw(Image.new("RGBA", (10, 10), (0, 0, 0, 0)))
        joined = "\n".join(" ".join(line) for line in lines) or " "
        bbox = draw.multiline_textbbox(
            (0, 0),
            joined,
            font=font,
            stroke_width=stroke_width,
            spacing=6,
            align="center",
        )
        return max(1, int(bbox[2] - bbox[0])), max(1, int(bbox[3] - bbox[1]))

    def _render_subtitle_image(
        self,
        text: str,
        max_width: int | None = None,
        max_height: int | None = None,
        active_word_index: int | None = None,
        words: List[str] | None = None,
    ) -> np.ndarray:
        stroke_width = int(self._style.stroke_width)
        words = words or text.split()
        font = self._resolve_font()
        wrapped_lines = self._wrap_text(words, font, stroke_width, max_width)
        text_width, text_height = self._measure_lines(wrapped_lines, font, stroke_width)
        pad_x, pad_y = (int(self._style.background_padding[0]), int(self._style.background_padding[1]))
        if self._style.auto_fit:
            for candidate_size in range(int(self._style.fontsize), int(self._style.min_fontsize) - 1, -2):
                candidate_font = self._resolve_font(candidate_size)
                candidate_lines = self._wrap_text(words, candidate_font, stroke_width, max_width)
                candidate_width, candidate_height = self._measure_lines(candidate_lines, candidate_font, stroke_width)
                total_width = int(candidate_width + pad_x * 2)
                total_height = int(candidate_height + pad_y * 2)
                if (max_width is None or total_width <= max_width + pad_x * 2) and (
                    max_height is None or total_height <= max_height
                ):
                    font = candidate_font
                    wrapped_lines = candidate_lines
                    text_width = candidate_width
                    text_height = candidate_height
                    break
        image_width = int(text_width + pad_x * 2)
        image_height = int(text_height + pad_y * 2)
        bg_alpha = int(max(0.0, min(1.0, self._style.background_opacity)) * 255)
        background = self._parse_color(self._style.background_color, bg_alpha)
        canvas = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        if self._style.background_enabled:
            draw.rounded_rectangle(
                (0, 0, int(image_width - 1), int(image_height - 1)),
                radius=int(18),
                fill=background,
            )
        line_height_box = draw.textbbox(
            (0, 0),
            "Ag",
            font=font,
            stroke_width=stroke_width,
        )
        line_height = max(1, int(line_height_box[3] - line_height_box[1]))
        spacing = 6
        base_color = self._parse_color(self._style.color)
        highlight_color = self._parse_color(self._style.highlight_color)
        current_y = int(pad_y)
        word_cursor = 0
        for line in wrapped_lines:
            line_text = " ".join(line)
            line_box = draw.textbbox(
                (0, 0),
                line_text,
                font=font,
                stroke_width=stroke_width,
            )
            line_width = int(line_box[2] - line_box[0])
            current_x = int((image_width - line_width) / 2)
            for line_index, word in enumerate(line):
                word_box = draw.textbbox(
                    (0, 0),
                    word,
                    font=font,
                    stroke_width=stroke_width,
                )
                word_width = int(word_box[2] - word_box[0])
                fill_color = highlight_color if active_word_index == word_cursor else base_color
                draw.text(
                    (current_x, current_y),
                    word,
                    font=font,
                    fill=fill_color,
                    stroke_width=stroke_width,
                    stroke_fill=self._parse_color(self._style.stroke_color),
                )
                current_x += word_width
                if line_index < len(line) - 1:
                    space_box = draw.textbbox((0, 0), " ", font=font, stroke_width=stroke_width)
                    current_x += int(space_box[2] - space_box[0])
                word_cursor += 1
            current_y += line_height + spacing
        return np.array(canvas)

    def get_processed_fragment(self, edited_fragment: VideoClip) -> VideoClip:
        if not self._subtitles or not self._style.enabled:
            return edited_fragment
        layers: List[VideoClip] = [edited_fragment]
        max_text_width = int(edited_fragment.size[0] * self._style.max_width_ratio) - int(
            self._style.background_padding[0] * 2
        )
        max_text_height = int(edited_fragment.size[1] * self._style.max_height_ratio)
        for entry in self._subtitles:
            start = float(entry.get("start", 0.0))
            end = float(entry.get("end", 0.0))
            text = str(entry.get("text", "")).strip()
            duration = max(0.0, end - start)
            if not text or duration <= 0:
                continue
            words = [str(item.get("text", "")).strip() for item in entry.get("words", []) if str(item.get("text", "")).strip()]
            subtitle_image = self._render_subtitle_image(
                text,
                max_width=max_text_width,
                max_height=max_text_height,
                active_word_index=int(entry.get("active_word_index", -1)),
                words=words or None,
            )
            subtitle_clip = ImageClip(subtitle_image).with_start(start).with_duration(duration)
            position = self._resolve_position(subtitle_clip.size, edited_fragment.size)
            subtitle_clip = subtitle_clip.with_position(position)
            layers.append(subtitle_clip)
        return CompositeVideoClip(layers, size=edited_fragment.size)
