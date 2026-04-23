"""Simple desktop GUI for the video cutter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import ffmpeg

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from final_project.gpu import format_runtime_summary, probe_runtime_diagnostics
    from final_project.main import build_generator
    from final_project.models import ProcessingRequest, SubtitleStyle
    from final_project.runtime_logging import configure_runtime_logging
else:
    from .gpu import format_runtime_summary, probe_runtime_diagnostics
    from .main import build_generator
    from .models import ProcessingRequest, SubtitleStyle
    from .runtime_logging import configure_runtime_logging


def _parse_input_paths(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(";") if item.strip()]


@dataclass
class QueueItem:
    input_path: str
    clip_count: int
    use_banner: bool = True
    banner_clip_count: int | None = None
    duration_seconds: float | None = None


def _sanitize_output_folder_name(file_path: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    stem = Path(file_path).stem.strip() or "video"
    sanitized = "".join("_" if char in invalid_chars else char for char in stem).rstrip(" .")
    return sanitized or "video"


def _build_batch_output_dirs(input_paths: List[str], output_root: str) -> List[str]:
    if len(input_paths) == 1:
        return [output_root]
    used_names: Dict[str, int] = {}
    output_dirs: List[str] = []
    for input_path in input_paths:
        base_name = _sanitize_output_folder_name(input_path)
        occurrence = used_names.get(base_name, 0) + 1
        used_names[base_name] = occurrence
        folder_name = base_name if occurrence == 1 else f"{base_name}_{occurrence}"
        output_dirs.append(str(Path(output_root) / folder_name))
    return output_dirs


def _probe_video_duration_seconds(input_path: str) -> float | None:
    try:
        probe = ffmpeg.probe(input_path)
        raw_duration = probe.get("format", {}).get("duration")
        if raw_duration is None:
            return None
        duration = float(raw_duration)
        return duration if duration >= 0 else None
    except Exception:
        return None


def _format_duration(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "--:--"
    rounded = max(0, int(round(duration_seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_runtime_duration(duration_seconds: float) -> str:
    total_seconds = max(0, int(duration_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _parse_clip_count_overrides(raw_value: str) -> List[int]:
    normalized = raw_value.replace("\n", ";").replace(",", ";")
    values = [item.strip() for item in normalized.split(";") if item.strip()]
    counts: List[int] = []
    for value in values:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError("Per-video clip counts must be integers") from exc
        if parsed <= 0:
            raise ValueError("Per-video clip counts must be greater than zero")
        counts.append(parsed)
    return counts


def _parse_banner_paths(raw_value: str) -> List[str]:
    normalized = raw_value.replace("\n", ";").replace(",", ";")
    values = [item.strip() for item in normalized.split(";") if item.strip()]
    deduplicated: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return deduplicated


def _merge_banner_path_values(current_value: str, added_paths: List[str]) -> str:
    merged = _parse_banner_paths(current_value)
    seen = set(merged)
    for added_path in added_paths:
        normalized = added_path.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return "; ".join(merged)


def _resolve_batch_clip_counts(
    input_paths: List[str],
    default_clip_count: int,
    raw_overrides: str,
) -> List[int]:
    overrides = _parse_clip_count_overrides(raw_overrides)
    if not overrides:
        return [default_clip_count] * len(input_paths)
    if len(overrides) == 1:
        return overrides * len(input_paths)
    if len(overrides) != len(input_paths):
        raise ValueError("Per-video clip counts must contain either one value or one value per input video")
    return overrides


def build_requests_from_queue_items(
    queue_items: List[QueueItem | Tuple[str, int]],
    output_dir: str,
    min_clip_duration: str = "15",
    max_clip_duration: str = "20",
    gif_path: str = "",
    banner_clip_count: str = "",
    audio_path: str = "",
    mode: str = "speech",
    subtitles_enabled: bool = True,
    subtitle_background_enabled: bool = True,
    subtitle_font: str = "",
    subtitle_auto_fit: bool = True,
) -> List[ProcessingRequest]:
    normalized_queue_items: List[QueueItem] = []
    for item in queue_items:
        if isinstance(item, QueueItem):
            normalized = item
        else:
            normalized = QueueItem(input_path=item[0], clip_count=int(item[1]))
        if normalized.input_path.strip():
            normalized_queue_items.append(normalized)

    input_paths = [item.input_path.strip() for item in normalized_queue_items]
    if not input_paths:
        raise ValueError("At least one input video path is required")
    clip_counts = [int(item.clip_count) for item in normalized_queue_items[: len(input_paths)]]
    if any(count <= 0 for count in clip_counts):
        raise ValueError("Per-video clip counts must be greater than zero")
    if any(item.banner_clip_count is not None and int(item.banner_clip_count) < 0 for item in normalized_queue_items):
        raise ValueError("Per-video banner clip counts must be zero or greater")
    output_root = output_dir.strip()
    if not output_root:
        raise ValueError("Output directory is required")
    output_dirs = _build_batch_output_dirs(input_paths, output_root)
    requests: List[ProcessingRequest] = []
    for queue_item, input_path, output_path, clip_count in zip(
        normalized_queue_items,
        input_paths,
        output_dirs,
        clip_counts,
    ):
        requests.append(
            build_request_from_form_values(
                input_path=input_path,
                output_dir=output_path,
                clip_count=str(clip_count),
                min_clip_duration=min_clip_duration,
                max_clip_duration=max_clip_duration,
                gif_path=gif_path if queue_item.use_banner else "",
                banner_clip_count=(
                    (str(queue_item.banner_clip_count) if queue_item.banner_clip_count is not None else banner_clip_count)
                    if queue_item.use_banner
                    else ""
                ),
                audio_path=audio_path,
                mode=mode,
                subtitles_enabled=subtitles_enabled,
                subtitle_background_enabled=subtitle_background_enabled,
                subtitle_font=subtitle_font,
                subtitle_auto_fit=subtitle_auto_fit,
            )
        )
    return requests


def discover_system_fonts() -> List[Tuple[str, str]]:
    font_dirs = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path("/System/Library/Fonts"),
    ]
    discovered: Dict[str, str] = {}
    for font_dir in font_dirs:
        if not font_dir.exists():
            continue
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            for path in font_dir.rglob(pattern):
                discovered.setdefault(path.stem, str(path))
    return sorted(discovered.items(), key=lambda item: item[0].lower())


def filter_font_labels(labels: List[str], query: str) -> List[str]:
    normalized = query.strip().lower()
    if not normalized:
        return labels
    starts_with = [label for label in labels if label.lower().startswith(normalized)]
    if starts_with:
        return starts_with
    return [label for label in labels if normalized in label.lower()]


def build_request_from_form_values(
    input_path: str,
    output_dir: str,
    clip_count: str,
    min_clip_duration: str = "15",
    max_clip_duration: str = "20",
    gif_path: str = "",
    banner_clip_count: str = "",
    audio_path: str = "",
    mode: str = "speech",
    subtitles_enabled: bool = True,
    subtitle_background_enabled: bool = True,
    subtitle_font: str = "",
    subtitle_auto_fit: bool = True,
) -> ProcessingRequest:
    """Normalize GUI values into a validated processing request."""
    normalized_input = input_path.strip()
    normalized_output = output_dir.strip()
    normalized_banner_paths = _parse_banner_paths(gif_path)
    normalized_gif_path = normalized_banner_paths[0] if normalized_banner_paths else None
    normalized_audio_path = audio_path.strip() or None
    if not normalized_input:
        raise ValueError("Input video path is required")
    if not normalized_output:
        raise ValueError("Output directory is required")

    try:
        normalized_clip_count = int(clip_count)
    except ValueError as exc:
        raise ValueError("Clip count must be an integer") from exc
    try:
        normalized_min_duration = float(min_clip_duration)
        normalized_max_duration = float(max_clip_duration)
    except ValueError as exc:
        raise ValueError("Clip duration values must be numeric") from exc

    if normalized_clip_count <= 0:
        raise ValueError("Clip count must be greater than zero")
    if normalized_min_duration <= 0 or normalized_max_duration <= 0:
        raise ValueError("Clip durations must be greater than zero")
    if normalized_min_duration > normalized_max_duration:
        raise ValueError("Minimum duration must be less than or equal to maximum duration")
    if mode not in {"speech", "beat"}:
        raise ValueError("Mode must be either 'speech' or 'beat'")
    normalized_banner_clip_count: int | None = None
    if normalized_banner_paths:
        raw_banner_clip_count = banner_clip_count.strip()
        if raw_banner_clip_count:
            try:
                normalized_banner_clip_count = int(raw_banner_clip_count)
            except ValueError as exc:
                raise ValueError("Banner clip count must be an integer") from exc
            if normalized_banner_clip_count < 0:
                raise ValueError("Banner clip count must be zero or greater")

    subtitle_style = SubtitleStyle(
        enabled=subtitles_enabled,
        background_enabled=subtitle_background_enabled,
        font=subtitle_font.strip(),
        auto_fit=subtitle_auto_fit,
    )
    return ProcessingRequest(
        input_path=normalized_input,
        output_dir=normalized_output,
        mode=mode,
        clip_count=normalized_clip_count,
        min_clip_duration=normalized_min_duration,
        max_clip_duration=normalized_max_duration,
        sticker_path=normalized_gif_path,
        sticker_paths=tuple(normalized_banner_paths),
        sticker_clips_count=normalized_banner_clip_count,
        sticker_position="below_subtitles_center" if normalized_gif_path else "bottom",
        sound_path=normalized_audio_path,
        subtitle_style=subtitle_style,
    )


def build_requests_from_form_values(
    input_path: str,
    output_dir: str,
    clip_count: str,
    min_clip_duration: str = "15",
    max_clip_duration: str = "20",
    gif_path: str = "",
    banner_clip_count: str = "",
    audio_path: str = "",
    mode: str = "speech",
    subtitles_enabled: bool = True,
    subtitle_background_enabled: bool = True,
    subtitle_font: str = "",
    subtitle_auto_fit: bool = True,
    per_video_clip_counts: str = "",
) -> List[ProcessingRequest]:
    input_paths = _parse_input_paths(input_path)
    if not input_paths:
        raise ValueError("At least one input video path is required")
    output_root = output_dir.strip()
    if not output_root:
        raise ValueError("Output directory is required")
    try:
        default_clip_count = int(clip_count)
    except ValueError as exc:
        raise ValueError("Clip count must be an integer") from exc
    if default_clip_count <= 0:
        raise ValueError("Clip count must be greater than zero")
    output_dirs = _build_batch_output_dirs(input_paths, output_root)
    clip_counts = _resolve_batch_clip_counts(input_paths, default_clip_count, per_video_clip_counts)
    requests: List[ProcessingRequest] = []
    for single_input, single_output_dir, single_clip_count in zip(input_paths, output_dirs, clip_counts):
        requests.append(
            build_request_from_form_values(
                input_path=single_input,
                output_dir=single_output_dir,
                clip_count=str(single_clip_count),
                min_clip_duration=min_clip_duration,
                max_clip_duration=max_clip_duration,
                gif_path=gif_path,
                banner_clip_count=banner_clip_count,
                audio_path=audio_path,
                mode=mode,
                subtitles_enabled=subtitles_enabled,
                subtitle_background_enabled=subtitle_background_enabled,
                subtitle_font=subtitle_font,
                subtitle_auto_fit=subtitle_auto_fit,
            )
        )
    return requests


class VideoCutterApp:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self._ttk = ttk
        self.root = tk.Tk()
        self.root.title("Final Project Video Cutter")
        self.root.geometry("760x590")
        self.root.minsize(700, 520)

        self.input_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.clip_count_var = tk.StringVar(value="5")
        self.per_video_clip_counts_var = tk.StringVar()
        self.selected_queue_clip_count_var = tk.StringVar(value="5")
        self.selected_queue_banner_clip_count_var = tk.StringVar()
        self.min_duration_var = tk.StringVar(value="15")
        self.max_duration_var = tk.StringVar(value="20")
        self.mode_var = tk.StringVar(value="speech")
        self.gif_path_var = tk.StringVar()
        self.banner_clip_count_var = tk.StringVar()
        self.audio_path_var = tk.StringVar()
        self.subtitles_var = tk.BooleanVar(value=True)
        self.subtitle_background_var = tk.BooleanVar(value=True)
        self.subtitle_auto_fit_var = tk.BooleanVar(value=True)
        self._subtitle_font_map = {"Auto (recommended)": ""}
        self._subtitle_font_map.update({label: path for label, path in discover_system_fonts()})
        self._subtitle_font_labels = list(self._subtitle_font_map.keys())
        self.subtitle_font_var = tk.StringVar(value="Auto (recommended)")
        self.status_var = tk.StringVar(value="Ready")
        self.runtime_var = tk.StringVar(value="Elapsed 00:00")
        self.progress_var = tk.DoubleVar(value=0.0)
        self._queue_items: List[QueueItem] = []
        self._processing_started_at: float | None = None

        self._message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._is_processing = False
        self._build_layout()
        self._report_runtime_environment()
        self.root.after(150, self._poll_queue)

    def _build_layout(self) -> None:
        tk = self._tk
        ttk = self._ttk

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        outer = ttk.Frame(self.root, padding=18)
        outer.grid(sticky="nsew")
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(15, weight=1)

        ttk.Label(outer, text="Input video(s)").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.input_path_var).grid(
            row=0, column=1, sticky="ew", padx=(12, 8), pady=(0, 8)
        )
        ttk.Button(outer, text="Add", command=self._pick_input).grid(row=0, column=2, pady=(0, 8))

        queue_frame = ttk.LabelFrame(outer, text="Queue", padding=10)
        queue_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(0, 10))
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.columnconfigure(4, weight=1)
        queue_frame.rowconfigure(0, weight=1)

        self.queue_tree = ttk.Treeview(
            queue_frame,
            columns=("clips", "banner", "banner_clips", "duration"),
            show="tree headings",
            height=6,
            selectmode="extended",
        )
        self.queue_tree.heading("#0", text="Input video")
        self.queue_tree.heading("clips", text="Clips")
        self.queue_tree.heading("banner", text="Banner")
        self.queue_tree.heading("banner_clips", text="Banner clips")
        self.queue_tree.heading("duration", text="Duration")
        self.queue_tree.column("#0", width=520, stretch=True)
        self.queue_tree.column("clips", width=90, anchor="center", stretch=False)
        self.queue_tree.column("banner", width=80, anchor="center", stretch=False)
        self.queue_tree.column("banner_clips", width=100, anchor="center", stretch=False)
        self.queue_tree.column("duration", width=90, anchor="center", stretch=False)
        self.queue_tree.grid(row=0, column=0, columnspan=4, sticky="nsew")
        self.queue_tree.bind("<<TreeviewSelect>>", self._on_queue_selection_changed)
        self.queue_tree.bind("<ButtonRelease-1>", self._on_queue_tree_click)

        ttk.Label(queue_frame, text="Clips for selected").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(queue_frame, textvariable=self.selected_queue_clip_count_var, width=8).grid(
            row=1, column=1, sticky="w", padx=(8, 8), pady=(8, 0)
        )
        ttk.Button(queue_frame, text="Apply to selected", command=self._apply_selected_queue_clip_count).grid(
            row=1, column=2, sticky="w", pady=(8, 0)
        )
        ttk.Button(queue_frame, text="Remove", command=self._remove_selected_queue_item).grid(
            row=1, column=3, sticky="e", padx=(8, 8), pady=(8, 0)
        )
        ttk.Button(queue_frame, text="Clear", command=self._clear_queue).grid(
            row=1, column=4, sticky="e", pady=(8, 0)
        )
        ttk.Button(queue_frame, text="Select all", command=self._select_all_queue_items).grid(
            row=1, column=5, sticky="e", padx=(8, 0), pady=(8, 0)
        )
        ttk.Label(queue_frame, text="Banner clips for selected").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(queue_frame, textvariable=self.selected_queue_banner_clip_count_var, width=8).grid(
            row=2, column=1, sticky="w", padx=(8, 8), pady=(8, 0)
        )
        ttk.Button(
            queue_frame,
            text="Apply banner clips",
            command=self._apply_selected_queue_banner_clip_count,
        ).grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Label(queue_frame, text="empty = all, 0 = none").grid(row=2, column=3, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(queue_frame, text="Use Ctrl/Shift to select multiple videos").grid(
            row=3, column=0, columnspan=6, sticky="w", pady=(8, 0)
        )

        ttk.Label(outer, text="Output folder").grid(row=2, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.output_dir_var).grid(
            row=2, column=1, sticky="ew", padx=(12, 8), pady=(0, 8)
        )
        ttk.Button(outer, text="Browse", command=self._pick_output).grid(row=2, column=2, pady=(0, 8))

        ttk.Label(outer, text="Clips to generate").grid(row=3, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.clip_count_var).grid(
            row=3, column=1, sticky="w", padx=(12, 8), pady=(0, 8)
        )

        ttk.Label(outer, text="Clips per video").grid(row=4, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.per_video_clip_counts_var).grid(
            row=4, column=1, sticky="ew", padx=(12, 8), pady=(0, 8)
        )
        ttk.Label(outer, text="e.g. 5;3;7").grid(row=4, column=2, sticky="w", pady=(0, 8))

        ttk.Label(outer, text="Min duration (s)").grid(row=5, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.min_duration_var, width=10).grid(
            row=5, column=1, sticky="w", padx=(12, 8), pady=(0, 8)
        )

        ttk.Label(outer, text="Max duration (s)").grid(row=6, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.max_duration_var, width=10).grid(
            row=6, column=1, sticky="w", padx=(12, 8), pady=(0, 8)
        )

        ttk.Label(outer, text="Mode").grid(row=7, column=0, sticky="w", pady=(0, 8))
        ttk.Combobox(
            outer,
            textvariable=self.mode_var,
            values=("speech", "beat"),
            state="readonly",
        ).grid(row=7, column=1, sticky="w", padx=(12, 8), pady=(0, 8))

        ttk.Label(outer, text="Banner file(s)").grid(row=8, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.gif_path_var).grid(
            row=8, column=1, sticky="ew", padx=(12, 8), pady=(0, 8)
        )
        ttk.Button(outer, text="Browse", command=self._pick_gif).grid(row=8, column=2, pady=(0, 8))
        ttk.Label(outer, text="e.g. a.gif; b.gif").grid(row=8, column=3, sticky="w", pady=(0, 8))

        ttk.Label(outer, text="Clips with banner").grid(row=9, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.banner_clip_count_var, width=10).grid(
            row=9, column=1, sticky="w", padx=(12, 8), pady=(0, 8)
        )
        ttk.Label(outer, text="empty = all, 0 = none").grid(row=9, column=2, sticky="w", pady=(0, 8))

        ttk.Label(outer, text="Extra audio").grid(row=10, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.audio_path_var).grid(
            row=10, column=1, sticky="ew", padx=(12, 8), pady=(0, 8)
        )
        ttk.Button(outer, text="Browse", command=self._pick_audio).grid(row=10, column=2, pady=(0, 8))

        ttk.Label(outer, text="Subtitle font").grid(row=11, column=0, sticky="w", pady=(0, 8))
        self.subtitle_font_combobox = ttk.Combobox(
            outer,
            textvariable=self.subtitle_font_var,
            values=self._subtitle_font_labels,
        )
        self.subtitle_font_combobox.grid(row=11, column=1, sticky="ew", padx=(12, 8), pady=(0, 8))
        self.subtitle_font_combobox.bind("<KeyRelease>", self._on_font_search)
        self.subtitle_font_combobox.bind("<<ComboboxSelected>>", self._on_font_selected)
        ttk.Button(outer, text="Choose file", command=self._pick_subtitle_font).grid(
            row=11, column=2, pady=(0, 8)
        )

        options = ttk.Frame(outer)
        options.grid(row=12, column=0, columnspan=3, sticky="w", pady=(2, 12))
        ttk.Checkbutton(options, text="Subtitles", variable=self.subtitles_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Checkbutton(
            options,
            text="Subtitle background",
            variable=self.subtitle_background_var,
        ).grid(row=0, column=1, sticky="w", padx=(16, 0))
        ttk.Checkbutton(
            options,
            text="Auto-fit subtitle size",
            variable=self.subtitle_auto_fit_var,
        ).grid(row=0, column=2, sticky="w", padx=(16, 0))

        actions = ttk.Frame(outer)
        actions.grid(row=13, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        actions.columnconfigure(0, weight=1)
        self.start_button = ttk.Button(actions, text="Generate", command=self._start_processing)
        self.start_button.grid(row=0, column=0, sticky="w")
        ttk.Label(actions, textvariable=self.runtime_var).grid(row=0, column=1, sticky="e", padx=(0, 12))
        ttk.Label(actions, textvariable=self.status_var).grid(row=0, column=2, sticky="e")

        self.progress_bar = ttk.Progressbar(
            outer,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
        )
        self.progress_bar.grid(row=14, column=0, columnspan=3, sticky="ew", pady=(0, 12))

        self.log_text = tk.Text(
            outer,
            height=14,
            wrap="word",
            bg="#11161c",
            fg="#f3f4f6",
            insertbackground="#f3f4f6",
            relief="flat",
            padx=12,
            pady=12,
        )
        self.log_text.grid(row=15, column=0, columnspan=3, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _pick_input(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilenames(
            title="Choose input video(s)",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")],
        )
        if selected:
            default_clip_count = self.clip_count_var.get().strip() or "5"
            try:
                parsed_default_clip_count = int(default_clip_count)
            except ValueError:
                parsed_default_clip_count = 5
            raw_default_banner_clip_count = self.banner_clip_count_var.get().strip()
            parsed_default_banner_clip_count: int | None = None
            if raw_default_banner_clip_count:
                try:
                    parsed_default_banner_clip_count = max(0, int(raw_default_banner_clip_count))
                except ValueError:
                    parsed_default_banner_clip_count = None
            for item in selected:
                self._queue_items.append(
                    QueueItem(
                        input_path=item,
                        clip_count=parsed_default_clip_count,
                        use_banner=True,
                        banner_clip_count=parsed_default_banner_clip_count,
                        duration_seconds=_probe_video_duration_seconds(item),
                    )
                )
            self._refresh_queue_view()
            if not self.output_dir_var.get().strip():
                self.output_dir_var.set(str(Path(selected[0]).resolve().parent / "output"))

    def _pick_output(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askdirectory(title="Choose output folder")
        if selected:
            self.output_dir_var.set(selected)

    def _pick_gif(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilenames(
            title="Choose GIF(s) or sticker video(s)",
            filetypes=[
                ("Animated sticker", "*.gif *.mp4 *.mov *.avi"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            merged_value = _merge_banner_path_values(self.gif_path_var.get(), list(selected))
            self.gif_path_var.set(merged_value)

    def _pick_audio(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            title="Choose extra audio",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.audio_path_var.set(selected)

    def _pick_subtitle_font(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            title="Choose subtitle font",
            filetypes=[("Font files", "*.ttf *.otf *.ttc"), ("All files", "*.*")],
        )
        if not selected:
            return
        label = Path(selected).stem
        self._subtitle_font_map[label] = selected
        if label not in self._subtitle_font_labels:
            self._subtitle_font_labels.append(label)
            self._subtitle_font_labels.sort(key=str.lower)
        self.subtitle_font_combobox["values"] = self._subtitle_font_labels
        self.subtitle_font_var.set(label)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _refresh_queue_view(self) -> None:
        self.queue_tree.delete(*self.queue_tree.get_children())
        for index, item in enumerate(self._queue_items):
            self.queue_tree.insert(
                "",
                "end",
                iid=str(index),
                text=item.input_path,
                values=(
                    item.clip_count,
                    "[x]" if item.use_banner else "[ ]",
                    ("all" if item.banner_clip_count is None else item.banner_clip_count) if item.use_banner else "-",
                    _format_duration(item.duration_seconds),
                ),
            )
        self.input_path_var.set("; ".join(item.input_path for item in self._queue_items))
        self.per_video_clip_counts_var.set("; ".join(str(item.clip_count) for item in self._queue_items))
        if self._queue_items:
            self.selected_queue_clip_count_var.set(str(self._queue_items[0].clip_count))
            first_banner_count = self._queue_items[0].banner_clip_count
            self.selected_queue_banner_clip_count_var.set("" if first_banner_count is None else str(first_banner_count))
        else:
            self.selected_queue_clip_count_var.set(self.clip_count_var.get().strip() or "5")
            self.selected_queue_banner_clip_count_var.set(self.banner_clip_count_var.get().strip())

    def _on_queue_selection_changed(self, _event=None) -> None:
        selection = self.queue_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        self.selected_queue_clip_count_var.set(str(self._queue_items[index].clip_count))
        banner_count = self._queue_items[index].banner_clip_count
        self.selected_queue_banner_clip_count_var.set("" if banner_count is None else str(banner_count))

    def _on_queue_tree_click(self, event) -> None:
        row_id = self.queue_tree.identify_row(event.y)
        column_id = self.queue_tree.identify_column(event.x)
        if not row_id or column_id != "#2":
            return
        index = int(row_id)
        self._queue_items[index].use_banner = not self._queue_items[index].use_banner
        self._refresh_queue_view()
        self.queue_tree.selection_set(row_id)

    def _apply_selected_queue_clip_count(self) -> None:
        selection = self.queue_tree.selection()
        if not selection:
            self._append_log("Select a queued video first.")
            return
        try:
            new_count = int(self.selected_queue_clip_count_var.get().strip())
        except ValueError:
            self._append_log("Clip count must be an integer.")
            return
        if new_count <= 0:
            self._append_log("Clip count must be greater than zero.")
            return
        for item_id in selection:
            index = int(item_id)
            self._queue_items[index].clip_count = new_count
        self._refresh_queue_view()
        self.queue_tree.selection_set(selection)

    def _apply_selected_queue_banner_clip_count(self) -> None:
        selection = self.queue_tree.selection()
        if not selection:
            self._append_log("Select a queued video first.")
            return
        raw_value = self.selected_queue_banner_clip_count_var.get().strip()
        if not raw_value:
            new_count: int | None = None
        else:
            try:
                new_count = int(raw_value)
            except ValueError:
                self._append_log("Banner clip count must be an integer.")
                return
            if new_count < 0:
                self._append_log("Banner clip count must be zero or greater.")
                return
        for item_id in selection:
            index = int(item_id)
            self._queue_items[index].banner_clip_count = new_count
        self._refresh_queue_view()
        self.queue_tree.selection_set(selection)

    def _remove_selected_queue_item(self) -> None:
        selection = sorted((int(item_id) for item_id in self.queue_tree.selection()), reverse=True)
        if not selection:
            self._append_log("Select queued video(s) to remove.")
            return
        for index in selection:
            del self._queue_items[index]
        self._refresh_queue_view()

    def _clear_queue(self) -> None:
        self._queue_items.clear()
        self._refresh_queue_view()

    def _select_all_queue_items(self) -> None:
        item_ids = self.queue_tree.get_children()
        if not item_ids:
            self._append_log("Queue is empty.")
            return
        self.queue_tree.selection_set(item_ids)
        self._on_queue_selection_changed()

    def _report_runtime_environment(self) -> None:
        from tkinter import messagebox

        diagnostics = probe_runtime_diagnostics()
        for line in format_runtime_summary(diagnostics):
            self._append_log(line)
        if diagnostics.recommendations:
            messagebox.showwarning(
                "Runtime setup",
                "\n".join(
                    [
                        f"App will run on {diagnostics.execution_device.upper()}.",
                        "",
                        *diagnostics.recommendations,
                    ]
                ),
            )

    def _on_font_search(self, _event=None) -> None:
        filtered = filter_font_labels(self._subtitle_font_labels, self.subtitle_font_var.get())
        self.subtitle_font_combobox["values"] = filtered or self._subtitle_font_labels

    def _on_font_selected(self, _event=None) -> None:
        selected = self.subtitle_font_var.get()
        if selected in self._subtitle_font_map:
            self.subtitle_font_combobox["values"] = self._subtitle_font_labels

    def _poll_queue(self) -> None:
        self._refresh_runtime_display()
        while True:
            try:
                kind, message = self._message_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._append_log(message)
            elif kind == "status":
                self.status_var.set(message)
            elif kind == "progress":
                value, status = message
                self.progress_var.set(float(value))
                self.status_var.set(str(status))
            elif kind == "done":
                self.status_var.set(message)
                self.progress_var.set(100.0)
                self._refresh_runtime_display(final=True)
                self._is_processing = False
                self.start_button.configure(state="normal")
            elif kind == "error":
                self.status_var.set("Failed")
                self._append_log(message)
                self.progress_var.set(0.0)
                self._refresh_runtime_display(final=True)
                self._is_processing = False
                self.start_button.configure(state="normal")
        self.root.after(150, self._poll_queue)

    def _refresh_runtime_display(self, final: bool = False) -> None:
        if self._processing_started_at is None:
            self.runtime_var.set("Elapsed 00:00")
            return
        elapsed = time.monotonic() - self._processing_started_at
        if final or not self._is_processing:
            self.runtime_var.set(f"Elapsed {_format_runtime_duration(elapsed)}")
            return
        progress = float(self.progress_var.get())
        if progress > 0:
            total_estimate = elapsed / max(progress / 100.0, 1e-6)
            eta = max(0.0, total_estimate - elapsed)
            self.runtime_var.set(
                f"Elapsed {_format_runtime_duration(elapsed)} | ETA {_format_runtime_duration(eta)}"
            )
        else:
            self.runtime_var.set(f"Elapsed {_format_runtime_duration(elapsed)} | ETA --:--")

    def _start_processing(self) -> None:
        if self._is_processing:
            return
        try:
            if self._queue_items:
                requests = build_requests_from_queue_items(
                    queue_items=self._queue_items,
                    output_dir=self.output_dir_var.get(),
                    min_clip_duration=self.min_duration_var.get(),
                    max_clip_duration=self.max_duration_var.get(),
                    gif_path=self.gif_path_var.get(),
                    banner_clip_count=self.banner_clip_count_var.get(),
                    audio_path=self.audio_path_var.get(),
                    mode=self.mode_var.get(),
                    subtitles_enabled=self.subtitles_var.get(),
                    subtitle_background_enabled=self.subtitle_background_var.get(),
                    subtitle_font=self._subtitle_font_map.get(self.subtitle_font_var.get(), ""),
                    subtitle_auto_fit=self.subtitle_auto_fit_var.get(),
                )
            else:
                requests = build_requests_from_form_values(
                    input_path=self.input_path_var.get(),
                    output_dir=self.output_dir_var.get(),
                    clip_count=self.clip_count_var.get(),
                    min_clip_duration=self.min_duration_var.get(),
                    max_clip_duration=self.max_duration_var.get(),
                    gif_path=self.gif_path_var.get(),
                    banner_clip_count=self.banner_clip_count_var.get(),
                    audio_path=self.audio_path_var.get(),
                    mode=self.mode_var.get(),
                    subtitles_enabled=self.subtitles_var.get(),
                    subtitle_background_enabled=self.subtitle_background_var.get(),
                    subtitle_font=self._subtitle_font_map.get(self.subtitle_font_var.get(), ""),
                    subtitle_auto_fit=self.subtitle_auto_fit_var.get(),
                    per_video_clip_counts=self.per_video_clip_counts_var.get(),
                )
        except ValueError as exc:
            self._append_log(f"Validation error: {exc}")
            self.status_var.set("Invalid settings")
            return

        self._is_processing = True
        self._processing_started_at = time.monotonic()
        self.start_button.configure(state="disabled")
        self.status_var.set("Processing...")
        self.progress_var.set(0.0)
        self._refresh_runtime_display()
        input_paths = [request.input_path for request in requests]
        self._append_log(
            "Starting: "
            f"videos={len(requests)}, mode={requests[0].mode}, default_clips={self.clip_count_var.get()}, "
            f"duration={requests[0].min_clip_duration:g}-{requests[0].max_clip_duration:g}s, "
            f"banners={len(requests[0].sticker_paths) if requests[0].sticker_paths else 0}, "
            f"banner_clips={'all' if requests[0].sticker_clips_count is None else requests[0].sticker_clips_count}, "
            f"audio={'yes' if requests[0].sound_path else 'no'}, "
            f"input={input_paths[0]}"
        )
        for queued_request in requests:
            self._append_log(
                f"Queued input: {queued_request.input_path} | clips={queued_request.clip_count} | "
                f"banner_paths={len(queued_request.sticker_paths)} | "
                f"banner_clips={'all' if queued_request.sticker_clips_count is None else queued_request.sticker_clips_count}"
            )

        worker = threading.Thread(
            target=self._run_processing_worker,
            args=(requests,),
            daemon=True,
        )
        worker.start()

    def _run_processing_worker(self, requests: List[ProcessingRequest]) -> None:
        try:
            all_output_paths: List[str] = []
            total_requests = max(1, len(requests))
            for index, request in enumerate(requests, start=1):
                def _progress_callback(value: float, message: str, current_index: int = index) -> None:
                    overall_value = (((current_index - 1) + (float(value) / 100.0)) / total_requests) * 100.0
                    self._message_queue.put(
                        ("progress", (overall_value, f"[{current_index}/{total_requests}] {message}"))
                    )

                self._message_queue.put(
                    ("log", f"Processing video {index}/{total_requests}: {request.input_path} -> {request.output_dir}")
                )
                generator = build_generator(request, progress_callback=_progress_callback)
                output_paths = asyncio.run(generator.process(request))
                all_output_paths.extend(output_paths)
                self._message_queue.put(
                    ("log", f"Completed video {index}/{total_requests}: {len(output_paths)} clips"))
        except BaseException as exc:
            self._message_queue.put(("error", f"Processing error: {exc}"))
            self._message_queue.put(("log", traceback.format_exc()))
            return

        self._message_queue.put(("log", "Completed successfully."))
        for output_path in all_output_paths:
            self._message_queue.put(("log", output_path))
        self._message_queue.put(("done", f"Done: {len(all_output_paths)} clips"))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    configure_runtime_logging("gui")
    app = VideoCutterApp()
    app.run()


if __name__ == "__main__":
    main()
