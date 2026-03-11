"""Simple desktop GUI for the video cutter."""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from final_project.gpu import format_runtime_summary, probe_runtime_diagnostics
    from final_project.main import build_generator
    from final_project.models import ProcessingRequest, SubtitleStyle
else:
    from .gpu import format_runtime_summary, probe_runtime_diagnostics
    from .main import build_generator
    from .models import ProcessingRequest, SubtitleStyle


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
    normalized_gif_path = gif_path.strip() or None
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
        sticker_position="top_right" if normalized_gif_path else "bottom",
        sound_path=normalized_audio_path,
        subtitle_style=subtitle_style,
    )


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
        self.min_duration_var = tk.StringVar(value="15")
        self.max_duration_var = tk.StringVar(value="20")
        self.mode_var = tk.StringVar(value="speech")
        self.gif_path_var = tk.StringVar()
        self.audio_path_var = tk.StringVar()
        self.subtitles_var = tk.BooleanVar(value=True)
        self.subtitle_background_var = tk.BooleanVar(value=True)
        self.subtitle_auto_fit_var = tk.BooleanVar(value=True)
        self._subtitle_font_map = {"Auto (recommended)": ""}
        self._subtitle_font_map.update({label: path for label, path in discover_system_fonts()})
        self._subtitle_font_labels = list(self._subtitle_font_map.keys())
        self.subtitle_font_var = tk.StringVar(value="Auto (recommended)")
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)

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
        outer.rowconfigure(12, weight=1)

        ttk.Label(outer, text="Input video").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.input_path_var).grid(
            row=0, column=1, sticky="ew", padx=(12, 8), pady=(0, 8)
        )
        ttk.Button(outer, text="Browse", command=self._pick_input).grid(row=0, column=2, pady=(0, 8))

        ttk.Label(outer, text="Output folder").grid(row=1, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.output_dir_var).grid(
            row=1, column=1, sticky="ew", padx=(12, 8), pady=(0, 8)
        )
        ttk.Button(outer, text="Browse", command=self._pick_output).grid(row=1, column=2, pady=(0, 8))

        ttk.Label(outer, text="Clips to generate").grid(row=2, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.clip_count_var).grid(
            row=2, column=1, sticky="w", padx=(12, 8), pady=(0, 8)
        )

        ttk.Label(outer, text="Min duration (s)").grid(row=3, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.min_duration_var, width=10).grid(
            row=3, column=1, sticky="w", padx=(12, 8), pady=(0, 8)
        )

        ttk.Label(outer, text="Max duration (s)").grid(row=4, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.max_duration_var, width=10).grid(
            row=4, column=1, sticky="w", padx=(12, 8), pady=(0, 8)
        )

        ttk.Label(outer, text="Mode").grid(row=5, column=0, sticky="w", pady=(0, 8))
        ttk.Combobox(
            outer,
            textvariable=self.mode_var,
            values=("speech", "beat"),
            state="readonly",
        ).grid(row=5, column=1, sticky="w", padx=(12, 8), pady=(0, 8))

        ttk.Label(outer, text="GIF overlay").grid(row=6, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.gif_path_var).grid(
            row=6, column=1, sticky="ew", padx=(12, 8), pady=(0, 8)
        )
        ttk.Button(outer, text="Browse", command=self._pick_gif).grid(row=6, column=2, pady=(0, 8))

        ttk.Label(outer, text="Extra audio").grid(row=7, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(outer, textvariable=self.audio_path_var).grid(
            row=7, column=1, sticky="ew", padx=(12, 8), pady=(0, 8)
        )
        ttk.Button(outer, text="Browse", command=self._pick_audio).grid(row=7, column=2, pady=(0, 8))

        ttk.Label(outer, text="Subtitle font").grid(row=8, column=0, sticky="w", pady=(0, 8))
        self.subtitle_font_combobox = ttk.Combobox(
            outer,
            textvariable=self.subtitle_font_var,
            values=self._subtitle_font_labels,
        )
        self.subtitle_font_combobox.grid(row=8, column=1, sticky="ew", padx=(12, 8), pady=(0, 8))
        self.subtitle_font_combobox.bind("<KeyRelease>", self._on_font_search)
        self.subtitle_font_combobox.bind("<<ComboboxSelected>>", self._on_font_selected)
        ttk.Button(outer, text="Choose file", command=self._pick_subtitle_font).grid(
            row=8, column=2, pady=(0, 8)
        )

        options = ttk.Frame(outer)
        options.grid(row=9, column=0, columnspan=3, sticky="w", pady=(2, 12))
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
        actions.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        actions.columnconfigure(0, weight=1)
        self.start_button = ttk.Button(actions, text="Generate", command=self._start_processing)
        self.start_button.grid(row=0, column=0, sticky="w")
        ttk.Label(actions, textvariable=self.status_var).grid(row=0, column=1, sticky="e")

        self.progress_bar = ttk.Progressbar(
            outer,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
        )
        self.progress_bar.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(0, 12))

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
        self.log_text.grid(row=12, column=0, columnspan=3, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _pick_input(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            title="Choose input video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")],
        )
        if selected:
            self.input_path_var.set(selected)
            if not self.output_dir_var.get().strip():
                self.output_dir_var.set(str(Path(selected).resolve().parent / "output"))

    def _pick_output(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askdirectory(title="Choose output folder")
        if selected:
            self.output_dir_var.set(selected)

    def _pick_gif(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            title="Choose GIF or sticker video",
            filetypes=[
                ("Animated sticker", "*.gif *.mp4 *.mov *.avi"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.gif_path_var.set(selected)

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
                self._is_processing = False
                self.start_button.configure(state="normal")
            elif kind == "error":
                self.status_var.set("Failed")
                self._append_log(message)
                self.progress_var.set(0.0)
                self._is_processing = False
                self.start_button.configure(state="normal")
        self.root.after(150, self._poll_queue)

    def _start_processing(self) -> None:
        if self._is_processing:
            return
        try:
            request = build_request_from_form_values(
                input_path=self.input_path_var.get(),
                output_dir=self.output_dir_var.get(),
                clip_count=self.clip_count_var.get(),
                min_clip_duration=self.min_duration_var.get(),
                max_clip_duration=self.max_duration_var.get(),
                gif_path=self.gif_path_var.get(),
                audio_path=self.audio_path_var.get(),
                mode=self.mode_var.get(),
                subtitles_enabled=self.subtitles_var.get(),
                subtitle_background_enabled=self.subtitle_background_var.get(),
                subtitle_font=self._subtitle_font_map.get(self.subtitle_font_var.get(), ""),
                subtitle_auto_fit=self.subtitle_auto_fit_var.get(),
            )
        except ValueError as exc:
            self._append_log(f"Validation error: {exc}")
            self.status_var.set("Invalid settings")
            return

        self._is_processing = True
        self.start_button.configure(state="disabled")
        self.status_var.set("Processing...")
        self.progress_var.set(0.0)
        self._append_log(
            "Starting: "
            f"mode={request.mode}, clips={request.clip_count}, "
            f"duration={request.min_clip_duration:g}-{request.max_clip_duration:g}s, "
            f"gif={'yes' if request.sticker_path else 'no'}, "
            f"audio={'yes' if request.sound_path else 'no'}, "
            f"input={request.input_path}"
        )

        worker = threading.Thread(
            target=self._run_processing_worker,
            args=(request,),
            daemon=True,
        )
        worker.start()

    def _run_processing_worker(self, request: ProcessingRequest) -> None:
        def _progress_callback(value: float, message: str) -> None:
            self._message_queue.put(("progress", (value, message)))

        try:
            generator = build_generator(request, progress_callback=_progress_callback)
            output_paths = asyncio.run(generator.process(request))
        except BaseException as exc:
            self._message_queue.put(("error", f"Processing error: {exc}"))
            self._message_queue.put(("log", traceback.format_exc()))
            return

        self._message_queue.put(("log", "Completed successfully."))
        for output_path in output_paths:
            self._message_queue.put(("log", output_path))
        self._message_queue.put(("done", f"Done: {len(output_paths)} clips"))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = VideoCutterApp()
    app.run()


if __name__ == "__main__":
    main()
