import sys
import types
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


moviepy_stub = types.ModuleType("moviepy")


class _StubClip:
    def __init__(self, *args, **kwargs):
        self.size = kwargs.get("size", (0, 0))
        self.duration = kwargs.get("duration", 0.0)
        self.start = 0.0

    def with_position(self, *args, **kwargs):
        return self

    def with_duration(self, *args, **kwargs):
        return self

    def with_start(self, *args, **kwargs):
        return self

    def with_audio(self, *args, **kwargs):
        return self

    def with_opacity(self, *args, **kwargs):
        return self

    def rotated(self, *args, **kwargs):
        return self

    def resized(self, *args, **kwargs):
        return self

    def cropped(self, *args, **kwargs):
        return self

    def subclipped(self, *args, **kwargs):
        return self

    def image_transform(self, *args, **kwargs):
        return self

    def write_videofile(self, *args, **kwargs):
        return None


moviepy_stub.CompositeVideoClip = _StubClip
moviepy_stub.VideoClip = _StubClip
moviepy_stub.ImageClip = _StubClip
moviepy_stub.ColorClip = _StubClip
moviepy_stub.TextClip = _StubClip
moviepy_stub.VideoFileClip = _StubClip
moviepy_stub.AudioClip = _StubClip
moviepy_stub.AudioFileClip = _StubClip
moviepy_stub.CompositeAudioClip = _StubClip
moviepy_stub.afx = types.SimpleNamespace(AudioLoop=lambda *args, **kwargs: None)
moviepy_stub.vfx = types.SimpleNamespace(Loop=lambda *args, **kwargs: None)

head_blur_module = types.ModuleType("moviepy.video.fx.HeadBlur")


class _HeadBlur:
    def __init__(self, *args, **kwargs):
        pass

    def apply(self, clip):
        return clip


head_blur_module.HeadBlur = _HeadBlur

sys.modules.setdefault("moviepy", moviepy_stub)
sys.modules.setdefault("moviepy.video", types.ModuleType("moviepy.video"))
sys.modules.setdefault("moviepy.video.fx", types.ModuleType("moviepy.video.fx"))
sys.modules.setdefault("moviepy.video.fx.HeadBlur", head_blur_module)

ffmpeg_stub = types.ModuleType("ffmpeg")
ffmpeg_stub.Error = RuntimeError
ffmpeg_stub.input = lambda *args, **kwargs: None
ffmpeg_stub.probe = lambda *args, **kwargs: {"format": {"duration": 0}}
sys.modules.setdefault("ffmpeg", ffmpeg_stub)

from final_project.core.frame_editor import EditorStandard
from final_project.generator import Candidate, ShortsGenerator
from final_project.gpu import RuntimeDiagnostics, format_runtime_summary, torch_cuda_available
from final_project.gui import build_request_from_form_values, filter_font_labels
from final_project.main import build_processing_request, parse_args
from final_project.models import SubtitleStyle
from final_project.decorators.sticker_overlay import StickerOverlay
from final_project.decorators.subtitle_overlay import SubtitlesOverlay
from final_project import segmentation
from final_project.segmentation import adjust_clip_boundaries, generate_beats_segments


class DummyClip:
    def __init__(self, size):
        self.size = size
        self.duration = 10.0
        self.resize_calls = []
        self.crop_calls = []
        self.position = None
        self.image_transform_calls = 0

    def resized(self, arg):
        self.resize_calls.append(arg)
        if isinstance(arg, (int, float)):
            self.size = (int(self.size[0] * arg), int(self.size[1] * arg))
        if isinstance(arg, tuple):
            self.size = arg
        return self

    def cropped(self, **kwargs):
        self.crop_calls.append(kwargs)
        self.size = (
            int(kwargs["x2"] - kwargs["x1"]),
            int(kwargs["y2"] - kwargs["y1"]),
        )
        return self

    def with_position(self, position):
        self.position = position
        return self

    def image_transform(self, _func):
        self.image_transform_calls += 1
        return self


def test_count_argument_overrides_max_clips(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--input", "in.mp4", "--output", "out", "--count", "3"],
    )
    args = parse_args()
    assert args.max_clips == 3


def test_build_processing_request_maps_subtitle_settings(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            "in.mp4",
            "--output",
            "out",
            "--count",
            "2",
            "--subtitle-background",
            "--subtitle-fontsize",
            "54",
        ],
    )
    args = parse_args()
    request = build_processing_request(args)

    assert request.clip_count == 2
    assert request.subtitle_style == SubtitleStyle(
        enabled=True,
        fontsize=54,
        background_enabled=True,
    )
    assert request.subtitle_style.font == ""


def test_editor_standard_builds_blurred_background_and_scaled_foreground():
    clip = DummyClip((1920, 1080))
    editor = EditorStandard()

    result = editor.get_short_video(clip)

    assert hasattr(result, "size")
    assert result.size == (1080, 1920)
    assert len(clip.resize_calls) >= 2
    assert clip.image_transform_calls == 1
    assert clip.crop_calls


def test_gui_form_values_build_request():
    request = build_request_from_form_values(
        input_path="input.mp4",
        output_dir="output",
        clip_count="7",
        min_clip_duration="12",
        max_clip_duration="22",
        gif_path="sticker.gif",
        audio_path="music.mp3",
        mode="beat",
        subtitles_enabled=False,
        subtitle_background_enabled=True,
        subtitle_font="C:/Windows/Fonts/arial.ttf",
        subtitle_auto_fit=False,
    )

    assert request.input_path == "input.mp4"
    assert request.output_dir == "output"
    assert request.clip_count == 7
    assert request.min_clip_duration == 12
    assert request.max_clip_duration == 22
    assert request.mode == "beat"
    assert request.sticker_path == "sticker.gif"
    assert request.sticker_position == "top_right"
    assert request.sound_path == "music.mp3"
    assert request.subtitle_style.enabled is False
    assert request.subtitle_style.background_enabled is True
    assert request.subtitle_style.font == "C:/Windows/Fonts/arial.ttf"
    assert request.subtitle_style.auto_fit is False


def test_gui_form_values_reject_invalid_clip_count():
    try:
        build_request_from_form_values(
            input_path="input.mp4",
            output_dir="output",
            clip_count="abc",
        )
    except ValueError as exc:
        assert str(exc) == "Clip count must be an integer"
    else:
        raise AssertionError("Expected ValueError for invalid clip count")


def test_gui_form_values_reject_invalid_duration_range():
    try:
        build_request_from_form_values(
            input_path="input.mp4",
            output_dir="output",
            clip_count="3",
            min_clip_duration="25",
            max_clip_duration="20",
        )
    except ValueError as exc:
        assert str(exc) == "Minimum duration must be less than or equal to maximum duration"
    else:
        raise AssertionError("Expected ValueError for invalid duration range")


def test_filter_font_labels_prefers_prefix_matches():
    labels = ["Arial", "Game of Thrones KG", "Garamond", "Roboto"]

    filtered = filter_font_labels(labels, "Ga")

    assert filtered == ["Game of Thrones KG", "Garamond"]


def test_torch_cuda_available_uses_torch_runtime(monkeypatch):
    torch_module = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    assert torch_cuda_available() is True


def test_format_runtime_summary_includes_recommendations():
    diagnostics = RuntimeDiagnostics(
        torch_installed=True,
        torch_cuda_available=False,
        torch_version="2.8.0+cpu",
        torch_cuda_version="",
        gpu_name="NVIDIA GeForce RTX 5080",
        nvidia_smi_available=True,
        ffmpeg_available=True,
        ffmpeg_nvenc_available=False,
        execution_device="cpu",
        recommendations=["Install a CUDA-enabled torch build."],
    )

    lines = format_runtime_summary(diagnostics)

    assert "Runtime device: CPU" in lines
    assert any("Recommendation:" in line for line in lines)


def test_transcribe_audio_prefers_faster_whisper(monkeypatch):
    monkeypatch.setattr(segmentation, "_get_media_duration", lambda _path: 0.0)
    monkeypatch.setattr(
        segmentation,
        "_transcribe_with_faster_whisper",
        lambda *args, **kwargs: ("privet", [{"start": 0.0, "end": 1.0, "text": "privet"}]),
    )

    async def _run():
        return await segmentation.transcribe_audio("audio.wav")

    import asyncio

    transcript, segments = asyncio.run(_run())
    assert transcript == "privet"
    assert segments[0]["text"] == "privet"


def test_transcribe_precise_clip_requests_word_timestamps(monkeypatch):
    captured = {}

    def _fake_transcribe(*args, **kwargs):
        captured.update(kwargs)
        return ("precise", [{"start": 0.0, "end": 0.5, "text": "privet", "words": []}])

    monkeypatch.setattr(segmentation, "_transcribe_with_faster_whisper", _fake_transcribe)

    async def _run():
        return await segmentation.transcribe_precise_clip("clip.wav")

    import asyncio

    transcript, segments = asyncio.run(_run())
    assert transcript == "precise"
    assert captured["word_timestamps"] is True
    assert segments[0]["text"] == "privet"


def test_transcribe_audio_uses_chunked_mode_for_long_audio(monkeypatch):
    monkeypatch.setattr(segmentation, "_get_media_duration", lambda _path: 4000.0)
    monkeypatch.setattr(
        segmentation,
        "_transcribe_audio_in_chunks",
        lambda *args, **kwargs: ("chunked", [{"start": 0.0, "end": 1.0, "text": "chunked"}]),
    )

    async def _run():
        return await segmentation.transcribe_audio("audio.wav")

    import asyncio

    transcript, segments = asyncio.run(_run())
    assert transcript == "chunked"
    assert segments[0]["text"] == "chunked"


def test_get_faster_whisper_model_caches_instance(monkeypatch):
    created = []

    class DummyWhisperModel:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))

    monkeypatch.setattr(segmentation, "_FASTER_WHISPER_MODELS", {})
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=DummyWhisperModel))

    model_a, *_ = segmentation._get_faster_whisper_model("medium")
    model_b, *_ = segmentation._get_faster_whisper_model("medium")

    assert model_a is model_b
    assert len(created) == 1


def test_analyze_text_is_lightweight_by_default():
    result = segmentation.analyze_text("privet mir")
    assert result["sentiment"] == "neutral"
    assert result["keywords"] == []
    assert segmentation.ENABLE_TEXT_ANALYSIS is False
    assert segmentation.ENABLE_TEXT_POSTPROCESS is False


def test_subtitles_overlay_system_font_resolver(monkeypatch):
    monkeypatch.setattr(SubtitlesOverlay, "_find_system_font", staticmethod(lambda: "C:/Windows/Fonts/arial.ttf"))
    overlay = SubtitlesOverlay(subtitles=[{"start": 0.0, "end": 1.0, "text": "test"}])
    resolved = overlay._find_system_font()
    assert resolved.endswith("arial.ttf")


def test_sticker_overlay_top_right_position_uses_full_sticker_size():
    position = StickerOverlay._parse_position("top_right", (1080, 1920), (320, 300))
    assert position == (712, 50)


def test_sticker_overlay_bottom_right_position_uses_full_sticker_size():
    position = StickerOverlay._parse_position("bottom_right", (1080, 1920), (320, 300))
    assert position == (712, 1572)


def test_subtitles_overlay_renders_with_float_style_values():
    style = SubtitleStyle(
        fontsize=42,
        stroke_width=2.0,
        background_padding=(40, 24),
    )
    overlay = SubtitlesOverlay(subtitles=[{"start": 0.0, "end": 1.0, "text": "privet"}], style=style)

    image = overlay._render_subtitle_image("privet")

    assert isinstance(image, np.ndarray)
    assert image.ndim == 3


def test_build_dynamic_subtitles_uses_word_timestamps():
    raw_segments = [
        {
            "start": 10.0,
            "end": 12.0,
            "text": "privet kak dela",
            "words": [
                {"start": 10.0, "end": 10.3, "text": "privet"},
                {"start": 10.31, "end": 10.6, "text": "kak"},
                {"start": 10.61, "end": 11.0, "text": "dela"},
            ],
        }
    ]

    subtitles = segmentation.build_dynamic_subtitles(raw_segments, 10.0, 12.0, max_words_per_caption=2)

    assert subtitles
    assert subtitles[0]["start"] == 0.0
    assert subtitles[0]["text"] == "privet kak"
    assert subtitles[0]["active_word_index"] == 0
    assert subtitles[1]["active_word_index"] == 1
    assert len(subtitles[0]["words"]) == 2


def test_build_dynamic_subtitles_trims_caption_on_pause():
    raw_segments = [
        {
            "start": 10.0,
            "end": 12.0,
            "text": "privet kak",
            "words": [
                {"start": 10.0, "end": 10.6, "text": "privet"},
                {"start": 10.61, "end": 11.4, "text": "kak"},
            ],
        }
    ]

    subtitles = segmentation.build_dynamic_subtitles(
        raw_segments,
        10.0,
        12.0,
        pauses=[(10.9, 11.5)],
    )

    assert subtitles
    assert round(float(subtitles[1]["end"]), 2) == 0.9


def test_subtitles_overlay_accepts_highlight_metadata():
    style = SubtitleStyle(highlight_color="yellow")
    overlay = SubtitlesOverlay(
        subtitles=[
            {
                "start": 0.0,
                "end": 0.5,
                "text": "privet kak",
                "words": [{"text": "privet"}, {"text": "kak"}],
                "active_word_index": 1,
            }
        ],
        style=style,
    )

    image = overlay._render_subtitle_image(
        "privet kak",
        max_width=500,
        active_word_index=1,
        words=["privet", "kak"],
    )

    assert isinstance(image, np.ndarray)


def test_subtitles_overlay_positions_below_foreground():
    overlay = SubtitlesOverlay(subtitles=[{"start": 0.0, "end": 1.0, "text": "test"}])

    position = overlay._resolve_position((400, 120), (1080, 1920))

    assert position == (340, 1348)


def test_subtitles_overlay_auto_fits_large_text():
    style = SubtitleStyle(fontsize=60, min_fontsize=20, auto_fit=True, background_padding=(40, 24))
    overlay = SubtitlesOverlay(
        subtitles=[{"start": 0.0, "end": 1.0, "text": "odin dva tri chetyre pyat shest"}],
        style=style,
    )

    image = overlay._render_subtitle_image(
        "odin dva tri chetyre pyat shest",
        max_width=320,
        max_height=140,
        words=["odin", "dva", "tri", "chetyre", "pyat", "shest"],
    )

    assert isinstance(image, np.ndarray)
    assert image.shape[0] <= 140


def test_fit_sticker_duration_trims_to_target_duration():
    clip = _StubClip(duration=2.99)

    result = StickerOverlay._fit_sticker_duration(clip, 20.0)

    assert result is clip


def test_adjust_clip_boundaries_prefers_pause_aligned_end():
    start, end = adjust_clip_boundaries(
        0.0,
        30.0,
        pauses=[(5.0, 6.0), (18.0, 19.0), (21.0, 22.0)],
        audio_duration=40.0,
        min_duration=15.0,
        max_duration=20.0,
    )
    assert start == 0.0
    assert end == 18.0


def test_generator_deduplicates_repeated_candidate_ranges():
    candidates = [
        Candidate(0.0, 20.0, "a"),
        Candidate(0.0, 20.0, "b"),
        Candidate(22.0, 40.0, "c"),
    ]
    unique = ShortsGenerator._deduplicate_candidates(
        candidates=candidates,
        pauses=[],
        video_duration=100.0,
        mode="beat",
        min_duration=15.0,
        max_duration=20.0,
        max_clips=3,
    )
    assert [(item.start, item.end) for item in unique] == [(0.0, 20.0), (22.0, 40.0)]


def test_generator_rejects_overlapping_candidate_ranges():
    candidates = [
        Candidate(0.0, 20.0, "a"),
        Candidate(18.0, 30.0, "b"),
        Candidate(30.0, 44.0, "c"),
        Candidate(43.5, 52.0, "d"),
    ]
    unique = ShortsGenerator._deduplicate_candidates(
        candidates=candidates,
        pauses=[],
        video_duration=100.0,
        mode="speech",
        min_duration=15.0,
        max_duration=20.0,
        max_clips=10,
    )
    assert [(item.start, item.end) for item in unique] == [(0.0, 20.0), (30.0, 44.0)]


def test_speech_coverage_ratio_detects_sparse_speech():
    raw_segments = [
        {"start": 10.0, "end": 11.0, "text": "odin"},
        {"start": 19.0, "end": 19.5, "text": "dva"},
    ]

    ratio = ShortsGenerator._speech_coverage_ratio(raw_segments, 10.0, 20.0)

    assert round(ratio, 2) == 0.15


def test_generate_beats_segments_keeps_segments_within_duration_bounds():
    beat_times = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]

    segments = generate_beats_segments(
        beat_times, video_duration=30.0, min_duration=10.0, max_duration=15.0
    )

    assert segments
    for segment in segments:
        duration = segment["end"] - segment["start"]
        assert duration >= 10.0
        assert duration <= 15.0
