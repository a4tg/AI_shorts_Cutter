import sys
import asyncio
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
        self.write_videofile_args = args
        self.write_videofile_kwargs = kwargs
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
from final_project import generator as generator_module
from final_project.generator import BannerRenderJob, Candidate, ShortsGenerator
from final_project.gpu import RuntimeDiagnostics, format_runtime_summary, torch_cuda_available
from final_project.gui import (
    QueueItem,
    _format_runtime_duration,
    _merge_banner_path_values,
    build_request_from_form_values,
    build_requests_from_form_values,
    build_requests_from_queue_items,
    filter_font_labels,
)
from final_project.main import build_processing_request, build_processing_requests, parse_args
from final_project.models import ProcessingRequest, SubtitleStyle
from final_project.moviepy_compat import make_reader_close_safe, safe_close_video_clip
from final_project.runtime_logging import configure_runtime_logging
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
    assert args.input == ["in.mp4"]


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


def test_build_processing_requests_create_named_output_subfolders(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            "C:/videos/alpha.mp4",
            "C:/videos/alpha.mov",
            "--output",
            "C:/clips",
        ],
    )
    args = parse_args()

    requests = build_processing_requests(args)

    assert [Path(item.output_dir) for item in requests] == [Path("C:/clips/alpha"), Path("C:/clips/alpha_2")]


def test_build_processing_requests_apply_individual_clip_counts(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            "C:/videos/alpha.mp4",
            "C:/videos/beta.mov",
            "--output",
            "C:/clips",
            "--clip-counts",
            "3",
            "7",
        ],
    )
    args = parse_args()

    requests = build_processing_requests(args)

    assert [item.clip_count for item in requests] == [3, 7]


def test_cli_sticker_argument_accepts_multiple_banner_paths(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input",
            "in.mp4",
            "--output",
            "out",
            "--sticker",
            "C:/banners/one.mov; C:/banners/two.mov",
        ],
    )
    args = parse_args()

    request = build_processing_request(args)

    assert request.sticker_path == "C:/banners/one.mov"
    assert request.sticker_paths == ("C:/banners/one.mov", "C:/banners/two.mov")


def test_editor_standard_builds_blurred_background_and_scaled_foreground():
    clip = DummyClip((1920, 1080))
    editor = EditorStandard()

    result = editor.get_short_video(clip)

    assert hasattr(result, "size")
    assert result.size == (1080, 1920)
    assert len(clip.resize_calls) >= 2
    assert clip.image_transform_calls == 1
    assert clip.crop_calls


def test_editor_standard_fast_background_blur_downscales_before_blurring():
    class FakeCv2:
        INTER_AREA = 3
        INTER_LINEAR = 1

        def __init__(self):
            self.resize_calls = []
            self.blur_calls = []

        def resize(self, frame, size, interpolation):
            self.resize_calls.append((size, interpolation))
            width, height = size
            return np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)

        def GaussianBlur(self, frame, kernel, sigma):
            self.blur_calls.append((frame.shape, kernel, sigma))
            return frame

    fake_cv2 = FakeCv2()
    editor = EditorStandard(blur_kernel=81, background_blur_scale=0.5)
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

    result = editor._blur_frame(frame, fake_cv2)

    assert result.shape == frame.shape
    assert fake_cv2.resize_calls == [((540, 960), fake_cv2.INTER_AREA), ((1080, 1920), fake_cv2.INTER_LINEAR)]
    assert fake_cv2.blur_calls[0][1] == (41, 41)


def test_editor_standard_background_blur_scale_one_uses_full_frame_blur():
    class FakeCv2:
        INTER_AREA = 3
        INTER_LINEAR = 1

        def __init__(self):
            self.resize_calls = []
            self.blur_calls = []

        def resize(self, frame, size, interpolation):
            self.resize_calls.append((size, interpolation))
            return frame

        def GaussianBlur(self, frame, kernel, sigma):
            self.blur_calls.append((frame.shape, kernel, sigma))
            return frame

    fake_cv2 = FakeCv2()
    editor = EditorStandard(blur_kernel=81, background_blur_scale=1.0)
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

    result = editor._blur_frame(frame, fake_cv2)

    assert result is frame
    assert fake_cv2.resize_calls == []
    assert fake_cv2.blur_calls == [((1920, 1080, 3), (81, 81), 0)]


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
    assert request.sticker_paths == ("sticker.gif",)
    assert request.sticker_clips_count is None
    assert request.sticker_position == "below_subtitles_center"
    assert request.sound_path == "music.mp3"
    assert request.subtitle_style.enabled is False
    assert request.subtitle_style.background_enabled is True
    assert request.subtitle_style.font == "C:/Windows/Fonts/arial.ttf"
    assert request.subtitle_style.auto_fit is False


def test_gui_form_values_build_batch_requests_into_named_subfolders():
    requests = build_requests_from_form_values(
        input_path="C:/videos/alpha.mp4; C:/videos/beta.mp4",
        output_dir="C:/output",
        clip_count="7",
    )

    assert [item.input_path for item in requests] == ["C:/videos/alpha.mp4", "C:/videos/beta.mp4"]
    assert [Path(item.output_dir) for item in requests] == [Path("C:/output/alpha"), Path("C:/output/beta")]


def test_gui_form_values_build_batch_requests_apply_individual_clip_counts():
    requests = build_requests_from_form_values(
        input_path="C:/videos/alpha.mp4; C:/videos/beta.mp4",
        output_dir="C:/output",
        clip_count="5",
        per_video_clip_counts="2; 9",
    )

    assert [item.clip_count for item in requests] == [2, 9]


def test_gui_queue_items_build_requests_apply_individual_clip_counts():
    requests = build_requests_from_queue_items(
        queue_items=[
            ("C:/videos/alpha.mp4", 4),
            ("C:/videos/beta.mp4", 6),
        ],
        output_dir="C:/output",
    )

    assert [item.input_path for item in requests] == ["C:/videos/alpha.mp4", "C:/videos/beta.mp4"]
    assert [item.clip_count for item in requests] == [4, 6]
    assert [Path(item.output_dir) for item in requests] == [Path("C:/output/alpha"), Path("C:/output/beta")]


def test_gui_queue_items_apply_banner_only_to_checked_videos():
    requests = build_requests_from_queue_items(
        queue_items=[
            QueueItem("C:/videos/alpha.mp4", 4, use_banner=True),
            QueueItem("C:/videos/beta.mp4", 6, use_banner=False),
        ],
        output_dir="C:/output",
        gif_path="banner.gif",
    )

    assert requests[0].sticker_path == "banner.gif"
    assert requests[0].sticker_paths == ("banner.gif",)
    assert requests[1].sticker_path is None
    assert requests[1].sticker_paths == ()


def test_gui_queue_items_apply_individual_banner_clip_counts():
    requests = build_requests_from_queue_items(
        queue_items=[
            QueueItem("C:/videos/alpha.mp4", 4, use_banner=True, banner_clip_count=1),
            QueueItem("C:/videos/beta.mp4", 6, use_banner=True, banner_clip_count=3),
            QueueItem("C:/videos/gamma.mp4", 5, use_banner=False, banner_clip_count=2),
        ],
        output_dir="C:/output",
        gif_path="banner.gif",
    )

    assert [item.sticker_clips_count for item in requests] == [1, 3, None]
    assert [item.sticker_path for item in requests] == ["banner.gif", "banner.gif", None]


def test_gui_form_values_build_request_with_multiple_banners_and_clip_limit():
    request = build_request_from_form_values(
        input_path="input.mp4",
        output_dir="output",
        clip_count="7",
        gif_path="one.gif; two.gif",
        banner_clip_count="3",
    )

    assert request.sticker_path == "one.gif"
    assert request.sticker_paths == ("one.gif", "two.gif")
    assert request.sticker_clips_count == 3


def test_merge_banner_path_values_appends_without_duplicates():
    merged = _merge_banner_path_values(
        "C:/stickers/one.gif",
        ["C:/stickers/two.gif", "C:/stickers/one.gif"],
    )

    assert merged == "C:/stickers/one.gif; C:/stickers/two.gif"


def test_gui_form_values_reject_invalid_banner_clip_count():
    try:
        build_request_from_form_values(
            input_path="input.mp4",
            output_dir="output",
            clip_count="3",
            gif_path="one.gif",
            banner_clip_count="abc",
        )
    except ValueError as exc:
        assert str(exc) == "Banner clip count must be an integer"
    else:
        raise AssertionError("Expected ValueError for invalid banner clip count")


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


def test_get_speech_candidates_reuses_provided_pauses(monkeypatch):
    async def fake_transcribe_audio(_audio_path):
        return (
            "privet kak dela",
            [{"start": 0.0, "end": 5.0, "text": "privet kak dela"}],
        )

    def fail_if_called(_audio_path):
        raise AssertionError("pause detection should be reused")

    monkeypatch.setattr(generator_module, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(generator_module, "detect_speech_pauses", fail_if_called)

    generator = ShortsGenerator(editor=EditorStandard())

    candidates = asyncio.run(
        generator._get_speech_candidates(
            audio_path="audio.wav",
            video_duration=10.0,
            min_clip_duration=1.0,
            max_clip_duration=5.0,
            clip_count=1,
            pauses=[],
        )
    )

    assert candidates


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


def test_sticker_overlay_below_subtitles_center_position():
    position = StickerOverlay._parse_position("below_subtitles_center", (1080, 1920), (320, 300))
    assert position == ("center", 1860)


def test_sticker_overlay_below_subtitles_center_fits_to_video_width():
    clip = DummyClip((320, 240))
    overlay = StickerOverlay(position="below_subtitles_center")

    result = overlay._fit_to_quadrant(clip, (1080, 1920))

    assert result is clip
    assert clip.resize_calls == [1080 / 320]


def test_sticker_overlay_crop_white_margins_detects_content_bbox():
    class WhiteBorderClip:
        def __init__(self):
            self.cropped_args = None

        def get_frame(self, _t):
            frame = np.full((6, 8, 3), 255, dtype=np.uint8)
            frame[2:5, 3:7] = 0
            return frame

        def cropped(self, **kwargs):
            self.cropped_args = kwargs
            return self

    clip = WhiteBorderClip()

    result = StickerOverlay._crop_white_margins(clip, white_threshold=245)

    assert result is clip
    assert clip.cropped_args == {"x1": 3, "y1": 2, "x2": 7, "y2": 5}


def test_sticker_overlay_caches_static_image_arrays():
    from PIL import Image

    root = Path("test_runtime_logging_tmp") / "sticker_cache"
    root.mkdir(parents=True, exist_ok=True)
    sticker_path = root / "sticker.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(sticker_path)
    with StickerOverlay._STATIC_IMAGE_CACHE_LOCK:
        StickerOverlay._STATIC_IMAGE_CACHE.clear()

    first = StickerOverlay._load_static_image_array(sticker_path)
    second = StickerOverlay._load_static_image_array(sticker_path)

    assert first is second
    assert first.shape == (2, 2, 4)


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


def test_build_export_settings_uses_nvenc_when_available(monkeypatch):
    monkeypatch.setattr(generator_module, "ffmpeg_nvenc_available", lambda: True)
    monkeypatch.setenv("FINAL_PROJECT_NVENC_PRESET", "p7")
    monkeypatch.setenv("FINAL_PROJECT_NVENC_CQ", "18")
    monkeypatch.setenv("FINAL_PROJECT_NVENC_RC", "vbr")

    codec, ffmpeg_params = ShortsGenerator._build_export_settings()

    assert codec == "h264_nvenc"
    assert ffmpeg_params == [
        "-preset",
        "p7",
        "-rc",
        "vbr",
        "-cq",
        "18",
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
    ]


def test_build_export_settings_falls_back_to_x264(monkeypatch):
    monkeypatch.setattr(generator_module, "ffmpeg_nvenc_available", lambda: False)
    monkeypatch.setenv("FINAL_PROJECT_X264_CRF", "17")

    codec, ffmpeg_params = ShortsGenerator._build_export_settings()

    assert codec == "libx264"
    assert ffmpeg_params == ["-crf", "17", "-pix_fmt", "yuv420p"]


def test_write_clip_uses_unique_moviepy_temp_audiofile(monkeypatch):
    clip = _StubClip()
    generator = ShortsGenerator(editor=EditorStandard())
    monkeypatch.setattr(generator_module, "ffmpeg_nvenc_available", lambda: False)

    generator._write_clip(clip, "C:/out/clip_0.mp4", threads=1)

    temp_audiofile = clip.write_videofile_kwargs["temp_audiofile"]
    assert temp_audiofile.endswith(".mp3")
    assert "clip_0_audio_" in temp_audiofile
    assert clip.write_videofile_kwargs["remove_temp"] is True


def test_resolve_parallel_export_plan_uses_cpu_threads_by_default(monkeypatch):
    monkeypatch.delenv("FINAL_PROJECT_PARALLEL_EXPORTS", raising=False)
    monkeypatch.delenv("FINAL_PROJECT_EXPORT_THREADS", raising=False)
    monkeypatch.setattr(generator_module, "ffmpeg_nvenc_available", lambda: False)

    workers, threads = ShortsGenerator._resolve_parallel_export_plan(clip_count=5, cpu_count=8)

    assert workers == 1
    assert threads == 7


def test_resolve_parallel_export_plan_honors_env_override(monkeypatch):
    monkeypatch.setenv("FINAL_PROJECT_PARALLEL_EXPORTS", "3")
    monkeypatch.delenv("FINAL_PROJECT_EXPORT_THREADS", raising=False)
    monkeypatch.setattr(generator_module, "ffmpeg_nvenc_available", lambda: False)

    workers, threads = ShortsGenerator._resolve_parallel_export_plan(clip_count=2, cpu_count=12)

    assert workers == 2
    assert threads == 6


def test_resolve_parallel_export_plan_prefers_more_workers_with_nvenc(monkeypatch):
    monkeypatch.delenv("FINAL_PROJECT_PARALLEL_EXPORTS", raising=False)
    monkeypatch.delenv("FINAL_PROJECT_EXPORT_THREADS", raising=False)
    monkeypatch.setattr(generator_module, "ffmpeg_nvenc_available", lambda: True)

    workers, threads = ShortsGenerator._resolve_parallel_export_plan(clip_count=5, cpu_count=8)

    assert workers == 2
    assert threads == 1


def test_build_banner_candidate_indices_selects_random_subset(monkeypatch):
    monkeypatch.setattr(generator_module.random, "sample", lambda population, k: [0, 2])

    indices = ShortsGenerator._build_banner_candidate_indices(sticker_clips_count=2, total_candidates=4)

    assert indices == {0, 2}


def test_reuse_base_renders_for_banners_reads_env(monkeypatch):
    monkeypatch.setenv("FINAL_PROJECT_REUSE_BASE_FOR_BANNERS", "false")

    assert ShortsGenerator._reuse_base_renders_for_banners() is False

    monkeypatch.setenv("FINAL_PROJECT_REUSE_BASE_FOR_BANNERS", "true")

    assert ShortsGenerator._reuse_base_renders_for_banners() is True


def test_render_banner_jobs_parallel_keeps_multiple_banners(monkeypatch):
    generator = ShortsGenerator(editor=EditorStandard())
    request = ProcessingRequest(input_path="input.mp4", output_dir="output")
    candidate = Candidate(0.0, 10.0, "hello")
    jobs = [
        BannerRenderJob(
            candidate_index=0,
            candidate=candidate,
            source_path="base.mp4",
            output_path="with_banner_a.mp4",
            subtitles=[],
            banner_path="banner_a.png",
        ),
        BannerRenderJob(
            candidate_index=0,
            candidate=candidate,
            source_path="base.mp4",
            output_path="with_banner_b.mp4",
            subtitles=[],
            banner_path="banner_b.png",
        ),
    ]
    seen_banners = []

    monkeypatch.setattr(generator, "_resolve_parallel_export_plan", lambda clip_count, cpu_count=None: (2, 1))

    def fake_render_banner_job(request, job, total_jobs, export_threads, job_index, progress_tracker):
        del request, total_jobs, export_threads, job_index, progress_tracker
        seen_banners.append(job.banner_path)
        return job.output_path

    monkeypatch.setattr(generator, "_render_banner_job", fake_render_banner_job)

    result = asyncio.run(generator._render_banner_jobs_parallel(request, jobs))

    assert sorted(seen_banners) == ["banner_a.png", "banner_b.png"]
    assert sorted(result) == ["with_banner_a.mp4", "with_banner_b.mp4"]


def test_make_temp_audio_path_uses_unique_files_in_output_dir():
    root = Path("test_runtime_logging_tmp") / "audio_temp"
    root.mkdir(parents=True, exist_ok=True)
    first = Path(ShortsGenerator._make_temp_audio_path(str(root)))
    second = Path(ShortsGenerator._make_temp_audio_path(str(root)))

    try:
        assert first.parent == root.resolve()
        assert second.parent == root.resolve()
        assert first != second
        assert first.exists()
        assert second.exists()
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


def test_resolve_clip_output_path_splits_banner_and_non_banner():
    root = Path("test_runtime_logging_tmp") / "banner_output_split"
    root.mkdir(parents=True, exist_ok=True)
    request = ProcessingRequest(
        input_path="in.mp4",
        output_dir=str(root / "video_a"),
    )

    with_banner = ShortsGenerator._resolve_clip_output_path(
        request,
        candidate_index=1,
        banner_folder="with_banner_banner_a",
    )
    without_banner = ShortsGenerator._resolve_clip_output_path(request, candidate_index=2, banner_folder=None)

    assert Path(with_banner).parent.name == "with_banner_banner_a"
    assert Path(without_banner).parent.name == "without_banner"
    assert Path(with_banner).name == "clip_1.mp4"
    assert Path(without_banner).name == "clip_2.mp4"


def test_resolve_banner_folder_map_builds_unique_named_folders():
    mapping = ShortsGenerator._resolve_banner_folder_map(
        ["C:/stickers/brand.gif", "D:/more/brand.mp4", "C:/stickers/logo.gif"]
    )

    assert mapping["C:/stickers/brand.gif"] == "with_banner_brand"
    assert mapping["D:/more/brand.mp4"] == "with_banner_brand_2"
    assert mapping["C:/stickers/logo.gif"] == "with_banner_logo"


def test_render_jobs_parallel_retries_sequentially_after_parallel_failure(monkeypatch):
    editor = EditorStandard()
    generator = ShortsGenerator(editor=editor)
    request = types.SimpleNamespace()
    jobs = [types.SimpleNamespace(), types.SimpleNamespace()]
    calls = []

    monkeypatch.setattr(
        generator,
        "_resolve_parallel_export_plan",
        lambda clip_count, cpu_count=None: (2, 1),
    )

    async def fake_render_jobs_with_worker_count(request, jobs, worker_count, export_threads):
        calls.append((worker_count, export_threads))
        if worker_count == 2:
            raise RuntimeError("parallel failure")
        return ["clip_0.mp4", "clip_1.mp4"]

    monkeypatch.setattr(generator, "_render_jobs_with_worker_count", fake_render_jobs_with_worker_count)

    result = asyncio.run(generator._render_jobs_parallel(request, jobs))

    assert result == ["clip_0.mp4", "clip_1.mp4"]
    assert calls == [(2, 1), (1, 1)]


def test_make_reader_close_safe_suppresses_invalid_handle_errors():
    class DummyReader:
        def __init__(self):
            self.proc = types.SimpleNamespace(stdin=None, stdout=None, stderr=None)
            self.last_read = "frame"

    def failing_close(_self, delete_lastread=True):
        del delete_lastread
        exc = OSError("invalid handle")
        exc.winerror = 6
        raise exc

    safe_close = make_reader_close_safe(failing_close)
    reader = DummyReader()

    safe_close(reader)

    assert reader.proc is None
    assert reader.last_read is None


def test_safe_close_video_clip_suppresses_invalid_handle_errors():
    class DummyResource:
        def close(self):
            exc = OSError("invalid handle")
            exc.winerror = 6
            raise exc

    class DummyClip:
        def __init__(self):
            self.audio = DummyResource()
            self.mask = DummyResource()
            self.reader = DummyResource()

        def close(self):
            exc = OSError("invalid handle")
            exc.winerror = 6
            raise exc

    safe_close_video_clip(DummyClip())


def test_precise_subtitle_refinement_skips_after_configured_limit(monkeypatch):
    editor = EditorStandard()
    generator = ShortsGenerator(editor=editor, subtitle_style=SubtitleStyle(enabled=True))
    request = types.SimpleNamespace(mode="speech", output_dir="C:/output", input_path="video.mp4")
    candidate = Candidate(0.0, 10.0, "hello", subtitles=[{"start": 0.0, "end": 1.0, "text": "hello"}])

    monkeypatch.setattr(generator_module, "ENABLE_PRECISE_SUBTITLE_REFINEMENT", True)
    monkeypatch.setattr(generator_module, "DEFAULT_PRECISE_SUBTITLE_MAX_CLIPS", 2)

    result = asyncio.run(generator._refine_candidate_subtitles(request, candidate, 2, 2))

    assert result == candidate.subtitles


def test_resolve_precise_subtitle_limit_disables_refinement_for_very_long_jobs(monkeypatch):
    monkeypatch.setattr(generator_module, "DEFAULT_PRECISE_SUBTITLE_MAX_CLIPS", 12)
    monkeypatch.setattr(generator_module, "VERY_LONG_VIDEO_SECONDS", 3 * 3600)
    monkeypatch.setattr(generator_module, "LONG_VIDEO_SECONDS", 3600)

    limit = ShortsGenerator._resolve_precise_subtitle_limit(clip_count=50, video_duration=100 * 3600)

    assert limit == 0


def test_format_runtime_duration_formats_elapsed_time():
    assert _format_runtime_duration(59) == "00:59"
    assert _format_runtime_duration(3661) == "1:01:01"


def test_configure_runtime_logging_creates_log_file(monkeypatch):
    temp_dir = Path("test_runtime_logging_tmp")
    temp_dir.mkdir(exist_ok=True)
    monkeypatch.chdir(temp_dir)

    log_path = configure_runtime_logging("gui_test")

    assert log_path.exists()
    assert log_path.parent.name == "logs"


def test_parallel_progress_tracker_reports_average_progress():
    updates = []
    tracker = generator_module.ParallelProgressTracker(
        callback=lambda value, message: updates.append((value, message)),
        total_jobs=2,
        stage_start=65.0,
        stage_span=33.0,
    )

    tracker.update(0, 0.5, "Encoding clip 1/2")
    tracker.update(1, 1.0, "Saved clip 2/2")

    assert updates[0] == (73.25, "Encoding clip 1/2")
    assert updates[1] == (89.75, "Saved clip 2/2")


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
