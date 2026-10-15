"""The CTranslate2 engine must honour the same decode contract as MLX.

§14 and §15 were paid for in measurement on one engine. Nothing makes them
carry over to a second one except a test that says so — and the failure mode
is silent: a transcript still appears, it is just four times slower, or full of
loops, or timestamped against audio that was cut twice.

Everything here stubs ``faster_whisper``, so it runs on any machine with no
CTranslate2, no CUDA and no weights.
"""

from __future__ import annotations

import sys
import types
import wave
from pathlib import Path

import pytest

from transcriptor.domain.transcript import AudioSlice


# --------------------------------------------------------------- the stub


class _StubSegment:
    def __init__(self, start, end, text, words=None):
        self.start = start
        self.end = end
        self.text = text
        self.words = words


class _StubWord:
    def __init__(self, start, end, word, probability=0.9):
        self.start = start
        self.end = end
        self.word = word
        self.probability = probability


class _StubInfo:
    def __init__(self, language="en"):
        self.language = language


class _StubModel:
    """Records what it was asked to do, and hands back a lazy generator.

    Lazy on purpose: the real library returns a generator, and an engine that
    forgets to materialise it would move decoding out of the inference
    executor and onto whichever thread happened to iterate. A stub that
    returned a list could not express that bug — §19's "a fake that cannot
    express the failure" — so this one behaves like the real thing.
    """

    def __init__(self, path, **kwargs):
        self.path = path
        self.init_kwargs = kwargs
        self.calls: list[dict] = []
        self.iterated = False
        self.segments = [_StubSegment(0.0, 1.0, "Hello there.")]

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, **kwargs})

        def generate():
            self.iterated = True
            yield from self.segments

        return generate(), _StubInfo()


@pytest.fixture
def stub_faster_whisper(monkeypatch):
    """Install a fake ``faster_whisper`` module and hand back the last model."""
    built: list[_StubModel] = []

    module = types.ModuleType("faster_whisper")

    def WhisperModel(path, **kwargs):  # noqa: N802 - matching the real name
        model = _StubModel(path, **kwargs)
        built.append(model)
        return model

    module.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return built


@pytest.fixture
def wav(tmp_path) -> Path:
    """Four seconds of 16kHz mono PCM, which is what the extract stage makes."""
    path = tmp_path / "speech.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16_000)
        fh.writeframes(b"\x01\x00" * 16_000 * 4)
    return path


def _engine(tmp_path, **kwargs):
    from transcriptor.adapters.faster_whisper_engine import FasterWhisperEngine

    engine = FasterWhisperEngine(
        model="Systran/faster-whisper-large-v3",
        models_dir=tmp_path / "models",
        device=kwargs.pop("device", "cpu"),
        **kwargs,
    )
    engine._resolved = tmp_path / "snapshot"
    return engine


def _transcribe(tmp_path, stub, wav, **kwargs):
    engine = _engine(tmp_path, **kwargs.pop("engine_kwargs", {}))
    engine.transcribe(
        AudioSlice(path=wav, start_s=kwargs.pop("start_s", 0.0), end_s=None),
        kwargs.pop("language", None),
        **kwargs,
    )
    return stub[-1].calls[-1]


# ------------------------------------------------------- the decode contract


def test_previous_text_conditioning_is_off(tmp_path, stub_faster_whisper, wav) -> None:
    """§14, the single biggest throughput lever — ~4x, and 140x less
    repetition. It has to hold on every engine or the measurement was about
    one adapter rather than about Whisper."""
    call = _transcribe(tmp_path, stub_faster_whisper, wav)
    assert call["condition_on_previous_text"] is False


def test_the_temperature_fallback_is_left_alone(
    tmp_path, stub_faster_whisper, wav
) -> None:
    """Pinning temperature=0 looks like a 1.78x speedup and removes the escape
    hatch that detects degenerate decoding: 503 words from 120 seconds."""
    assert "temperature" not in _transcribe(tmp_path, stub_faster_whisper, wav)


def test_the_built_in_vad_is_explicitly_disabled(
    tmp_path, stub_faster_whisper, wav
) -> None:
    """§13 owns voice activity detection.

    faster-whisper has a VAD of its own that MLX does not, so this is the one
    decode option with no counterpart in the older engine. Silero has already
    cut this audio and every timestamp will be mapped back through
    Timeline.to_original — a second pass here would move all of them, and the
    transcript would look entirely plausible while being wrong.
    """
    assert _transcribe(tmp_path, stub_faster_whisper, wav)["vad_filter"] is False


def test_the_seam_prompt_is_passed_through(tmp_path, stub_faster_whisper, wav) -> None:
    """§15: one sentence of context, or the chunk starts cold and loses casing
    and punctuation for the rest of its length."""
    call = _transcribe(
        tmp_path, stub_faster_whisper, wav, prompt="The previous sentence ended here."
    )
    assert call["initial_prompt"] == "The previous sentence ended here."


def test_no_prompt_means_no_initial_prompt_key(
    tmp_path, stub_faster_whisper, wav
) -> None:
    """An empty prompt must be absent rather than empty: the library treats ""
    as a prompt and tokenises it."""
    assert "initial_prompt" not in _transcribe(tmp_path, stub_faster_whisper, wav)


def test_word_timestamps_stay_off_by_default(
    tmp_path, stub_faster_whisper, wav
) -> None:
    assert _transcribe(tmp_path, stub_faster_whisper, wav)["word_timestamps"] is False


# ------------------------------------------------------------ the generator


def test_the_segment_generator_is_materialised_before_returning(
    tmp_path, stub_faster_whisper, wav
) -> None:
    """The trap this library sets.

    ``transcribe`` returns a generator and decodes nothing until it is
    iterated. Handing that generator back would satisfy every type check, pass
    every other test here, and move the actual inference onto whichever thread
    consumed it — breaking N2 and I5 at once, and only in production.
    """
    engine = _engine(tmp_path)
    segments = engine.transcribe(AudioSlice(path=wav, start_s=0.0, end_s=None), None)

    assert stub_faster_whisper[-1].iterated, (
        "decoding must happen inside transcribe(), which only ever runs on the "
        "inference executor"
    )
    assert isinstance(segments, list)


# ------------------------------------------------------------- remapping


def test_timings_are_offset_onto_the_original_timeline(
    tmp_path, stub_faster_whisper, wav
) -> None:
    """The library reports times relative to the slice it was given."""
    engine = _engine(tmp_path)
    engine.transcribe(AudioSlice(path=wav, start_s=0.0, end_s=None), None)
    stub_faster_whisper[-1].segments = [_StubSegment(0.5, 1.5, "Later words.")]

    segments = engine.transcribe(AudioSlice(path=wav, start_s=2.0, end_s=None), None)
    assert segments[0].start == pytest.approx(2.5)
    assert segments[0].end == pytest.approx(3.5)


def test_empty_segments_are_dropped(tmp_path, stub_faster_whisper, wav) -> None:
    engine = _engine(tmp_path)
    engine.transcribe(AudioSlice(path=wav, start_s=0.0, end_s=None), None)
    stub_faster_whisper[-1].segments = [
        _StubSegment(0.0, 1.0, "   "),
        _StubSegment(1.0, 2.0, "Real text."),
    ]
    segments = engine.transcribe(AudioSlice(path=wav, start_s=0.0, end_s=None), None)
    assert [s.text for s in segments] == ["Real text."]


def test_word_timings_are_offset_too(tmp_path, stub_faster_whisper, wav) -> None:
    engine = _engine(tmp_path, word_timestamps=True)
    engine.transcribe(AudioSlice(path=wav, start_s=0.0, end_s=None), None)
    stub_faster_whisper[-1].segments = [
        _StubSegment(0.0, 1.0, "Hi there.", words=[_StubWord(0.0, 0.4, "Hi")])
    ]
    segments = engine.transcribe(AudioSlice(path=wav, start_s=2.0, end_s=None), None)
    assert segments[0].words[0].start == pytest.approx(2.0)
    assert segments[0].words[0].text == "Hi"


# ------------------------------------------------------- device and memory


def test_cpu_gets_int8_and_no_vram_budget(tmp_path) -> None:
    engine = _engine(tmp_path, device="cpu")
    assert engine.compute_type == "int8"
    assert engine.vram_estimate_gb == 0.0, (
        "there is no second memory pool on a CPU, so a VRAM floor there would "
        "refuse work for a reason that does not exist"
    )
    assert engine.vram_gb() is None


def test_cuda_gets_float16_and_a_vram_budget(tmp_path) -> None:
    engine = _engine(tmp_path, device="cuda")
    assert engine.compute_type == "float16"
    assert engine.vram_estimate_gb > 0.0


def test_auto_falls_back_to_cpu_without_ctranslate2(tmp_path, monkeypatch) -> None:
    """The probe must answer "no CUDA" rather than raise when the library is
    missing entirely — doctor calls it before anything is installed."""
    import transcriptor.adapters.faster_whisper_engine as mod

    monkeypatch.setattr(mod, "cuda_device_count", lambda: 0)
    engine = _engine(tmp_path, device="auto")
    assert engine.device == "cpu"


def test_auto_takes_cuda_when_ctranslate2_reports_a_device(
    tmp_path, monkeypatch
) -> None:
    import transcriptor.adapters.faster_whisper_engine as mod

    monkeypatch.setattr(mod, "cuda_device_count", lambda: 1)
    engine = _engine(tmp_path, device="auto")
    assert engine.device == "cuda"


def test_the_model_id_is_the_engine_name(tmp_path) -> None:
    """`name` lands in result_key (§7). If both engines called themselves
    "whisper-large-v3", a CPU transcript would be served as though the GPU had
    produced it — a correctness bug dressed as a cache hit."""
    assert _engine(tmp_path).name == "Systran/faster-whisper-large-v3"


def test_a_missing_model_is_a_named_error_not_a_surprise_download(tmp_path) -> None:
    from transcriptor.adapters.faster_whisper_engine import FasterWhisperEngine
    from transcriptor.domain.errors import ModelNotDownloaded

    engine = FasterWhisperEngine(
        model="Systran/nothing-here", models_dir=tmp_path / "empty"
    )
    with pytest.raises(ModelNotDownloaded) as excinfo:
        engine._resolve()
    assert excinfo.value.code == "model_not_downloaded"
    assert "transcriptor-mcp setup" in excinfo.value.user_message


def test_the_weights_are_never_fetched_mid_job(
    tmp_path, stub_faster_whisper, wav
) -> None:
    """local_files_only, so a load cannot quietly become a download."""
    engine = _engine(tmp_path)
    engine._load_sync()
    assert stub_faster_whisper[-1].init_kwargs["local_files_only"] is True


# --------------------------------------------------------------- wav window


def test_only_the_requested_window_is_read(tmp_path, stub_faster_whisper, wav) -> None:
    """Seeking rather than reading the whole file and slicing it: a 4-hour
    source must not cost 4 hours of IO per chunk."""
    engine = _engine(tmp_path)
    engine.transcribe(AudioSlice(path=wav, start_s=1.0, end_s=2.0), None)
    audio = stub_faster_whisper[-1].calls[-1]["audio"]
    assert len(audio) == 16_000, "one second at 16kHz"


def test_a_window_past_the_end_is_empty_not_an_error(
    tmp_path, stub_faster_whisper, wav
) -> None:
    engine = _engine(tmp_path)
    assert engine.transcribe(AudioSlice(path=wav, start_s=99.0, end_s=None), None) == []
