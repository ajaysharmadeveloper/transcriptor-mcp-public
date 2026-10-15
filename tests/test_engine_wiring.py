"""The composition root must hand the engine the inference executor.

MLX streams are per-thread. Loading the model on one thread and transcribing on
another fails at runtime with "There is no Stream(gpu, N) in current thread" —
and only under real MLX, so no FakeEngine test would ever catch it. This asserts
the wiring instead, which needs neither MLX nor weights.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from transcriptor.config import Settings
from transcriptor.server import _build_engine


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Pinned to the MLX engine.

    Not incidental: `engine="auto"` resolves to faster-whisper off Apple
    Silicon, so a fixture that let it float made these tests assert whatever
    the *runner* happened to be. They passed on the author's Mac and failed on
    every CI runner, which is the same class of mistake as a test that asserts
    a consequence instead of causing it — it looked like coverage of the MLX
    wiring and was coverage of nothing in particular.
    """
    return Settings(
        _env_file=None, data_dir=tmp_path, model="test-model", engine="mlx"
    )


@pytest.fixture
def any_engine_settings(tmp_path, request) -> Settings:
    return Settings(
        _env_file=None, data_dir=tmp_path, model="test-model", engine=request.param
    )


def test_the_mlx_engine_is_given_the_inference_executor(settings, monkeypatch) -> None:
    monkeypatch.delenv("TRANSCRIPTOR_FAKE_ENGINE", raising=False)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        engine = _build_engine(settings, executor)
        assert engine._executor is executor, (
            "load() and transcribe() must run on the same thread, or MLX raises "
            "'no Stream(gpu, N) in current thread'"
        )
        assert engine._self_cap_gb == settings.mem_self_cap_gb
        assert engine.name == "test-model"
    finally:
        executor.shutdown(wait=False)


@pytest.mark.parametrize(
    "any_engine_settings", ["mlx", "faster-whisper"], indirect=True
)
def test_every_engine_is_given_the_inference_executor(
    any_engine_settings, monkeypatch
) -> None:
    """The executor must reach whichever engine this machine builds.

    MLX's reason is the loudest — its streams are per-thread and it dies
    outright — but the rule is invariant I5 and it belongs to the port, not to
    one implementation. An engine wired without the executor would run
    inference on the default pool, where nothing bounds concurrency.
    """
    monkeypatch.delenv("TRANSCRIPTOR_FAKE_ENGINE", raising=False)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        engine = _build_engine(any_engine_settings, executor)
        assert engine._executor is executor
        assert engine.name == "test-model"
    finally:
        executor.shutdown(wait=False)


def test_that_executor_has_exactly_one_worker(settings) -> None:
    """max_workers=1 is not a tuning choice: it *is* the Semaphore(1), and it is
    what gives model residency one owner (invariant I5)."""
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        _build_engine(settings, executor)
        assert executor._max_workers == 1
    finally:
        executor.shutdown(wait=False)


def test_the_fake_engine_is_selected_by_env(settings, monkeypatch) -> None:
    monkeypatch.setenv("TRANSCRIPTOR_FAKE_ENGINE", "1")
    engine = _build_engine(settings, None)
    assert engine.name == "fake-engine"
    assert engine.memory_estimate_gb == 0.0


def test_a_missing_model_is_a_named_error_not_a_surprise_download(settings) -> None:
    from transcriptor.domain.errors import ModelNotDownloaded
    from transcriptor.adapters.whisper_mlx import WhisperMLXEngine

    engine = WhisperMLXEngine(
        model="mlx-community/nothing-here", models_dir=Path("/nonexistent")
    )
    with pytest.raises(ModelNotDownloaded) as excinfo:
        engine._resolve()
    assert excinfo.value.code == "model_not_downloaded"
    assert excinfo.value.resumable is False
    assert "transcriptor-mcp setup" in excinfo.value.user_message


def test_memory_estimates_match_the_documented_builds() -> None:
    from transcriptor.adapters.whisper_mlx import MEMORY_ESTIMATE_GB

    assert MEMORY_ESTIMATE_GB["mlx-community/whisper-large-v3-mlx-8bit"] == 2.5
    assert MEMORY_ESTIMATE_GB["mlx-community/whisper-large-v3-mlx"] == 6.5


# ---------------------------------------------------------------- decoding


def _captured_options(monkeypatch, **engine_kwargs) -> dict:
    """Call transcribe with MLX stubbed, and return the options it passed.

    Lets the decode settings be asserted without MLX, weights or Apple Silicon —
    they are a contract with the model, and a silent change to them is a 4x
    throughput swing.
    """
    import sys
    import types

    from transcriptor.adapters.whisper_mlx import WhisperMLXEngine
    from transcriptor.domain.transcript import AudioSlice

    seen: dict = {}

    def fake_transcribe(audio, **kw):
        seen.update(kw)
        return {"segments": [], "language": "en"}

    stub = types.ModuleType("mlx_whisper")
    stub.transcribe = fake_transcribe
    audio_stub = types.ModuleType("mlx_whisper.audio")
    audio_stub.SAMPLE_RATE = 16000
    audio_stub.load_audio = lambda path, sr: __import__("numpy").zeros(sr * 2)
    monkeypatch.setitem(sys.modules, "mlx_whisper", stub)
    monkeypatch.setitem(sys.modules, "mlx_whisper.audio", audio_stub)

    engine = WhisperMLXEngine(model="m", models_dir=Path("/tmp"), **engine_kwargs)
    monkeypatch.setattr(engine, "_resolve", lambda: Path("/tmp/model"))
    engine.transcribe(AudioSlice(path=Path("/tmp/a.wav"), start_s=0.0, end_s=1.0), None)
    return seen


def test_previous_text_conditioning_is_off(monkeypatch) -> None:
    """The single biggest throughput lever, measured at ~4x on real speech.

    With it on, Whisper feeds its own decoded text back as context for each
    successive 30s window; the context grows, each window costs more than the
    last, and a degraded decode makes the model chase its own tail — 140
    repeated 5-grams against 1 with it off.
    """
    assert _captured_options(monkeypatch)["condition_on_previous_text"] is False


def test_the_temperature_fallback_is_left_alone(monkeypatch) -> None:
    """Pinning temperature=0 looks like a 1.78x speedup and is a correctness
    hazard: it removes the escape hatch that detects degenerate decoding. It
    produced 503 words from 120 seconds of speech — about 4.2 words per second,
    which is not physically plausible."""
    assert "temperature" not in _captured_options(monkeypatch)


def test_word_timestamps_stay_off(monkeypatch) -> None:
    """DTW word timings roughly triple segment row size and cost real time; no
    phase 2 tool exposes them."""
    assert _captured_options(monkeypatch)["word_timestamps"] is False


def test_the_seam_prompt_is_passed_through(monkeypatch) -> None:
    from transcriptor.adapters.whisper_mlx import WhisperMLXEngine
    from transcriptor.domain.transcript import AudioSlice
    import sys, types

    seen: dict = {}
    stub = types.ModuleType("mlx_whisper")
    stub.transcribe = lambda audio, **kw: (seen.update(kw), {"segments": [], "language": "en"})[1]
    audio_stub = types.ModuleType("mlx_whisper.audio")
    audio_stub.SAMPLE_RATE = 16000
    audio_stub.load_audio = lambda path, sr: __import__("numpy").zeros(sr * 2)
    monkeypatch.setitem(sys.modules, "mlx_whisper", stub)
    monkeypatch.setitem(sys.modules, "mlx_whisper.audio", audio_stub)

    engine = WhisperMLXEngine(model="m", models_dir=Path("/tmp"))
    monkeypatch.setattr(engine, "_resolve", lambda: Path("/tmp/model"))
    engine.transcribe(
        AudioSlice(path=Path("/tmp/a.wav"), start_s=0.0, end_s=1.0),
        None,
        prompt="The previous sentence ended here.",
    )
    assert seen["initial_prompt"] == "The previous sentence ended here."


# --------------------------------------------------- the GUI launch environment


def test_the_venv_and_platform_dirs_are_put_on_path(monkeypatch) -> None:
    """Claude Desktop launches this server from a GUI context, which inherits a
    PATH of roughly /usr/bin:/bin:/usr/sbin:/sbin. Homebrew's ffmpeg and the
    venv's yt-dlp are both invisible to it, so every subprocess failed with
    "not on PATH" while the binaries were installed and doctor passed in a
    terminal.

    Split on ``os.pathsep``, not on ":" — on Windows the separator is ";" and
    a hardcoded colon would make this pass while asserting nothing."""
    import os
    import sys

    from transcriptor.server import ensure_tool_path

    inherited = os.pathsep.join(_existing_dirs())
    monkeypatch.setenv("PATH", inherited)
    result = ensure_tool_path().split(os.pathsep)

    assert str(Path(sys.executable).parent) in result
    for entry in _existing_dirs():
        assert entry in result, "the inherited PATH is kept, not replaced"


def test_putting_tools_on_path_is_idempotent(monkeypatch) -> None:
    import os

    from transcriptor.server import ensure_tool_path

    monkeypatch.setenv("PATH", os.pathsep.join(_existing_dirs()))
    once = ensure_tool_path()
    twice = ensure_tool_path()
    assert once == twice, "repeated calls must not keep growing PATH"


def test_only_directories_that_exist_are_added(monkeypatch) -> None:
    import os

    from transcriptor.server import ensure_tool_path

    monkeypatch.setenv("PATH", _existing_dirs()[0])
    for entry in ensure_tool_path().split(os.pathsep):
        assert Path(entry).is_dir(), f"{entry} does not exist"


def _existing_dirs() -> list[str]:
    """A couple of directories that really exist on whichever OS is running."""
    import sys

    if sys.platform == "win32":
        return [str(Path(os.environ["SystemRoot"]) / "System32"), os.environ["SystemRoot"]]
    return ["/usr/bin", "/bin"]
