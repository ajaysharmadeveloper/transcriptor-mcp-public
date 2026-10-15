"""Silero VAD against real audio.

project-details §5: "Whisper hallucinates over silence and drifts on long-form
audio. Silero VAD is a required stage... Skipping it produces confident invented
sentences over intro music."

§17 names silence-only and music-bed fixtures that must yield zero segments.
These use the real detector — a fake would prove nothing here.
"""

from __future__ import annotations

import pytest

from transcriptor.adapters.silero_detector import SileroDetector
from transcriptor.domain.transcript import Timeline

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def detector():
    pytest.importorskip("silero_vad", reason="run: uv sync --extra vad")
    return SileroDetector()


# ------------------------------------------------------------ hallucination


def test_silence_yields_no_speech_regions(detector, silence_wav) -> None:
    timeline = detector.speech_regions(silence_wav)
    assert timeline.regions == []
    assert timeline.is_empty


def test_a_music_bed_yields_no_speech_regions(detector, music_bed_wav) -> None:
    """Non-speech audio that is not silent — the case Whisper most reliably
    invents sentences over."""
    timeline = detector.speech_regions(music_bed_wav)
    assert timeline.speech_seconds == 0.0, (
        f"a tone bed was heard as {timeline.speech_seconds:.1f}s of speech"
    )


# ------------------------------------------------------------- real speech


def test_real_speech_is_detected(detector, speech_wav) -> None:
    timeline = detector.speech_regions(speech_wav)
    assert timeline.regions, "20s of continuous speech must be detected"
    assert timeline.speech_seconds > 10.0, (
        f"only {timeline.speech_seconds:.1f}s of 20s found"
    )


def test_regions_stay_inside_the_recording(detector, speech_wav) -> None:
    timeline = detector.speech_regions(speech_wav)
    for start, end in timeline.regions:
        assert 0.0 <= start < end <= 20.5, f"region ({start}, {end}) is out of bounds"


def test_regions_are_ordered_and_do_not_overlap(detector, speech_wav) -> None:
    timeline = detector.speech_regions(speech_wav)
    for previous, following in zip(timeline.regions, timeline.regions[1:]):
        assert previous[1] <= following[0], "merged() must leave no overlap"


def test_micro_pauses_are_not_cut_into_fragments(detector, speech_wav) -> None:
    """Cutting at every breath hands Whisper a stream of fragments with no
    context, which is how VAD makes a transcript worse rather than better."""
    timeline = detector.speech_regions(speech_wav)
    assert len(timeline.regions) <= 8, (
        f"20s of continuous speech split into {len(timeline.regions)} regions"
    )


# ------------------------------------------------------------ speech-only wav


async def test_the_speech_only_wav_contains_only_the_regions(
    detector, speech_wav, tmp_path, executors
) -> None:
    import wave

    from transcriptor.adapters.ffmpeg_audio import FfmpegAudio

    _, cpu = executors
    timeline = detector.speech_regions(speech_wav)
    dest = await FfmpegAudio(cpu_executor=cpu).write_speech_only(
        speech_wav, timeline, tmp_path / "speech-only.wav"
    )

    with wave.open(str(dest), "rb") as fh:
        assert fh.getframerate() == 16000
        assert fh.getnchannels() == 1
        produced = fh.getnframes() / 16000.0

    assert produced == pytest.approx(timeline.speech_seconds, abs=0.05)


async def test_an_empty_timeline_produces_an_empty_wav(
    speech_wav, tmp_path, executors
) -> None:
    import wave

    from transcriptor.adapters.ffmpeg_audio import FfmpegAudio

    _, cpu = executors
    dest = await FfmpegAudio(cpu_executor=cpu).write_speech_only(
        speech_wav, Timeline([]), tmp_path / "empty.wav"
    )
    with wave.open(str(dest), "rb") as fh:
        assert fh.getnframes() == 0
