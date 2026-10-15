"""SileroDetector — speech regions via Silero VAD.

VAD is mandatory, not an optimisation.  Whisper hallucinates confident invented
sentences over silence and drifts on long-form audio; feeding it only the speech
is what stops a music bed becoming a paragraph of plausible nonsense.

Runs in the inference executor alongside MLX — same single thread, same reason:
these are the two things in the process that are neither fast nor interruptible.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

from transcriptor.domain.errors import InfraError
from transcriptor.domain.transcript import Timeline

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

# Silero's own defaults are tuned for chopping audio into utterances.  These are
# tuned for the opposite job: keep speech intact, drop only real silence.
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_MS = 250
DEFAULT_MIN_SILENCE_MS = 300
# Padding either side of each region, so a detector that clips a word's first
# phoneme does not cost the word.
DEFAULT_PAD_MS = 120
# Gaps shorter than this are closed. Cutting at every micro-pause would hand
# Whisper a stream of fragments with no context, which is how VAD makes a
# transcript worse rather than better.
DEFAULT_MERGE_GAP_S = 0.6


class SileroDetector:
    """Implements :class:`transcriptor.ports.detector.SpeechDetector`."""

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        min_speech_ms: int = DEFAULT_MIN_SPEECH_MS,
        min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
        pad_ms: int = DEFAULT_PAD_MS,
        merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    ) -> None:
        self.name = "silero-vad"
        self._threshold = threshold
        self._min_speech_ms = min_speech_ms
        self._min_silence_ms = min_silence_ms
        self._pad_ms = pad_ms
        self._merge_gap_s = merge_gap_s
        self._model = None

    def load(self) -> None:
        """~2MB, loaded once and kept.  Synchronous: the only caller is the
        inference executor."""
        if self._model is not None:
            return
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise InfraError(
                "silero-vad is not installed — run: uv sync --extra vad",
                detector="silero-vad",
            ) from exc
        self._model = load_silero_vad()

    # Sync — runs in the inference executor.
    def speech_regions(self, wav: Path) -> Timeline:
        """Speech regions on the original clock, non-speech dropped."""
        from silero_vad import get_speech_timestamps

        self.load()
        audio = _read_wav(wav)

        stamps = get_speech_timestamps(
            audio,
            self._model,
            sampling_rate=SAMPLE_RATE,
            threshold=self._threshold,
            min_speech_duration_ms=self._min_speech_ms,
            min_silence_duration_ms=self._min_silence_ms,
            speech_pad_ms=self._pad_ms,
            return_seconds=True,
        )

        timeline = Timeline(
            [(float(s["start"]), float(s["end"])) for s in stamps]
        ).merged(gap_s=self._merge_gap_s)

        log.info(
            "vad complete",
            extra={
                "regions": len(timeline.regions),
                "speech_s": round(timeline.speech_seconds, 2),
                "audio_s": round(len(audio) / SAMPLE_RATE, 2),
            },
        )
        return timeline


def _read_wav(path: Path):
    """Load 16kHz mono PCM into a float32 tensor in [-1, 1].

    Deliberately not ``silero_vad.read_audio``: that goes through torchaudio,
    which from 2.9 requires a separate ``torchcodec`` package for audio I/O.
    The extract stage always produces 16kHz mono signed-16 PCM, so reading it
    here is a few lines and removes a whole dependency chain.
    """
    import torch

    with wave.open(str(path), "rb") as fh:
        if fh.getnchannels() != 1 or fh.getsampwidth() != 2:
            raise InfraError(
                f"expected 16-bit mono PCM, got {fh.getnchannels()} channel(s) "
                f"at {fh.getsampwidth() * 8}-bit",
                path=str(path),
            )
        if fh.getframerate() != SAMPLE_RATE:
            raise InfraError(
                f"expected {SAMPLE_RATE}Hz, got {fh.getframerate()}Hz",
                path=str(path),
            )
        frames = fh.readframes(fh.getnframes())

    samples = torch.frombuffer(bytearray(frames), dtype=torch.int16)
    return samples.float() / 32768.0
