"""FakeDetector — a SpeechDetector for tests that are not about VAD.

Most of the suite uses synthetic tones, which a real detector correctly reports
as silence.  Running Silero there would make every pipeline test fail with
``no_speech_detected`` for reasons that have nothing to do with what they are
checking, so they get a detector that says "it is all speech".

Real VAD behaviour is tested against real audio in ``test_vad.py``.
"""

from __future__ import annotations

import wave
from pathlib import Path

from transcriptor.domain.transcript import Timeline


class FakeDetector:
    """Implements :class:`transcriptor.ports.detector.SpeechDetector`."""

    def __init__(self, *, timeline: Timeline | None = None) -> None:
        self.name = "fake-detector"
        self._timeline = timeline
        self.calls: list[Path] = []

    def speech_regions(self, wav: Path) -> Timeline:
        self.calls.append(wav)
        if self._timeline is not None:
            return self._timeline
        with wave.open(str(wav), "rb") as fh:
            duration = fh.getnframes() / float(fh.getframerate())
        return Timeline.whole(duration)
