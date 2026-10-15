"""SpeechDetector.

Whisper hallucinates confident invented sentences over silence and drifts on
long-form audio, so VAD is a required stage rather than an optimisation.
Phase 2 implements it; the port is declared now so the pipeline shape does not
have to change later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from transcriptor.domain.transcript import Timeline


@runtime_checkable
class SpeechDetector(Protocol):
    # Sync — runs in the inference executor alongside the engine.
    def speech_regions(self, wav: Path) -> Timeline:
        """Speech regions on the original timeline, non-speech dropped."""
        ...
