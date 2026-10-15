"""SpeakerDiarizer — who spoke when.

Declared, not implemented.  The port exists so the pipeline shape does not have
to change when a diarizer is chosen.

Sync, like the other model-backed ports: it runs in the inference executor
beside MLX and Silero, for the same reason they do.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """One continuous stretch of one voice, on the original clock."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@runtime_checkable
class SpeakerDiarizer(Protocol):
    name: str
    memory_estimate_gb: float

    # Sync — the inference executor is the only legal caller, as with ASREngine.
    def turns(self, wav: Path, *, max_speakers: int | None = None) -> list[SpeakerTurn]:
        """Speaker turns for the whole recording.

        Returned on the *original* clock, not the speech-only one: a caller
        assigning speakers to segments works in the timeline it was given.
        """
        ...
