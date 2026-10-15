"""SubtitleRenderer — formatting, injected rather than imported.

The clip service needs an SRT rebased to each clip's own clock.  Rendering
lives in ``adapters``, so the service depends on this interface and
``server.py`` supplies the implementation.  Reaching sideways for
``adapters.renderers`` would work and is exactly what the dependency rule
forbids — the import contract caught it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from transcriptor.domain.transcript import Segment


@runtime_checkable
class SubtitleRenderer(Protocol):
    # Sync — pure string building, microseconds.
    def clip_srt(self, segments: list[Segment], start: float, end: float) -> str:
        """An SRT covering ``[start, end)``, with times rebased to zero."""
        ...
