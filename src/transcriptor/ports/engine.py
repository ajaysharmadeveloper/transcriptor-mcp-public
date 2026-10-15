"""ASREngine.

The port exists even with one implementation: it is what makes swapping the
model a config change rather than a rewrite (driver D7), and what lets the whole
application layer test against a fake with no MLX, no 1.8GB model, and no Apple
Silicon.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from transcriptor.domain.transcript import AudioSlice, Segment


@runtime_checkable
class ASREngine(Protocol):
    """Speech to timestamped segments.

    Note which members are ``async def`` and which are plain ``def``.  That
    distinction *is* the concurrency design.
    """

    name: str
    memory_estimate_gb: float

    @property
    def loaded(self) -> bool: ...

    async def load(self) -> None:
        """Bring weights into memory.  Async because it is IO-bound and the
        caller wants to stay responsive while it happens."""
        ...

    async def unload(self) -> None:
        """Evict the model.  Called on idle timeout and immediately on abort."""
        ...

    # SYNCHRONOUS on purpose.  This is CPU/GPU-bound and runs in the inference
    # executor; making it async would invite calling it on the event loop, which
    # is the single most likely bug in this project.  Its only legal caller is
    # ThreadPoolExecutor(max_workers=1) — see invariant I5.
    def transcribe(
        self,
        audio: AudioSlice,
        language: str | None,
        *,
        prompt: str | None = None,
    ) -> list[Segment]:
        """Transcribe one slice.  Blocks for as long as it takes.

        ``prompt`` is the tail of the text already committed before this slice.
        Whisper starts each chunk cold without it and loses casing and
        punctuation for the rest of the chunk — measured, not assumed:

            no prompt:   "looks exactly like the agent working properly i still"
            with prompt: "And looks exactly like the agent working properly. I"

        Keyword-only with a default, so it extends the architecture §4 signature
        rather than breaking callers.  Owner decision on 2026-09-09.

        The detected language is published on :attr:`detected_language` rather
        than returned, keeping the return type as the doc specifies.  That is
        safe here precisely because exactly one thread ever calls this.
        """
        ...

    @property
    def detected_language(self) -> str | None:
        """Language identified by the most recent :meth:`transcribe` call."""
        ...
