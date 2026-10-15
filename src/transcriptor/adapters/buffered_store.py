"""BufferedTranscriptStore — survive a database outage mid-job.

A 40-minute transcription must not lose everything to a ``docker compose
restart``.  Failed per-chunk commits buffer in memory and flush on reconnect; if
the buffer fills first the job fails with ``database_lost`` and
``resumable = true``, committed chunks intact.

A decorator rather than logic inside :class:`SqliteTranscriptStore`, so the
retry policy is one small testable thing and the SQL stays about SQL.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from transcriptor.domain.errors import DatabaseLost
from transcriptor.domain.transcript import Chunk, Format, Segment

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Pending:
    job_id: uuid.UUID
    chunk: Chunk
    segments: list[Segment]


class BufferedTranscriptStore:
    """Implements :class:`transcriptor.ports.stores.TranscriptStore`.

    Only ``commit_chunk`` buffers.  Reads are not retried: a read that fails
    during an outage should surface as an error rather than silently return a
    stale or empty transcript.
    """

    def __init__(self, inner, *, max_segments: int = 500) -> None:
        self._inner = inner
        self._max_segments = max_segments
        self._pending: list[_Pending] = []

    @property
    def buffered_segments(self) -> int:
        return sum(len(p.segments) for p in self._pending)

    @property
    def buffered_chunks(self) -> int:
        return len(self._pending)

    async def commit_chunk(
        self, job_id: uuid.UUID, chunk: Chunk, segments: list[Segment]
    ) -> None:
        self._pending.append(_Pending(job_id, chunk, list(segments)))

        if self.buffered_segments > self._max_segments:
            held = self.buffered_segments
            self._pending.clear()
            raise DatabaseLost(
                f"the database stayed unreachable and the segment buffer filled "
                f"({held} segments) — the chunks committed before the outage are "
                f"intact, so this job can be resumed",
                buffered_segments=held,
            )

        await self._flush()

    async def _flush(self) -> None:
        """Write everything held, oldest first.

        Stops at the first failure so ordering is preserved: a later chunk must
        never land before an earlier one, or ``covered_until_s`` would claim
        coverage that has gaps.
        """
        while self._pending:
            item = self._pending[0]
            try:
                await self._inner.commit_chunk(item.job_id, item.chunk, item.segments)
            except (SQLAlchemyError, DBAPIError, OSError) as exc:
                log.warning(
                    "commit failed, buffering",
                    extra={
                        "job_id": str(item.job_id),
                        "chunk": item.chunk.idx,
                        "buffered_segments": self.buffered_segments,
                        "error": str(exc),
                    },
                )
                return
            self._pending.pop(0)
            if not self._pending:
                return
            log.info("flushed a buffered chunk", extra={"remaining": len(self._pending)})

    def __getattr__(self, name: str):
        """Everything except ``commit_chunk`` passes straight through.

        Delegated rather than listed: an explicit passthrough list means every
        method added to the inner store silently breaks this decorator, which is
        exactly what happened when `copy_from` was added and a real job failed
        with AttributeError. Interception is the exception here, so the code
        should say that rather than enumerate the rule.
        """
        return getattr(self._inner, name)
