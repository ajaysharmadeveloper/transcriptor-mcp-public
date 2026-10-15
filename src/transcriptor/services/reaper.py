"""Reaper — stale-heartbeat detection.

If the process dies mid-transcription its job sits in ``running`` forever, and
every future request for that source attaches to a job that will never finish.
That single failure would undo the whole dedupe design, so the reaper runs at
startup *and* every 60s.

Startup matters most: after a crash, the first thing to do is release orphaned
jobs so the next request can resume them rather than hanging off a dead one.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import uuid

from transcriptor.ports.stores import JobStore

log = logging.getLogger(__name__)


class Reaper:
    def __init__(
        self, jobs: JobStore, *, stale_after_s: int = 90, interval_s: float = 60.0
    ) -> None:
        self._jobs = jobs
        self._stale_after_s = stale_after_s
        self._interval_s = interval_s
        self._last_run_at: dt.datetime | None = None

    @property
    def last_run_at(self) -> dt.datetime | None:
        """Reported by ``transcript_health``.  A silently dead reaper is exactly
        the kind of failure that looks fine for a week and then does not."""
        return self._last_run_at

    async def run_once(self) -> list[uuid.UUID]:
        reaped = await self._jobs.reap_stale(self._stale_after_s)
        self._last_run_at = dt.datetime.now(dt.UTC)
        if reaped:
            log.warning(
                "released orphaned jobs",
                extra={"count": len(reaped), "job_ids": [str(j) for j in reaped]},
            )
        return reaped

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            await self.run_once()
