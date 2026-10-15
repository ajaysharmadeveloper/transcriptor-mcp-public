"""ScratchPruner — stop the disk filling with abandoned in-flight audio.

Scratch is wiped when a job ends and again at startup, so in normal operation
nothing accumulates.  This exists for what normal operation misses: a process
killed between those two moments, or a long-running server that never restarts.

Only directories whose job is *no longer active* are removed.  Age alone is not
enough — a four-hour transcription is older than a short TTL long before it is
finished, and deleting its audio mid-job would be a self-inflicted failure.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from transcriptor.ports.files import FileWriter
from transcriptor.ports.stores import JobStore

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PruneResult:
    removed: int
    freed_bytes: int
    kept_active: int


class ScratchPruner:
    def __init__(
        self,
        *,
        jobs: JobStore,
        files: FileWriter,
        scratch_dir: Path,
        max_age_days: float = 30.0,
        interval_s: float = 3600.0,
    ) -> None:
        self._jobs = jobs
        self._files = files
        self._scratch_dir = scratch_dir
        self._max_age_s = max_age_days * 86400
        self._interval_s = interval_s
        self._last_run_at: dt.datetime | None = None

    @property
    def last_run_at(self) -> dt.datetime | None:
        return self._last_run_at

    async def run(self) -> None:
        """Poll forever.  Supervised — see ``server.supervise``."""
        while True:
            await asyncio.sleep(self._interval_s)
            await self.run_once()

    async def run_once(self) -> PruneResult:
        self._last_run_at = dt.datetime.now(dt.UTC)
        if not self._scratch_dir.is_dir():
            return PruneResult(0, 0, 0)

        removed = freed = kept = 0
        now = time.time()

        for entry in self._scratch_dir.iterdir():
            if not entry.is_dir():
                continue

            try:
                job_id = uuid.UUID(entry.name)
            except ValueError:
                # Not a job directory at all; leave it alone rather than delete
                # something nobody asked us to manage.
                continue

            job = await self._jobs.get(job_id)
            if job is not None and job.is_active:
                # A four-hour transcription is older than a short TTL long
                # before it is finished. Age alone must never win here.
                kept += 1
                continue

            age = now - entry.stat().st_mtime
            if job is not None and age < self._max_age_s:
                continue

            freed += await self._files.remove_tree(entry)
            removed += 1

        if removed:
            log.info(
                "pruned abandoned scratch",
                extra={"removed": removed, "freed_bytes": freed, "kept_active": kept},
            )
        return PruneResult(removed=removed, freed_bytes=freed, kept_active=kept)
