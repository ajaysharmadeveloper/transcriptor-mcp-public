"""JobRunner — the single worker task, and the heartbeat that proves it alive.

Deliberately boring.  All the judgement lives in ``admission`` and ``pipeline``;
the loop itself has nothing to get wrong.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid

from transcriptor.domain.job import Job, JobStatus
from transcriptor.ports.stores import JobStore
from transcriptor.services.admission import Admission, AdmissionController
from transcriptor.services.governor import ResourceGovernor
from transcriptor.services.pipeline import JobContext, Pipeline

log = logging.getLogger(__name__)


class JobRunner:
    """One, and only one, of these runs."""

    def __init__(
        self,
        *,
        jobs: JobStore,
        pipeline: Pipeline,
        admission: AdmissionController,
        governor: ResourceGovernor,
        context_factory,
        wake: asyncio.Event,
        poll_s: float = 2.0,
        keep_scratch: bool = False,
    ) -> None:
        self._jobs = jobs
        self._pipeline = pipeline
        self._admission = admission
        self._governor = governor
        self._context_factory = context_factory
        self._wake = wake
        self._poll_s = poll_s
        self._keep_scratch = keep_scratch
        self._current: uuid.UUID | None = None
        self._ctx: JobContext | None = None
        self._last_hold: Admission | None = None

    @property
    def current_job_id(self) -> uuid.UUID | None:
        """What the heartbeat task should be touching, if anything."""
        return self._current

    @property
    def hold_reason(self) -> str | None:
        """Why the queue is not moving, or None if nothing is holding it.

        A job held by admission would otherwise sit `queued` with no
        explanation, which looks identical to a broken runner.  This is what
        `queued_reason` reports.
        """
        return self._last_hold.reason if self._last_hold is not None else None

    async def run(self) -> None:
        """Poll forever.  Supervised — see ``server.supervise``."""
        while True:
            await self._wait()
            await self.tick()

    async def tick(self) -> JobStatus | None:
        """One pass.  Separated from the loop so tests drive it directly rather
        than racing a background task."""
        snapshot = self._governor.snapshot
        decision = self._admission.may_start(
            pressure=snapshot.pressure,
            sample=snapshot.sample,
            active_jobs=await self._jobs.active_count(),
            engine_memory_gb=self._engine_memory_gb(),
            engine_vram_gb=self._engine_vram_gb(),
            elevated_for_s=self._governor.elevated_for_s(),
        )
        if not decision:
            self._last_hold = decision
            log.info("admission held the queue", extra={"reason": decision.reason})
            return None
        self._last_hold = None

        job = await self._jobs.next_queued()
        if job is None:
            return None

        return await self._execute(job)

    def request_cancel(self, job_id: uuid.UUID) -> bool:
        """Signal the job in flight, if it is this one.

        Returns whether it was signalled, so the caller knows whether to stop a
        queued job itself instead.
        """
        if self._current != job_id or self._ctx is None:
            return False
        self._ctx.cancel.cancel("cancelled by request")
        return True

    async def _execute(self, job: Job) -> JobStatus:
        self._current = job.id
        ctx: JobContext | None = None
        try:
            ctx = await self._context_factory(job)
            self._ctx = ctx
            return await self._pipeline.execute(ctx)
        finally:
            self._current = None
            self._ctx = None
            if ctx is not None:
                Pipeline.cleanup(ctx, keep=self._keep_scratch)

    async def _wait(self) -> None:
        """Wake on an insert, or fall through on the poll interval.

        The Event handles the common case, the poll handles anything the Event
        missed — which is what keeps "the queue is the database" from needing
        LISTEN/NOTIFY for a single process.
        """
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._wake.wait(), timeout=self._poll_s)
        self._wake.clear()

    def _engine_memory_gb(self) -> float:
        return getattr(self._context_factory, "engine_memory_gb", 0.0)

    def _engine_vram_gb(self) -> float:
        """Zero unless the engine runs on a discrete GPU, in which case the
        admission check has a second floor to respect."""
        return getattr(self._context_factory, "engine_vram_gb", 0.0)


class Heartbeat:
    """Touches ``heartbeat_at`` for whatever the runner is working on.

    Its own task, because a heartbeat driven from inside the pipeline would stop
    exactly when a stage blocks — which is when it matters most.  A dead
    heartbeat task means the reaper starts killing healthy jobs, so it is
    supervised like everything else.
    """

    def __init__(self, *, jobs: JobStore, runner: JobRunner, interval_s: float = 10.0):
        self._jobs = jobs
        self._runner = runner
        self._interval_s = interval_s

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            await self.beat()

    async def beat(self) -> bool:
        job_id = self._runner.current_job_id
        if job_id is None:
            return False
        await self._jobs.heartbeat(job_id)
        return True
