"""HealthService and the degraded-task registry.

Not in the architecture §3 module map, which describes what ``transcript_health``
reports (§10) without saying where it is assembled.  It sits in ``services``
because it is application logic over ports.

The memory numbers come from the governor's most recent sample rather than being
measured inline, which is what keeps the tool handler a database-and-memory read
(rule N4).  The sample is at most one poll interval old — real measurements,
not estimates.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any

from transcriptor.ports.engine import ASREngine
from transcriptor.ports.stores import JobStore
from transcriptor.services.governor import ResourceGovernor
from transcriptor.services.reaper import Reaper

log = logging.getLogger(__name__)


@dataclass
class TaskHealth:
    """Which supervised tasks are currently unwell.

    An unsupervised task that dies takes a guarantee with it — a dead heartbeat
    task means the reaper starts killing healthy jobs — so the supervisor
    records crashes here and ``transcript_health`` surfaces them.  A silently
    dead reaper is exactly the kind of failure that looks fine for a week and
    then does not.
    """

    degraded: dict[str, str] = field(default_factory=dict)
    restarts: dict[str, int] = field(default_factory=dict)

    def mark_degraded(self, task: str, reason: str) -> None:
        self.degraded[task] = reason
        self.restarts[task] = self.restarts.get(task, 0) + 1

    def mark_healthy(self, task: str) -> None:
        self.degraded.pop(task, None)

    @property
    def names(self) -> list[str]:
        return sorted(self.degraded)


class HealthService:
    def __init__(
        self,
        *,
        jobs: JobStore,
        engine: ASREngine,
        governor: ResourceGovernor,
        tasks: TaskHealth,
        reaper: Reaper | None = None,
        runner=None,
        model: str = "",
    ) -> None:
        self._jobs = jobs
        self._engine = engine
        self._governor = governor
        self._tasks = tasks
        self._reaper = reaper
        self._runner = runner
        self._model = model

    async def report(self) -> dict[str, Any]:
        snapshot = self._governor.snapshot
        sample = snapshot.sample
        thresholds = self._governor.thresholds

        database_ok = True
        queue_depth: int | None = None
        jobs_active: int | None = None
        try:
            queue_depth = await self._jobs.queue_depth()
            jobs_active = await self._jobs.active_count()
        except Exception as exc:  # noqa: BLE001
            # Health must answer even when the database does not — saying so is
            # the whole point of the tool.
            log.warning("health could not reach the database", extra={"error": str(exc)})
            database_ok = False

        report: dict[str, Any] = {
            "model": self._model or self._engine.name,
            "model_loaded": self._engine.loaded,
            # Which engine and device, because "why is this so slow" is
            # otherwise unanswerable: the same model on Metal, CUDA and a CPU
            # differ by an order of magnitude. Duck-typed so an engine that
            # predates these attributes still reports.
            "engine": getattr(self._engine, "kind", "unknown"),
            "device": getattr(self._engine, "device", None),
            "self_rss_gb": round(sample.rss_gb, 2),
            "self_cap_gb": thresholds.self_cap_gb,
            "system_available_gb": round(sample.available_gb, 2),
            "system_total_gb": round(sample.total_gb, 2),
            "system_floor_gb": thresholds.system_floor_gb,
            "swap_used_gb": round(sample.swap_used_gb, 2),
            "swap_delta_gb": round(sample.swap_delta_gb, 2),
            "pressure": snapshot.pressure.value,
            "sampled_at": snapshot.at.isoformat() if snapshot.at else None,
            "queue_depth": queue_depth,
            "jobs_active": jobs_active,
            "accepting_jobs": snapshot.accepting_jobs and database_ok,
            "queued_reason": (
                self._runner.hold_reason if self._runner is not None else None
            ),
            "database_reachable": database_ok,
            "degraded_tasks": self._tasks.names,
            "task_restarts": dict(self._tasks.restarts),
            "reaper_last_run_at": (
                self._reaper.last_run_at.isoformat()
                if self._reaper and self._reaper.last_run_at
                else None
            ),
        }

        # Only where there is a discrete GPU. Omitted rather than reported as
        # zero on unified memory, where the GPU already spends the memory
        # `system_available_gb` accounts for — a "vram: 0GB" line there would
        # read as a machine out of graphics memory rather than one that has no
        # separate pool to report.
        if sample.has_vram:
            report["vram_used_gb"] = round(sample.vram_used_gb, 2)
            report["vram_free_gb"] = round(sample.vram_free_gb, 2)
            report["vram_total_gb"] = round(sample.vram_total_gb, 2)
            report["vram_floor_gb"] = thresholds.vram_floor_gb

        return report
