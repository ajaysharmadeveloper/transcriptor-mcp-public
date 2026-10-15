"""AdmissionController — may a job start now?

A pure function of pressure, how many jobs are already running, and the engine's
memory estimate.  It never mutates anything, which makes it trivially testable
and keeps the "should we begin" decision separate from the "how do we behave
once running" ladder at the chunk boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from transcriptor.domain.pressure import Pressure, ResourceSample, Thresholds, may_load_model


@dataclass(frozen=True, slots=True)
class Admission:
    """The decision, with the reason attached.

    ``reason`` becomes ``queued_reason`` on the job, so a caller can see *why*
    their job has not started rather than watching a silent queue.
    """

    admitted: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.admitted


ADMITTED = Admission(True)


class AdmissionController:
    def __init__(
        self,
        *,
        thresholds: Thresholds | None = None,
        max_concurrent: int = 1,
        elevated_grace_s: float = 30.0,
    ) -> None:
        self._thresholds = thresholds or Thresholds()
        self._max_concurrent = max_concurrent
        self._elevated_grace_s = elevated_grace_s

    def may_start(
        self,
        *,
        pressure: Pressure,
        sample: ResourceSample,
        active_jobs: int,
        engine_memory_gb: float,
        elevated_for_s: float = 0.0,
        # What the engine wants from a discrete GPU. Zero for MLX and for any
        # CPU build, where there is no second pool to run out of.
        engine_vram_gb: float = 0.0,
    ) -> Admission:
        """Pure.  Same inputs, same answer, no side effects."""
        if active_jobs >= self._max_concurrent:
            return Admission(False, "max_concurrent")

        if pressure is Pressure.CRITICAL:
            return Admission(False, "memory_pressure")

        # Elevated is tolerated briefly — a passing spike should not stall the
        # queue — but sustained elevation means stop accepting new work while
        # letting whatever is running finish.
        if pressure is Pressure.ELEVATED and elevated_for_s >= self._elevated_grace_s:
            return Admission(False, "memory_backpressure")

        if not may_load_model(
            sample, engine_memory_gb, self._thresholds, vram_peak_gb=engine_vram_gb
        ):
            return Admission(False, "insufficient_memory")

        return ADMITTED
