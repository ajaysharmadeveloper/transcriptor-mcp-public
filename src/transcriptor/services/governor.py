"""ResourceGovernor — samples the machine and publishes a Pressure.

Two consumers, cleanly separated: the AdmissionController asks whether a job may
*begin*, and the pipeline chunk boundary asks how to *behave once running*.
Conflating them is how systems end up starting a job they immediately abort.

The governor holds the latest sample so tool handlers never sample inline —
``transcript_health`` reads what this published (rule N4).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from dataclasses import dataclass

from transcriptor.domain.pressure import (
    Pressure,
    ResourceSample,
    Thresholds,
    accepting_jobs,
    classify,
    may_load_model,
)
from transcriptor.ports.monitor import ResourceMonitor

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """What the governor last saw.  Cheap to read, never stale by more than the
    poll interval."""

    sample: ResourceSample
    pressure: Pressure
    at: dt.datetime
    elevated_since: dt.datetime | None = None

    @property
    def accepting_jobs(self) -> bool:
        return accepting_jobs(self.pressure)


class ResourceGovernor:
    def __init__(
        self,
        monitor: ResourceMonitor,
        *,
        thresholds: Thresholds | None = None,
        poll_s: float = 5.0,
    ) -> None:
        self._monitor = monitor
        self._thresholds = thresholds or Thresholds()
        self._poll_s = poll_s
        self._elevated_since: dt.datetime | None = None
        self._snapshot: Snapshot | None = None
        self.refresh()

    @property
    def thresholds(self) -> Thresholds:
        return self._thresholds

    @property
    def snapshot(self) -> Snapshot:
        """The most recent reading.  Never None — one is taken at construction
        so health can answer before the poll loop has run."""
        assert self._snapshot is not None
        return self._snapshot

    @property
    def pressure(self) -> Pressure:
        return self.snapshot.pressure

    def refresh(self) -> Snapshot:
        """Take a reading now.  Synchronous and microseconds, so it is safe
        anywhere — but only the monitor task should call it in a loop."""
        sample = self._monitor.sample()
        pressure = classify(sample, self._thresholds)
        now = sample.at or dt.datetime.now(dt.UTC)

        if pressure is Pressure.NORMAL:
            self._elevated_since = None
        elif self._elevated_since is None:
            self._elevated_since = now

        self._snapshot = Snapshot(
            sample=sample,
            pressure=pressure,
            at=now,
            elevated_since=self._elevated_since,
        )
        return self._snapshot

    def elevated_for_s(self) -> float:
        """How long pressure has been at or above ``elevated``.

        Sustained elevation for 30s is what stops new jobs being accepted, while
        the running one continues.
        """
        snap = self.snapshot
        if snap.elevated_since is None:
            return 0.0
        return (snap.at - snap.elevated_since).total_seconds()

    def may_load_model(self, model_peak_gb: float, vram_peak_gb: float = 0.0) -> bool:
        return may_load_model(
            self.snapshot.sample,
            model_peak_gb,
            self._thresholds,
            vram_peak_gb=vram_peak_gb,
        )

    async def run(self) -> None:
        """Poll forever.  Supervised — see ``server.supervise``."""
        while True:
            self.refresh()
            await asyncio.sleep(self._poll_s)
