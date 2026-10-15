"""ResourceMonitor and Clock.

``sample()`` is deliberately synchronous: it takes microseconds, so making it
async would add an await point for no reason and invite the impression that it
is expensive.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, runtime_checkable

from transcriptor.domain.pressure import ResourceSample


@runtime_checkable
class ResourceMonitor(Protocol):
    def sample(self) -> ResourceSample:
        """One reading: RSS, available memory, swap used and swap growth."""
        ...


@runtime_checkable
class Clock(Protocol):
    """Injected so time-dependent policy can be tested without waiting."""

    def now(self) -> dt.datetime: ...

    async def sleep(self, seconds: float) -> None: ...
