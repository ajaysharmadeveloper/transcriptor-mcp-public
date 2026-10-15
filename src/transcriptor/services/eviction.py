"""ModelEvictor — give the memory back when nothing is using it.

Idle residency is adaptive (project-details §9): 600s when the machine is calm,
60s when it is not, immediately under critical pressure.  Holding 2.5GB for ten
minutes on a quiet machine is fine; holding it while the machine is thrashing is
being a bad neighbour.
"""

from __future__ import annotations

import asyncio
import logging
import time

from transcriptor.domain.pressure import idle_evict_seconds
from transcriptor.ports.engine import ASREngine
from transcriptor.services.governor import ResourceGovernor

log = logging.getLogger(__name__)


class ModelEvictor:
    def __init__(
        self,
        *,
        engine: ASREngine,
        governor: ResourceGovernor,
        runner,
        base_idle_s: int = 600,
        poll_s: float = 10.0,
    ) -> None:
        self._engine = engine
        self._governor = governor
        self._runner = runner
        self._base_idle_s = base_idle_s
        self._poll_s = poll_s
        self._idle_since: float | None = None

    async def run(self) -> None:
        """Poll forever.  Supervised — see ``server.supervise``."""
        while True:
            await asyncio.sleep(self._poll_s)
            await self.tick()

    async def tick(self) -> bool:
        """One pass.  Returns whether the model was evicted."""
        if not self._engine.loaded:
            self._idle_since = None
            return False

        if self._runner.current_job_id is not None:
            # Busy: the clock only starts once nothing is using it.
            self._idle_since = None
            return False

        now = time.monotonic()
        if self._idle_since is None:
            self._idle_since = now

        limit = idle_evict_seconds(self._governor.pressure, self._base_idle_s)
        idle_for = now - self._idle_since
        if idle_for < limit:
            return False

        log.info(
            "evicting the idle model",
            extra={
                "idle_s": round(idle_for, 1),
                "limit_s": limit,
                "pressure": self._governor.pressure.value,
            },
        )
        await self._engine.unload()
        self._idle_since = None
        return True
