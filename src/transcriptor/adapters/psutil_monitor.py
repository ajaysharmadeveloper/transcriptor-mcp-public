"""PsutilResourceMonitor.

Reads both numbers that matter: ``available``, and swap growth since start.
``available`` alone flatters the situation on macOS, which compresses memory and
shares it with the GPU — so swap growth is the truer signal.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import psutil

from transcriptor.domain.pressure import ResourceSample

_GB = 1024**3


class PsutilResourceMonitor:
    """Implements :class:`transcriptor.ports.monitor.ResourceMonitor`.

    The swap baseline is taken at construction, so ``swap_delta_gb`` measures
    growth caused while this process has been running rather than whatever the
    machine had swapped out beforehand.
    """

    def __init__(
        self,
        *,
        allocator_gb: Callable[[], float] | None = None,
        vram_gb: Callable[[], tuple[float, float, float] | None] | None = None,
    ) -> None:
        self._process = psutil.Process()
        self._swap_baseline_gb = psutil.swap_memory().used / _GB
        # RSS does not reliably account for MLX's Metal buffers: on this machine
        # loading large-v3 8-bit moved RSS by 0.79GB while MLX reported 1.60GB
        # allocated. Enforcing a self cap on RSS alone would therefore let the
        # process grow well past it unnoticed, so the engine supplies its own
        # allocator reading and it is added in.
        #
        # Injected rather than imported: MLX stays behind the engine adapter.
        self._allocator_gb = allocator_gb
        # And the same again for a discrete GPU, whose memory psutil cannot see
        # at all. Without this the system floor reads healthy while the GPU is
        # out of memory — on unified memory those are one number, on a machine
        # with a separate card they are two. Injected for the same reason:
        # CUDA belongs behind the engine adapter, not in here.
        self._vram_gb = vram_gb

    @property
    def swap_baseline_gb(self) -> float:
        return self._swap_baseline_gb

    def reset_baseline(self) -> None:
        """Re-zero swap growth, e.g. after the model is evicted."""
        self._swap_baseline_gb = psutil.swap_memory().used / _GB

    def _occupancy_gb(self) -> float:
        """What this process actually occupies: RSS, or the allocator's own
        figure where that is larger.

        Not a sum — MLX's buffers are partly visible in RSS already, so adding
        them would double-count. The larger of the two is the honest floor on
        real occupancy.
        """
        rss = self._process.memory_info().rss / _GB
        if self._allocator_gb is None:
            return rss
        try:
            allocator = self._allocator_gb()
        except Exception:  # noqa: BLE001 - measurement must never break the loop
            return rss
        return max(rss, allocator)

    def _vram(self) -> tuple[float, float, float]:
        """``(used, free, total)`` in GB, or zeros where there is no discrete
        GPU — which is also what an unreadable one reports.

        Zeros mean "not applicable" throughout ``domain/pressure``, so a GPU
        that cannot be read degrades to governing on system memory alone
        rather than refusing work it cannot measure.
        """
        if self._vram_gb is None:
            return (0.0, 0.0, 0.0)
        try:
            reading = self._vram_gb()
        except Exception:  # noqa: BLE001 - measurement must never break the loop
            return (0.0, 0.0, 0.0)
        return reading or (0.0, 0.0, 0.0)

    def sample(self) -> ResourceSample:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        swap_used_gb = swap.used / _GB
        vram_used, vram_free, vram_total = self._vram()
        return ResourceSample(
            rss_gb=self._occupancy_gb(),
            available_gb=vm.available / _GB,
            swap_used_gb=swap_used_gb,
            # Clamped at zero: swap shrinking is not negative pressure.
            swap_delta_gb=max(0.0, swap_used_gb - self._swap_baseline_gb),
            total_gb=vm.total / _GB,
            at=dt.datetime.now(dt.UTC),
            vram_used_gb=vram_used,
            vram_free_gb=vram_free,
            vram_total_gb=vram_total,
        )
