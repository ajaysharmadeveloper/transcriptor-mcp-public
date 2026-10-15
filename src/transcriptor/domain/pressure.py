"""Memory pressure: thresholds as pure functions.

Transcriptor is a guest on this machine, not its owner.  A static budget table
(macOS 6GB, Docker 2GB, us 2.5GB) is a snapshot of one moment, not a plan —
Chrome grows, Docker restarts, a video export starts.  So the system measures
and gives up capacity rather than dragging the machine into swap.

Everything here is pure, so the whole escalation ladder is testable against
synthetic samples with no machine state at all.
"""

from __future__ import annotations

import datetime as dt
import enum
from dataclasses import dataclass


class Pressure(str, enum.Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _RANK[self]

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, Pressure):
            return NotImplemented
        return self.rank < other.rank


_RANK = {Pressure.NORMAL: 0, Pressure.ELEVATED: 1, Pressure.CRITICAL: 2}


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """One reading.  Cheap enough to take on the event loop (microseconds).

    ``swap_delta_gb`` is growth since the process started, and it is the signal
    that matters most: macOS compresses memory and shares it with the GPU, so
    ``available`` alone flatters the situation.  If swap is climbing while a job
    runs, the machine is already losing whatever ``available`` says.

    The ``vram_*`` fields describe a **discrete** GPU's own memory, and they are
    zero on every machine that does not have one.  On Apple Silicon they stay
    zero on purpose: unified memory means the GPU allocates out of the same pool
    ``available_gb`` already measures, so a second reading there would
    double-count the one thing the floor is protecting.  On a machine with a
    separate GPU the opposite is true — VRAM is invisible to the process table,
    so the system floor can read perfectly healthy while the GPU is out of
    memory.  Zero therefore means "not applicable", never "empty", and every
    rule below is written to skip rather than to assume.
    """

    rss_gb: float
    available_gb: float
    swap_used_gb: float
    swap_delta_gb: float
    total_gb: float = 0.0
    at: dt.datetime | None = None
    # Discrete GPU memory. Zero when there is no discrete GPU to speak of.
    vram_used_gb: float = 0.0
    vram_free_gb: float = 0.0
    vram_total_gb: float = 0.0

    @property
    def has_vram(self) -> bool:
        """Whether a discrete GPU is present and reporting.

        The one place the "zero means not applicable" convention is spelled
        out, so no rule has to repeat the reasoning.
        """
        return self.vram_total_gb > 0.0


@dataclass(frozen=True, slots=True)
class Thresholds:
    """The two limits, plus the margins that separate the three bands.

    Whichever binds first wins: the self cap stops Transcriptor being a bad
    neighbour, the system floor responds to what the neighbours are doing now.

    Both defaults are measured rather than guessed (BENCHMARKS.md, 2026-09-09,
    Apple M4 / 16GB):

    * peak process occupancy transcribing with large-v3 8-bit: **3.04GB**, so a
      4.0GB self cap leaves ~24% headroom and crosses the elevated threshold
      (90% of cap, 3.6GB) before the cap itself — the ladder starts clearing
      MLX's buffer cache before anything is at risk. fp16 at ~6.5GB is
      correctly refused by the cap on any machine.
    * *available* memory consumed by a job: **~1.95GB** — lower than occupancy,
      because part of the model is shared with the page cache. This is the
      figure the floor is compared against.
    * swap growth across every benchmark: **zero**, with available bottoming at
      3.66GB against a 2.5GB floor.

    Occupancy and availability-cost are different numbers and are checked
    against different limits. Conflating them either refuses jobs that fit or
    lets the process grow past its cap.

    The floor is deliberately **fixed rather than a proportion of total RAM**.
    A proportional rule looks tidier and behaves worse at both ends: 20% of
    64GB reserves 12GB for nothing, and 20% of 8GB reserves too little to
    matter. What the floor protects — room for the OS and the apps already
    open — does not scale with how much memory the machine happens to have.
    """

    self_cap_gb: float = 4.0
    # 3.0 was the original guess. It refused to start a job on a 16GB machine
    # with 5.03GB available, which then ran consuming ~1.95GB of availability
    # and no swap at all — a guess costing capacity it was not buying safety
    # with. At 2.5 the same run bottomed out at 3.66GB available, 1.16GB clear.
    system_floor_gb: float = 2.5
    # `elevated` starts within this much of the floor.
    elevated_headroom_gb: float = 1.0
    elevated_swap_delta_gb: float = 0.2
    critical_swap_delta_gb: float = 1.0
    # RSS this close to the cap is already worth reacting to.
    elevated_cap_fraction: float = 0.9

    # -- discrete GPU ------------------------------------------------------
    # Free VRAM to leave for everything else — the desktop compositor, another
    # CUDA process, the browser's hardware decoder. The same "guest on this
    # machine" reasoning as the system floor, applied to the other pool.
    #
    # UNMEASURED. The floor above is a benchmark result; this is a considered
    # default, chosen because 0.5GB is roughly what a desktop session needs to
    # keep compositing and is small enough not to refuse work on an 8GB card.
    # Anyone with a CUDA machine and a spare afternoon should replace it with a
    # number, and say so in BENCHMARKS.md when they do.
    vram_floor_gb: float = 0.5
    elevated_vram_headroom_gb: float = 1.0


def classify(sample: ResourceSample, t: Thresholds = Thresholds()) -> Pressure:
    """Map a reading onto the three bands.

    critical  available < floor, or swap grew > 1GB, or RSS is over the cap,
              or (discrete GPU only) free VRAM is under the VRAM floor
    elevated  available within 1GB of the floor, or swap grew > 200MB,
              or RSS is within 10% of the cap, or free VRAM is within 1GB of
              the VRAM floor
    normal    everything else

    The VRAM clauses apply only where a discrete GPU reported itself, so this
    function returns exactly what it always did on Apple Silicon and on any
    CPU-only machine.
    """
    if (
        sample.available_gb < t.system_floor_gb
        or sample.swap_delta_gb > t.critical_swap_delta_gb
        or sample.rss_gb > t.self_cap_gb
        or (sample.has_vram and sample.vram_free_gb < t.vram_floor_gb)
    ):
        return Pressure.CRITICAL

    if (
        sample.available_gb < t.system_floor_gb + t.elevated_headroom_gb
        or sample.swap_delta_gb > t.elevated_swap_delta_gb
        or sample.rss_gb > t.self_cap_gb * t.elevated_cap_fraction
        or (
            sample.has_vram
            and sample.vram_free_gb < t.vram_floor_gb + t.elevated_vram_headroom_gb
        )
    ):
        return Pressure.ELEVATED

    return Pressure.NORMAL


def may_load_model(
    sample: ResourceSample,
    model_peak_gb: float,
    t: Thresholds = Thresholds(),
    *,
    vram_peak_gb: float = 0.0,
) -> bool:
    """Would loading the model leave both floors intact and stay under the cap?

    Checked before the load rather than discovered after it, because a refused
    load is recoverable and an OOM is not.

    ``vram_peak_gb`` is what the model will want from a *discrete* GPU, and it
    is keyword-only with a zero default so every existing caller keeps its
    meaning.  Where the weights live on the GPU the two figures are different
    questions, not one: a CUDA build wants ~4.5GB of VRAM and comparatively
    little system memory, and checking either alone admits a job the other
    would have refused.
    """
    if model_peak_gb > t.self_cap_gb:
        return False
    if sample.available_gb < model_peak_gb + t.system_floor_gb:
        return False
    if vram_peak_gb > 0.0 and sample.has_vram:
        return sample.vram_free_gb >= vram_peak_gb + t.vram_floor_gb
    return True


def idle_evict_seconds(pressure: Pressure, base_s: int = 600) -> int:
    """How long an idle model may stay resident.

    Adaptive, from project-details §9: 600s normal, 60s elevated, immediate
    under critical.
    """
    if pressure is Pressure.CRITICAL:
        return 0
    if pressure is Pressure.ELEVATED:
        return min(base_s, 60)
    return base_s


def accepting_jobs(pressure: Pressure) -> bool:
    """New work starts only when the machine is comfortable.

    A job admitted under pressure is one that has to be aborted moments later.
    """
    return pressure is Pressure.NORMAL
