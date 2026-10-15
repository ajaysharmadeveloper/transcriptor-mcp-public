"""Pressure thresholds, the governor and admission.

All of it against synthetic samples with no machine state at all — which is the
point of keeping the thresholds as pure functions.
"""

from __future__ import annotations

import datetime as dt

import pytest

from transcriptor.domain.pressure import (
    Pressure,
    ResourceSample,
    Thresholds,
    accepting_jobs,
    classify,
    idle_evict_seconds,
    may_load_model,
)
from transcriptor.services.admission import AdmissionController
from transcriptor.services.governor import ResourceGovernor

T = Thresholds()


def sample(**kw) -> ResourceSample:
    base = dict(
        rss_gb=1.0, available_gb=8.0, swap_used_gb=0.0, swap_delta_gb=0.0,
        total_gb=16.0, at=dt.datetime.now(dt.UTC),
    )
    return ResourceSample(**{**base, **kw})


# ----------------------------------------------------------------- classify


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, Pressure.NORMAL),
        ({"available_gb": 4.0}, Pressure.NORMAL),          # floor + 1.5
        ({"available_gb": 3.4}, Pressure.ELEVATED),        # within 1GB of floor
        ({"available_gb": 2.4}, Pressure.CRITICAL),        # below the floor
        ({"swap_delta_gb": 0.25}, Pressure.ELEVATED),      # > 200MB of swap growth
        ({"swap_delta_gb": 1.5}, Pressure.CRITICAL),       # > 1GB
        ({"rss_gb": 3.7}, Pressure.ELEVATED),              # within 10% of the cap
        ({"rss_gb": 4.5}, Pressure.CRITICAL),              # over the self cap
    ],
)
def test_the_three_bands(kwargs: dict, expected: Pressure) -> None:
    assert classify(sample(**kwargs), T) is expected


def test_swap_growth_outranks_a_comfortable_available_reading() -> None:
    """macOS compresses memory and shares it with the GPU, so `available` alone
    flatters the situation. If swap is climbing, the machine is already losing."""
    looks_fine = sample(available_gb=10.0, swap_delta_gb=1.4)
    assert classify(looks_fine, T) is Pressure.CRITICAL


def test_whichever_limit_binds_first_wins() -> None:
    """The self cap stops us being a bad neighbour; the floor responds to what
    the neighbours are doing right now."""
    plenty_free_but_over_cap = sample(available_gb=12.0, rss_gb=5.0)
    assert classify(plenty_free_but_over_cap, T) is Pressure.CRITICAL


def test_the_ladder_is_ordered() -> None:
    assert Pressure.NORMAL < Pressure.ELEVATED < Pressure.CRITICAL


# ------------------------------------------------------------ adaptive rules


@pytest.mark.parametrize(
    ("pressure", "seconds"),
    [(Pressure.NORMAL, 600), (Pressure.ELEVATED, 60), (Pressure.CRITICAL, 0)],
)
def test_idle_eviction_tightens_with_pressure(pressure: Pressure, seconds: int) -> None:
    assert idle_evict_seconds(pressure) == seconds


def test_new_work_starts_only_when_the_machine_is_comfortable() -> None:
    assert accepting_jobs(Pressure.NORMAL) is True
    assert accepting_jobs(Pressure.ELEVATED) is False
    assert accepting_jobs(Pressure.CRITICAL) is False


def test_a_model_load_must_leave_the_floor_intact() -> None:
    """Checked before the load, because a refused load is recoverable and an
    OOM is not."""
    assert may_load_model(sample(available_gb=8.0), 2.5, T) is True
    # The boundary is exactly model peak + floor: 2.5 + 2.5 = 5.0.
    assert may_load_model(sample(available_gb=5.0), 2.5, T) is True
    assert may_load_model(sample(available_gb=4.9), 2.5, T) is False
    # fp16 exceeds the self cap outright, whatever the machine has free.
    assert may_load_model(sample(available_gb=15.0), 6.5, T) is False


# -------------------------------------------------------------- the governor


class FrozenMonitor:
    def __init__(self, s: ResourceSample) -> None:
        self.current = s

    def sample(self) -> ResourceSample:
        return self.current


def test_the_governor_publishes_a_snapshot_before_its_loop_runs() -> None:
    """Health has to answer immediately, so a reading is taken at construction."""
    g = ResourceGovernor(FrozenMonitor(sample()))
    assert g.snapshot is not None
    assert g.pressure is Pressure.NORMAL
    assert g.snapshot.accepting_jobs is True


def test_the_governor_tracks_how_long_pressure_has_been_elevated() -> None:
    now = dt.datetime.now(dt.UTC)
    monitor = FrozenMonitor(sample(available_gb=3.0, at=now))
    g = ResourceGovernor(monitor)
    assert g.pressure is Pressure.ELEVATED
    assert g.elevated_for_s() == 0.0

    monitor.current = sample(available_gb=3.0, at=now + dt.timedelta(seconds=45))
    g.refresh()
    assert g.elevated_for_s() == pytest.approx(45.0)


def test_returning_to_normal_clears_the_elevated_clock() -> None:
    now = dt.datetime.now(dt.UTC)
    monitor = FrozenMonitor(sample(available_gb=3.0, at=now))
    g = ResourceGovernor(monitor)

    monitor.current = sample(available_gb=9.0, at=now + dt.timedelta(seconds=45))
    g.refresh()
    assert g.pressure is Pressure.NORMAL
    assert g.elevated_for_s() == 0.0


# ------------------------------------------------------------- the admission


def test_admission_lets_a_job_start_on_a_quiet_machine() -> None:
    a = AdmissionController()
    assert a.may_start(
        pressure=Pressure.NORMAL, sample=sample(), active_jobs=0, engine_memory_gb=2.5
    )


def test_admission_refuses_a_second_concurrent_job() -> None:
    """Two simultaneous jobs on 16GB means swapping."""
    a = AdmissionController(max_concurrent=1)
    decision = a.may_start(
        pressure=Pressure.NORMAL, sample=sample(), active_jobs=1, engine_memory_gb=2.5
    )
    assert not decision
    assert decision.reason == "max_concurrent"


def test_admission_refuses_under_critical_pressure() -> None:
    a = AdmissionController()
    decision = a.may_start(
        pressure=Pressure.CRITICAL, sample=sample(available_gb=2.0),
        active_jobs=0, engine_memory_gb=2.5,
    )
    assert decision.reason == "memory_pressure"


def test_a_passing_spike_does_not_stall_the_queue() -> None:
    """Elevated is tolerated briefly — but sustained elevation stops new work
    while letting whatever is running finish."""
    a = AdmissionController(elevated_grace_s=30.0)
    # Elevated by swap growth, with room to spare — so this isolates the grace
    # window rather than tripping the memory check.
    spiking = sample(available_gb=9.0, swap_delta_gb=0.3)
    assert classify(spiking, T) is Pressure.ELEVATED

    brief = a.may_start(
        pressure=Pressure.ELEVATED, sample=spiking,
        active_jobs=0, engine_memory_gb=2.5, elevated_for_s=5.0,
    )
    assert brief

    sustained = a.may_start(
        pressure=Pressure.ELEVATED, sample=spiking,
        active_jobs=0, engine_memory_gb=2.5, elevated_for_s=31.0,
    )
    assert not sustained
    assert sustained.reason == "memory_backpressure"


def test_admission_refuses_when_the_model_would_not_fit() -> None:
    a = AdmissionController()
    decision = a.may_start(
        pressure=Pressure.NORMAL, sample=sample(available_gb=4.0),
        active_jobs=0, engine_memory_gb=2.5,   # needs 2.5 + 2.5 = 5.0
    )
    assert decision.reason == "insufficient_memory"


def test_admission_is_pure() -> None:
    """It never mutates anything, which is what makes it trivially testable and
    keeps 'may we begin' separate from 'how do we behave once running'."""
    a = AdmissionController()
    args = dict(
        pressure=Pressure.NORMAL, sample=sample(), active_jobs=0, engine_memory_gb=2.5
    )
    assert a.may_start(**args) == a.may_start(**args)


# ------------------------------------------------- occupancy, honestly measured


def test_the_monitor_prefers_the_allocator_when_rss_under_reports() -> None:
    """RSS does not reliably account for MLX's Metal buffers: loading large-v3
    8-bit moved RSS by 0.79GB while MLX reported 1.60GB allocated. A self cap
    enforced on RSS alone would let the process grow past it unnoticed."""
    from transcriptor.adapters.psutil_monitor import PsutilResourceMonitor

    monitor = PsutilResourceMonitor(allocator_gb=lambda: 3.5)
    assert monitor.sample().rss_gb == pytest.approx(3.5, abs=0.01)


def test_the_larger_of_the_two_is_taken_not_the_sum() -> None:
    """MLX's buffers are partly visible in RSS already, so adding them would
    double-count and refuse jobs that would have fitted."""
    from transcriptor.adapters.psutil_monitor import PsutilResourceMonitor

    naive = PsutilResourceMonitor().sample().rss_gb
    aware = PsutilResourceMonitor(allocator_gb=lambda: 0.0).sample().rss_gb
    assert aware == pytest.approx(naive, abs=0.05)


def test_a_failing_allocator_reading_never_breaks_the_sample() -> None:
    """The monitor runs in a supervised loop every 5s; a measurement that raises
    would take the pressure signal down with it."""
    from transcriptor.adapters.psutil_monitor import PsutilResourceMonitor

    def broken() -> float:
        raise RuntimeError("mlx is not loaded")

    sample = PsutilResourceMonitor(allocator_gb=broken).sample()
    assert sample.rss_gb > 0


def test_the_defaults_reflect_what_was_measured() -> None:
    """Peak occupancy 2.2GB with zero swap, so the cap has headroom and the
    floor is not costing capacity it does not buy back in safety."""
    t = Thresholds()
    assert t.self_cap_gb == 4.0
    assert t.system_floor_gb == 2.5

    # large-v3 8-bit fits; fp16 is refused by the cap whatever the machine has.
    assert may_load_model(sample(available_gb=5.0), 2.5, t) is True
    assert may_load_model(sample(available_gb=15.0), 6.5, t) is False


def test_the_new_floor_admits_the_machine_the_old_one_refused() -> None:
    """5.03GB available was the real reading that blocked a job which then ran
    with a 2.2GB peak and no swap."""
    from transcriptor.services.admission import AdmissionController

    observed = sample(available_gb=5.03)
    assert AdmissionController(thresholds=Thresholds(system_floor_gb=3.0)).may_start(
        pressure=classify(observed, Thresholds(system_floor_gb=3.0)),
        sample=observed, active_jobs=0, engine_memory_gb=2.5,
    ).reason == "insufficient_memory"

    assert AdmissionController().may_start(
        pressure=classify(observed, Thresholds()),
        sample=observed, active_jobs=0, engine_memory_gb=2.5,
    )


# ------------------------------------------------------------- discrete GPU
#
# VRAM is a second pool that psutil cannot see at all. On unified memory the
# GPU spends the memory `available_gb` already measures, so these rules must
# stay completely inert there — every existing test above is the proof that
# they do, since none of them sets a VRAM field.


def gpu_sample(**kw) -> ResourceSample:
    """A machine with a 12GB card, mostly free, and plenty of system RAM."""
    base = dict(vram_used_gb=2.0, vram_free_gb=10.0, vram_total_gb=12.0)
    return sample(**{**base, **kw})


def test_a_machine_with_no_gpu_reports_no_vram() -> None:
    """Zero means "not applicable", never "empty" — a distinction the whole
    design rests on."""
    assert sample().has_vram is False
    assert gpu_sample().has_vram is True


def test_vram_is_ignored_where_there_is_no_discrete_gpu() -> None:
    """The Apple Silicon guarantee, stated as a test: a sample with no VRAM
    fields classifies exactly as it did before VRAM existed."""
    starved = sample(vram_free_gb=0.0, vram_total_gb=0.0)
    assert classify(starved, T) is Pressure.NORMAL


def test_an_exhausted_gpu_is_critical_even_with_system_memory_to_spare() -> None:
    """The failure this exists to catch. 8GB of system memory free, and the
    job about to OOM on a pool the system floor cannot see."""
    starved = gpu_sample(vram_free_gb=0.2, available_gb=8.0)
    assert classify(starved, T) is Pressure.CRITICAL


def test_a_nearly_full_gpu_is_elevated() -> None:
    assert classify(gpu_sample(vram_free_gb=1.2), T) is Pressure.ELEVATED


def test_a_healthy_gpu_stays_normal() -> None:
    assert classify(gpu_sample(), T) is Pressure.NORMAL


def test_loading_is_refused_when_the_gpu_cannot_hold_the_model() -> None:
    """4.5GB of weights will not fit in 3GB of free VRAM, however much system
    memory is going spare."""
    assert may_load_model(gpu_sample(vram_free_gb=3.0), 1.5, T, vram_peak_gb=4.5) is False


def test_loading_is_allowed_when_both_pools_have_room() -> None:
    assert may_load_model(gpu_sample(), 1.5, T, vram_peak_gb=4.5) is True


def test_system_memory_still_binds_on_a_gpu_machine() -> None:
    """Whichever limit binds first wins. A huge GPU does not license using up
    the machine's RAM."""
    cramped = gpu_sample(available_gb=3.0)
    assert may_load_model(cramped, 1.5, T, vram_peak_gb=4.5) is False


def test_a_gpu_that_cannot_be_read_does_not_refuse_work() -> None:
    """Honest degradation: with no NVML and no torch the reading is zeros, and
    the right answer is to govern on system memory alone rather than to refuse
    a job we cannot measure."""
    unreadable = sample(vram_total_gb=0.0, vram_free_gb=0.0)
    assert may_load_model(unreadable, 1.5, T, vram_peak_gb=4.5) is True


def test_a_cpu_engine_is_not_held_to_a_vram_floor_on_a_gpu_machine() -> None:
    """vram_peak_gb=0 means "does not use the GPU", so a CPU build must not be
    refused because something else is using the card."""
    busy_card = gpu_sample(vram_free_gb=0.1)
    assert may_load_model(busy_card, 1.5, T, vram_peak_gb=0.0) is True


def test_admission_refuses_a_job_that_would_not_fit_in_vram() -> None:
    controller = AdmissionController(thresholds=T)
    decision = controller.may_start(
        pressure=Pressure.NORMAL,
        sample=gpu_sample(vram_free_gb=2.0),
        active_jobs=0,
        engine_memory_gb=1.5,
        engine_vram_gb=4.5,
    )
    assert not decision
    assert decision.reason == "insufficient_memory"


def test_admission_is_unchanged_for_engines_that_want_no_vram() -> None:
    controller = AdmissionController(thresholds=T)
    assert controller.may_start(
        pressure=Pressure.NORMAL,
        sample=sample(),
        active_jobs=0,
        engine_memory_gb=2.5,
    )


def test_the_monitor_reports_zeros_when_the_engine_offers_no_reading() -> None:
    from transcriptor.adapters.psutil_monitor import PsutilResourceMonitor

    reading = PsutilResourceMonitor().sample()
    assert reading.has_vram is False


def test_the_monitor_passes_through_a_gpu_reading() -> None:
    from transcriptor.adapters.psutil_monitor import PsutilResourceMonitor

    monitor = PsutilResourceMonitor(vram_gb=lambda: (2.0, 10.0, 12.0))
    reading = monitor.sample()
    assert (reading.vram_used_gb, reading.vram_free_gb, reading.vram_total_gb) == (
        2.0, 10.0, 12.0,
    )


def test_an_exploding_vram_probe_never_breaks_the_sample() -> None:
    """Measurement must never be the thing that fails — the monitor task runs
    every 5 seconds and a raise there takes the governor down with it."""
    from transcriptor.adapters.psutil_monitor import PsutilResourceMonitor

    def boom():
        raise RuntimeError("nvml exploded")

    reading = PsutilResourceMonitor(vram_gb=boom).sample()
    assert reading.has_vram is False
    assert reading.available_gb > 0
