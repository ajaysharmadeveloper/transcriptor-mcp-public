"""Phase 3: subprocess cancellation and scratch pruning."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import pytest

from transcriptor.adapters.ffmpeg_audio import FfmpegAudio
from transcriptor.domain.errors import Cancelled
from transcriptor.domain.job import CancelToken, JobStatus


# ---------------------------------------------------- subprocess cancellation


def test_a_callback_registered_after_cancelling_fires_at_once() -> None:
    token = CancelToken()
    token.cancel()
    fired: list[str] = []
    token.on_cancel(lambda: fired.append("now"))
    assert fired == ["now"]


def test_one_failing_callback_does_not_stop_the_others() -> None:
    token = CancelToken()
    fired: list[str] = []

    def explode() -> None:
        raise RuntimeError("this killer is broken")

    token.on_cancel(explode)
    token.on_cancel(lambda: fired.append("still ran"))
    token.cancel()
    assert fired == ["still ran"]


def test_a_finished_process_is_forgotten() -> None:
    """A killer left registered would fire against a process that has already
    exited, or worse against a recycled pid."""
    token = CancelToken()
    fired: list[str] = []
    killer = lambda: fired.append("kill")  # noqa: E731
    token.on_cancel(killer)
    token.forget(killer)
    token.cancel()
    assert fired == []


@pytest.mark.slow
async def test_cancelling_kills_a_running_ffmpeg(tmp_path, executors) -> None:
    """Unlike inference, a child process does not have to run to completion.

    `-re` makes ffmpeg consume input at real time, so this command would take an
    hour. The cancel must land in a fraction of a second.
    """
    _, cpu = executors
    extractor = FfmpegAudio(cpu_executor=cpu)
    token = CancelToken()

    async def cancel_soon() -> None:
        await asyncio.sleep(0.3)
        token.cancel("user changed their mind")

    started = time.monotonic()
    canceller = asyncio.create_task(cancel_soon())
    with pytest.raises(Cancelled):
        await extractor._run(
            token, "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-re", "-f", "lavfi", "-t", "3600", "-i", "sine=frequency=440",
            "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
            str(tmp_path / "out.wav"),
        )
    elapsed = time.monotonic() - started
    await canceller

    assert elapsed < 3.0, f"the kill took {elapsed:.1f}s — it is not immediate"


@pytest.mark.slow
async def test_extraction_passes_its_cancel_token_to_the_subprocess(
    tmp_path, executors, wav
) -> None:
    """Cancelling before the work starts must stop it, not decode anyway."""
    _, cpu = executors
    token = CancelToken()
    token.cancel("already cancelled")

    with pytest.raises(Cancelled):
        await FfmpegAudio(cpu_executor=cpu).to_wav16k(
            wav, tmp_path / "out.wav", cancel=token
        )


# ------------------------------------------------------------------ pruning


@pytest.mark.db
async def test_pruning_removes_scratch_for_a_finished_job(app, long_wav) -> None:
    created = await app.transcription.create(str(long_wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    from transcriptor.services.stages.extract import ExtractStage
    from transcriptor.services.stages.resolve import ResolveStage

    await ResolveStage().run(ctx)
    await ctx.refresh()
    await ExtractStage().run(ctx)
    assert ctx.job_scratch.exists()

    await app.jobs.transition(ctx.job_id, JobStatus.CANCELLED)
    result = await app.pruner.run_once()

    assert result.removed == 1
    assert result.freed_bytes > 0
    assert not ctx.job_scratch.exists()


@pytest.mark.db
async def test_pruning_never_touches_a_running_job(app, long_wav) -> None:
    """A four-hour transcription is older than a short TTL long before it is
    finished. Deleting its audio mid-job would be a self-inflicted failure."""
    created = await app.transcription.create(str(long_wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    from transcriptor.services.stages.extract import ExtractStage
    from transcriptor.services.stages.resolve import ResolveStage

    await ResolveStage().run(ctx)
    await ctx.refresh()
    await ExtractStage().run(ctx)

    result = await app.pruner.run_once()  # max_age_days=0 in the fixture
    assert result.removed == 0
    assert result.kept_active == 1
    assert ctx.job_scratch.exists()


@pytest.mark.db
async def test_pruning_leaves_directories_it_does_not_own(app) -> None:
    """Something that is not a job directory was not put there by us."""
    app.scratch.mkdir(parents=True, exist_ok=True)
    stranger = app.scratch / "not-a-job-id"
    stranger.mkdir()
    (stranger / "keep.txt").write_text("someone else's file")

    await app.pruner.run_once()
    assert stranger.exists()


@pytest.mark.db
async def test_pruning_removes_scratch_whose_job_is_gone(app) -> None:
    """The case normal operation misses: a process killed between the wipe on
    completion and the wipe at startup."""
    app.scratch.mkdir(parents=True, exist_ok=True)
    orphan = app.scratch / str(uuid.uuid4())
    orphan.mkdir()
    (orphan / "audio.wav").write_bytes(b"x" * 2048)

    result = await app.pruner.run_once()
    assert result.removed == 1
    assert not orphan.exists()
