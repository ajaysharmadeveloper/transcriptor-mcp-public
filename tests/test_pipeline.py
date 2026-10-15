"""End-to-end through the real pipeline, with FakeEngine.

This is the payoff of the ASREngine port: the resolve → extract → transcribe →
finalize path, the runner, the stages and the checkpoints all run for real,
against a real Postgres and real ffmpeg, with no MLX and no model weights.
"""

from __future__ import annotations

import datetime as dt

import threading
import uuid

import pytest
from sqlalchemy import text

from transcriptor.domain.errors import DurationExceedsLimit, NoSpeechDetected
from transcriptor.domain.job import JobStatus
from transcriptor.domain.transcript import ChunkStatus, Format

pytestmark = pytest.mark.db


async def _run_one(app) -> JobStatus:
    """Drive exactly one runner pass, rather than racing a background task."""
    return await app.runner.tick()


# ------------------------------------------------------------------- success


async def test_a_local_file_transcribes_end_to_end(app, wav) -> None:
    created = await app.transcription.create(str(wav))
    assert created.status == "queued"
    assert created.reused is False

    assert await _run_one(app) is JobStatus.DONE

    status = await app.reads.status(created.job_id)
    assert status.status == "done"
    assert status.progress_pct == 100.0
    assert status.word_count and status.word_count > 0
    assert status.duration_s == pytest.approx(3.0, abs=0.2)


async def test_srt_is_produced_from_the_transcript(app, wav) -> None:
    created = await app.transcription.create(str(wav))
    await _run_one(app)

    view = await app.reads.get(created.job_id, fmt="srt", mode="full")
    assert view.status == "done"
    assert view.partial is False
    # FakeEngine emits one segment per second over a 3s file.
    assert view.content.startswith("1\n00:00:00,000 --> 00:00:01,000\n")
    assert view.content.count(" --> ") == 3


async def test_txt_preview_truncates_but_reports_the_true_size(app, wav) -> None:
    created = await app.transcription.create(str(wav))
    await _run_one(app)

    full = await app.reads.get(created.job_id, fmt="txt", mode="full")
    preview = await app.reads.get(created.job_id, fmt="txt", max_chars=5)

    assert preview.truncated is True
    assert len(preview.content) == 5
    assert preview.total_chars == full.total_chars
    assert full.truncated is False


async def test_the_checkpoints_each_stage_promises_are_written(app, wav) -> None:
    created = await app.transcription.create(str(wav))
    await _run_one(app)

    job = await app.jobs.get(uuid.UUID(created.job_id))
    source = await app.jobs.get_source(job.source_id)

    assert job.duration_s is not None, "Resolve checkpoints duration_s"
    assert source["audio_hash"], "Extract checkpoints audio_hash"
    chunks = await app.transcripts.chunks(job.id)
    assert chunks and all(c.status is ChunkStatus.DONE for c in chunks), "Transcribe"
    assert job.word_count is not None, "Finalize checkpoints word_count"


async def test_scratch_is_removed_when_the_job_ends(app, wav) -> None:
    """Nothing durable lives on disk (invariant I8)."""
    created = await app.transcription.create(str(wav))
    await _run_one(app)
    assert not (app.scratch / created.job_id).exists()


async def test_job_events_record_the_whole_history(app, wav) -> None:
    created = await app.transcription.create(str(wav))
    await _run_one(app)

    events = await app.jobs.events(uuid.UUID(created.job_id))
    assert [e.to_status for e in events][0] is JobStatus.QUEUED
    assert events[-1].to_status is JobStatus.DONE
    stages = [e.stage.value for e in events if e.stage]
    assert stages == [
        "resolving",    # claimed
        "resolving",    # Resolve
        "extracting",   # Extract  (Fetch is satisfied for a local file)
        "vad",          # Detect
        "transcribing", # Transcribe
        "rendering",    # Punctuate
        "rendering",    # Finalize
    ]


# ------------------------------------------------------- concurrency contract


async def test_only_one_thread_ever_calls_transcribe(app, wav) -> None:
    """Invariant I5, asserted by identity rather than by inspecting the pool."""
    for _ in range(3):
        created = await app.transcription.create(str(wav), force=True)
        await _run_one(app)

    assert len(app.engine.calling_threads) == 1
    assert app.engine.calling_threads != {threading.get_ident()}, (
        "inference must not run on the event loop thread"
    )


async def test_transcribe_never_runs_on_the_event_loop(app, wav) -> None:
    import asyncio

    loop_thread = threading.get_ident()
    created = await app.transcription.create(str(wav))
    await _run_one(app)
    assert loop_thread not in app.engine.calling_threads


# ---------------------------------------------------------------- resumption


async def test_a_crashed_job_resumes_on_the_same_row(app, wav, db) -> None:
    """The full crash-recovery path: heartbeat goes stale, the reaper releases
    the job, and the next request resumes it instead of attaching to a corpse."""
    created = await app.transcription.create(str(wav))
    job_id = uuid.UUID(created.job_id)

    await app.jobs.next_queued()  # claimed, now "running"
    async with db.begin() as session:
        await session.execute(
            text("UPDATE jobs SET heartbeat_at = :stale WHERE id = :id"),
            {
                "stale": (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=5)).isoformat(),
                "id": job_id.hex,
            },
        )

    assert await app.reaper.run_once() == [job_id]
    assert (await app.reads.status(created.job_id)).status == "interrupted"

    again = await app.transcription.create(str(wav))
    assert again.job_id == created.job_id, "resume reuses the row"
    assert again.reuse_reason == "resumed"

    assert await _run_one(app) is JobStatus.DONE


async def test_which_stages_a_finished_job_reports_satisfied(app, wav) -> None:
    """Resume is not a special code path — it is this satisfied/skip loop.

    After a completed run: resolve, transcribe and finalize are satisfied, so a
    resumed job would skip them. Extract and Detect are deliberately *not*,
    because scratch is wiped on completion and at startup.
    """
    created = await app.transcription.create(str(wav))
    await _run_one(app)
    calls = len(app.engine.transcribe_calls)

    job = await app.jobs.get(uuid.UUID(created.job_id))
    ctx = await app.context_factory(job)

    satisfied = {s.name: await s.satisfied(ctx) for s in app.pipeline.stages}
    assert satisfied == {
        "resolve": True,
        # A local file needs no fetching — the source is the media.
        "fetch": True,
        # Extract and Detect both need their scratch audio, which is wiped on
        # completion and at startup. Re-decoding a few minutes of audio beats
        # redoing hours of inference.
        "extract": False,
        "detect": False,
        "transcribe": True,
        # Punctuate shares Finalize's checkpoint; it is idempotent, so a re-run
        # costs time rather than correctness.
        "punctuate": True,
        "finalize": True,
    }
    assert len(app.engine.transcribe_calls) == calls, "probing must not transcribe"


# -------------------------------------------------------------------- errors


async def test_audio_over_the_phase_1_cap_is_refused(app, wav, monkeypatch) -> None:
    from transcriptor.services.pipeline import JobContext

    created = await app.transcription.create(str(wav))
    original = app.context_factory

    async def short_cap(job):
        ctx = await original(job)
        ctx.max_duration_s = 1  # the 3s fixture now exceeds the cap
        return ctx

    short_cap.engine_memory_gb = 0.0
    app.runner._context_factory = short_cap

    assert await _run_one(app) is JobStatus.FAILED

    status = await app.reads.status(created.job_id)
    assert status.error_code == "duration_exceeds_limit"
    assert status.resumable is False, "a too-long file will not get shorter on retry"


async def test_a_missing_file_fails_before_any_work(app, tmp_path) -> None:
    from transcriptor.domain.errors import SourceNotFound

    with pytest.raises(SourceNotFound) as excinfo:
        await app.transcription.create(str(tmp_path / "gone.mp4"))
    assert excinfo.value.code == "source_not_found"


async def test_a_relative_path_is_refused(app) -> None:
    from transcriptor.domain.errors import RelativePathNotAllowed

    with pytest.raises(RelativePathNotAllowed):
        await app.transcription.create("videos/ep12.mp4")


async def test_an_engine_failure_is_recorded_with_its_code(app, wav) -> None:
    from transcriptor.domain.errors import EngineError

    app.engine._fail_with = EngineError("model exploded", engine="fake")
    created = await app.transcription.create(str(wav))

    assert await _run_one(app) is JobStatus.FAILED
    status = await app.reads.status(created.job_id)
    assert status.error_code == "engine_error"
    assert status.resumable is False


async def test_an_unexpected_exception_becomes_internal_error(app, wav) -> None:
    """Invariant I10: every failure path sets error_code and resumable."""
    app.engine._fail_with = RuntimeError("something nobody predicted")
    created = await app.transcription.create(str(wav))

    assert await _run_one(app) is JobStatus.FAILED
    status = await app.reads.status(created.job_id)
    assert status.error_code == "internal_error"
    assert status.resumable is False


async def test_silence_is_reported_not_hallucinated(app, wav) -> None:
    """An empty result must say so rather than inventing a transcript."""
    app.engine._phrases = ("",)
    app.engine._segment_s = 10.0  # one empty segment for the whole file
    created = await app.transcription.create(str(wav))

    assert await _run_one(app) is JobStatus.FAILED
    status = await app.reads.status(created.job_id)
    assert status.error_code == "no_speech_detected"


# ------------------------------------------------------------------ runner


async def test_the_runner_does_nothing_when_the_queue_is_empty(app) -> None:
    assert await _run_one(app) is None


async def test_admission_holds_the_queue_under_critical_pressure(
    app, wav, governor
) -> None:
    created = await app.transcription.create(str(wav))
    governor._monitor.gb = 0.5  # below the floor
    governor.refresh()

    assert await _run_one(app) is None, "no job may start under critical pressure"
    assert (await app.reads.status(created.job_id)).status == "queued"

    governor._monitor.gb = 12.0
    governor.refresh()
    assert await _run_one(app) is JobStatus.DONE, "and it starts once pressure clears"


async def test_heartbeat_only_beats_while_a_job_is_running(app) -> None:
    assert await app.heartbeat.beat() is False


# ------------------------------------------------------- a held queue is loud


async def test_a_job_held_by_admission_says_why(app, wav, governor) -> None:
    """A job that sits `queued` with no explanation looks exactly like a broken
    runner. project-details §9 and §14 both require a queued_reason."""
    created = await app.transcription.create(str(wav))

    governor._monitor.gb = 0.5  # below the floor
    governor.refresh()
    assert await _run_one(app) is None

    status = await app.reads.status(created.job_id)
    assert status.status == "queued"
    assert status.queued_reason == "memory_pressure"

    report = await app.health.report()
    assert report["accepting_jobs"] is False
    assert report["queued_reason"] == "memory_pressure"


async def test_the_hold_reason_clears_once_the_job_starts(app, wav, governor) -> None:
    created = await app.transcription.create(str(wav))
    governor._monitor.gb = 0.5
    governor.refresh()
    await _run_one(app)
    assert app.runner.hold_reason == "memory_pressure"

    governor._monitor.gb = 12.0
    governor.refresh()
    assert await _run_one(app) is JobStatus.DONE
    assert app.runner.hold_reason is None
    assert (await app.reads.status(created.job_id)).queued_reason is None


async def test_a_model_that_will_not_fit_is_named_as_the_reason(
    app, wav, governor
) -> None:
    """Distinct from memory_pressure: the machine is calm, the model simply does
    not fit alongside the floor."""
    created = await app.transcription.create(str(wav))
    app.runner._context_factory.engine_memory_gb = 10.0  # needs 10 + 3 floor

    assert await _run_one(app) is None
    assert (
        await app.reads.status(created.job_id)
    ).queued_reason == "insufficient_memory"
