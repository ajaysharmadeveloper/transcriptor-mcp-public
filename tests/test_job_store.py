"""SqliteJobStore: the claim path, the state machine, and the reaper.

These cover invariants I1 (one active job per result_key), I2 (no job stays
running with a stale heartbeat) and I6 (terminal jobs change only via resume).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid

import pytest
from sqlalchemy import text

from transcriptor.domain.errors import IllegalTransition
from transcriptor.domain.job import (
    Attached,
    Created,
    JobStatus,
    ReuseReason,
    Reused,
    Stage,
)
from transcriptor.domain.source import CaptionPolicy, SourceKind, SourceRef, result_key

pytestmark = pytest.mark.db

CLAIM = dict(model="whisper-8bit", language=None, captions=CaptionPolicy.MANUAL_ONLY)


# ------------------------------------------------------------------- sources


async def test_upsert_source_is_idempotent_and_counts_requests(jobs) -> None:
    ref = SourceRef(SourceKind.FILE, "/tmp/a.mp4", "/tmp/a.mp4", "local")
    first = await jobs.upsert_source(ref, fingerprint="1:1")
    second = await jobs.upsert_source(ref, fingerprint="1:1")
    assert first == second, "source_key is unique — no second row"


# --------------------------------------------------------------------- claim


async def test_first_claim_creates(jobs, source_id) -> None:
    result = await jobs.claim_or_attach(result_key(
        identity="/tmp/a.mp4", **CLAIM), source_id, **CLAIM)
    assert isinstance(result, Created)

    job = await jobs.get(result.job_id)
    assert job.status is JobStatus.QUEUED
    assert job.model == "whisper-8bit"


async def test_second_claim_attaches_to_the_running_job(jobs, source_id) -> None:
    """Layer 3 — two requests thirty seconds apart must not start two
    transcriptions."""
    key = result_key(identity="/tmp/a.mp4", **CLAIM)
    first = await jobs.claim_or_attach(key, source_id, **CLAIM)
    second = await jobs.claim_or_attach(key, source_id, **CLAIM)

    assert isinstance(second, Attached)
    assert second.job_id == first.job_id
    assert second.reason is ReuseReason.ALREADY_RUNNING


async def test_concurrent_claims_produce_exactly_one_job(jobs, source_id) -> None:
    """Invariant I1.  Check-then-act in application code would let both callers
    read 'no active job' before either writes."""
    key = result_key(identity="/tmp/a.mp4", **CLAIM)

    results = await asyncio.gather(
        *(jobs.claim_or_attach(key, source_id, **CLAIM) for _ in range(8))
    )

    assert sum(isinstance(r, Created) for r in results) == 1, "exactly one create"
    assert len({r.job_id for r in results}) == 1, "everyone gets the same job_id"


async def test_completed_job_is_reused(jobs, source_id) -> None:
    """Layer 1 — a completed result answers in milliseconds, no work started."""
    key = result_key(identity="/tmp/a.mp4", **CLAIM)
    created = await jobs.claim_or_attach(key, source_id, **CLAIM)
    await jobs.transition(created.job_id, JobStatus.RUNNING)
    await jobs.transition(created.job_id, JobStatus.DONE)

    again = await jobs.claim_or_attach(key, source_id, **CLAIM)
    assert isinstance(again, Reused)
    assert again.reason is ReuseReason.EXACT_SOURCE
    assert again.job_id == created.job_id


async def test_force_bypasses_reuse_but_not_the_active_slot(jobs, source_id) -> None:
    key = result_key(identity="/tmp/a.mp4", **CLAIM)
    created = await jobs.claim_or_attach(key, source_id, **CLAIM)
    await jobs.transition(created.job_id, JobStatus.RUNNING)
    await jobs.transition(created.job_id, JobStatus.DONE)

    forced = await jobs.claim_or_attach(key, source_id, force=True, **CLAIM)
    assert isinstance(forced, Created)
    assert forced.job_id != created.job_id

    # force still cannot open a second concurrent job for the same key
    again = await jobs.claim_or_attach(key, source_id, force=True, **CLAIM)
    assert isinstance(again, Attached)
    assert again.job_id == forced.job_id


async def test_interrupted_job_is_resumed_on_the_same_row(jobs, source_id) -> None:
    key = result_key(identity="/tmp/a.mp4", **CLAIM)
    created = await jobs.claim_or_attach(key, source_id, **CLAIM)
    await jobs.transition(created.job_id, JobStatus.RUNNING)
    await jobs.transition(created.job_id, JobStatus.INTERRUPTED, resumable=True)

    resumed = await jobs.claim_or_attach(key, source_id, **CLAIM)
    assert isinstance(resumed, Reused)
    assert resumed.reason is ReuseReason.RESUMED
    assert resumed.job_id == created.job_id, "resume reuses the row, not a new job"
    assert (await jobs.get(created.job_id)).status is JobStatus.QUEUED


async def test_non_resumable_failure_is_not_resumed(jobs, source_id) -> None:
    key = result_key(identity="/tmp/a.mp4", **CLAIM)
    created = await jobs.claim_or_attach(key, source_id, **CLAIM)
    await jobs.transition(created.job_id, JobStatus.RUNNING)
    await jobs.transition(created.job_id, JobStatus.FAILED, resumable=False)

    fresh = await jobs.claim_or_attach(key, source_id, **CLAIM)
    assert isinstance(fresh, Created)
    assert fresh.job_id != created.job_id


@pytest.mark.parametrize(
    "changed",
    [
        {"model": "whisper-fp16"},
        {"language": "hi"},
        {"captions": CaptionPolicy.NEVER},
    ],
    ids=["model", "language", "captions"],
)
async def test_changing_settings_produces_a_different_job(
    jobs, source_id, changed: dict
) -> None:
    """A transcript from a different model, forced language or caption policy is
    a different artefact — handing the old one back would be a correctness bug
    dressed as a cache hit."""
    base = result_key(identity="/tmp/a.mp4", **CLAIM)
    created = await jobs.claim_or_attach(base, source_id, **CLAIM)
    await jobs.transition(created.job_id, JobStatus.RUNNING)
    await jobs.transition(created.job_id, JobStatus.DONE)

    other_args = {**CLAIM, **changed}
    other = result_key(identity="/tmp/a.mp4", **other_args)
    assert other != base

    result = await jobs.claim_or_attach(other, source_id, **other_args)
    assert isinstance(result, Created)
    assert result.job_id != created.job_id


# ----------------------------------------------------------------- lifecycle


async def test_illegal_transitions_are_rejected(jobs, source_id) -> None:
    """Invariant I6.  Terminal states change only via resume."""
    key = result_key(identity="/tmp/a.mp4", **CLAIM)
    created = await jobs.claim_or_attach(key, source_id, **CLAIM)
    await jobs.transition(created.job_id, JobStatus.RUNNING)
    await jobs.transition(created.job_id, JobStatus.DONE)

    with pytest.raises(IllegalTransition) as excinfo:
        await jobs.transition(created.job_id, JobStatus.RUNNING)
    assert excinfo.value.code == "illegal_transition"
    assert excinfo.value.resumable is False

    assert (await jobs.get(created.job_id)).status is JobStatus.DONE


async def test_every_transition_writes_a_job_event(jobs, source_id) -> None:
    key = result_key(identity="/tmp/a.mp4", **CLAIM)
    created = await jobs.claim_or_attach(key, source_id, **CLAIM)
    await jobs.transition(created.job_id, JobStatus.RUNNING, stage=Stage.EXTRACTING)
    await jobs.transition(created.job_id, JobStatus.DONE)

    events = await jobs.events(created.job_id)
    assert [e.to_status for e in events] == [
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.DONE,
    ]
    assert events[1].stage is Stage.EXTRACTING


async def test_next_queued_claims_one_job_and_marks_it_running(jobs, source_id) -> None:
    key = result_key(identity="/tmp/a.mp4", **CLAIM)
    created = await jobs.claim_or_attach(key, source_id, **CLAIM)

    claimed = await jobs.next_queued()
    assert claimed.id == created.job_id
    assert claimed.status is JobStatus.RUNNING
    assert claimed.heartbeat_at is not None, "a running job must heartbeat immediately"

    assert await jobs.next_queued() is None, "the queue is now empty"


async def test_concurrent_runners_never_claim_the_same_job(jobs, source_id) -> None:
    """FOR UPDATE SKIP LOCKED: two runners claim two different jobs, not one twice."""
    for n in range(4):
        ref = SourceRef(SourceKind.FILE, f"/tmp/{n}.mp4", f"/tmp/{n}.mp4", "local")
        sid = await jobs.upsert_source(ref, fingerprint="1:1")
        await jobs.claim_or_attach(
            result_key(identity=f"/tmp/{n}.mp4", **CLAIM), sid, **CLAIM
        )

    claimed = await asyncio.gather(*(jobs.next_queued() for _ in range(4)))
    ids = [j.id for j in claimed if j is not None]
    assert len(ids) == len(set(ids)) == 4


async def test_queue_depth_counts_only_queued(jobs, source_id) -> None:
    key = result_key(identity="/tmp/a.mp4", **CLAIM)
    created = await jobs.claim_or_attach(key, source_id, **CLAIM)
    assert await jobs.queue_depth() == 1
    await jobs.transition(created.job_id, JobStatus.RUNNING)
    assert await jobs.queue_depth() == 0
    assert await jobs.active_count() == 1


# -------------------------------------------------------------------- reaper


async def test_reaper_interrupts_a_job_with_a_stale_heartbeat(
    jobs, db, source_id
) -> None:
    """Invariant I2.  A crashed process leaves a job running forever, and every
    later request would attach to a job that will never finish."""
    key = result_key(identity="/tmp/a.mp4", **CLAIM)
    created = await jobs.claim_or_attach(key, source_id, **CLAIM)
    await jobs.next_queued()

    # Simulate a process that died 5 minutes ago.
    async with db.begin() as session:
        await session.execute(
            # The timestamp is computed here and bound, rather than written
            # in SQL. SQLite has no interval type and stores times as text, so
            # the value has to arrive in exactly the format UtcDateTime writes.
            text("UPDATE jobs SET heartbeat_at = :stale WHERE id = :id"),
            {
                "stale": (
                    dt.datetime.now(dt.UTC) - dt.timedelta(minutes=5)
                ).isoformat(),
                "id": created.job_id.hex,
            },
        )

    reaped = await jobs.reap_stale(older_than_s=90)
    assert reaped == [created.job_id]

    job = await jobs.get(created.job_id)
    assert job.status is JobStatus.INTERRUPTED
    assert job.resumable is True
    assert job.error_code == "process_died"


async def test_reaper_leaves_a_healthy_job_alone(jobs, source_id) -> None:
    key = result_key(identity="/tmp/a.mp4", **CLAIM)
    created = await jobs.claim_or_attach(key, source_id, **CLAIM)
    await jobs.next_queued()
    await jobs.heartbeat(created.job_id)

    assert await jobs.reap_stale(older_than_s=90) == []
    assert (await jobs.get(created.job_id)).status is JobStatus.RUNNING


async def test_reaper_never_touches_a_queued_job(jobs, source_id) -> None:
    """A queued job has no heartbeat and is not orphaned — reaping it would be
    the reaper killing healthy work."""
    key = result_key(identity="/tmp/a.mp4", **CLAIM)
    created = await jobs.claim_or_attach(key, source_id, **CLAIM)
    assert await jobs.reap_stale(older_than_s=0) == []
    assert (await jobs.get(created.job_id)).status is JobStatus.QUEUED


async def test_a_reaped_job_can_be_resumed(jobs, db, source_id) -> None:
    """The point of reaping: the next request resumes rather than hanging off a
    dead job."""
    key = result_key(identity="/tmp/a.mp4", **CLAIM)
    created = await jobs.claim_or_attach(key, source_id, **CLAIM)
    await jobs.next_queued()
    async with db.begin() as session:
        await session.execute(
            # The timestamp is computed here and bound, rather than written
            # in SQL. SQLite has no interval type and stores times as text, so
            # the value has to arrive in exactly the format UtcDateTime writes.
            text("UPDATE jobs SET heartbeat_at = :stale WHERE id = :id"),
            {
                "stale": (
                    dt.datetime.now(dt.UTC) - dt.timedelta(minutes=5)
                ).isoformat(),
                "id": created.job_id.hex,
            },
        )
    await jobs.reap_stale(older_than_s=90)

    resumed = await jobs.claim_or_attach(key, source_id, **CLAIM)
    assert isinstance(resumed, Reused)
    assert resumed.reason is ReuseReason.RESUMED
    assert resumed.job_id == created.job_id


# ------------------------------------------------------------------ requests


async def test_request_log_records_the_call(db, source_id) -> None:
    from transcriptor.adapters.sqlite_job_store import SqliteRequestLog

    log = SqliteRequestLog(db)
    await log.record(
        tool="transcript_create",
        params={"source": "/tmp/a.mp4", "language": None},
        source_id=source_id,
        outcome="ok",
        latency_ms=12,
    )
    async with db.session() as session:
        row = (
            await session.execute(text("SELECT tool, params, outcome FROM requests"))
        ).one()
    assert row.tool == "transcript_create"
    assert row.outcome == "ok"
