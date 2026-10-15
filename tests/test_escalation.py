"""The escalation ladder, resume, and surviving a database outage.

project-details §9: elevated -> clear cache; elevated sustained -> stop
accepting; critical -> pause up to 60s; critical past 60s -> graceful abort with
a resumable partial. Never kill -9, never setrlimit, never crash.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.exc import OperationalError

from transcriptor.domain.errors import DatabaseLost
from transcriptor.domain.job import JobStatus
from transcriptor.domain.transcript import Chunk, ChunkStatus, Segment

pytestmark = pytest.mark.db


def squeeze(governor, gb: float) -> None:
    governor._monitor.gb = gb
    governor.refresh()


# ------------------------------------------------------------- the ladder


async def test_elevated_pressure_clears_the_cache_and_carries_on(
    app, long_wav, governor
) -> None:
    created = await app.transcription.create(str(long_wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    cleared = {"n": 0}
    ctx.engine.clear_cache = lambda: cleared.__setitem__("n", cleared["n"] + 1)
    squeeze(governor, 3.0)  # within 1GB of the 2.5GB floor

    assert await app.pipeline.execute(ctx) is JobStatus.DONE
    assert cleared["n"] > 0, "the buffer cache must be given back between chunks"


class Scripted:
    """A monitor that returns a fixed sequence of available-memory readings,
    repeating the last one forever.

    Scripted rather than timed, so the test asserts the ladder's behaviour
    instead of racing the scheduler.
    """

    def __init__(self, *available_gb: float) -> None:
        self.script = list(available_gb)
        self.readings: list[float] = []

    def sample(self):
        import datetime as dt

        from transcriptor.domain.pressure import ResourceSample

        value = self.script[min(len(self.readings), len(self.script) - 1)]
        self.readings.append(value)
        return ResourceSample(
            rss_gb=0.2, available_gb=value, swap_used_gb=0.0,
            swap_delta_gb=0.0, total_gb=16.0, at=dt.datetime.now(dt.UTC),
        )


async def test_critical_pressure_pauses_then_continues(
    app, long_wav, governor
) -> None:
    """Pausing is not a stall: the machine gets a chance to recover on its own,
    which on a shared desktop it usually does within seconds.  Aborting straight
    away would throw away a job that only needed to wait."""
    created = await app.transcription.create(str(long_wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    # Calm enough to load the model, critical at the first chunk boundary, then
    # calm again — so the ladder must pause and then resume.
    governor._monitor = Scripted(12.0, 0.5, 12.0)
    governor.refresh()

    status = await app.pipeline.execute(ctx)

    assert status is JobStatus.DONE, "the job must continue once pressure clears"
    events = [e.to_status for e in await app.jobs.events(ctx.job_id)]
    assert JobStatus.PAUSED in events, "it must actually have paused"
    assert events[-1] is JobStatus.DONE

    notes = [e.note or "" for e in await app.jobs.events(ctx.job_id)]
    assert any("pressure cleared" in n for n in notes), "and said why it resumed"


async def test_sustained_critical_pressure_aborts_with_a_resumable_partial(
    app, long_wav, governor
) -> None:
    """The worst-case memory outcome becomes 'finish it later', not 'lost 40
    minutes of work'."""
    created = await app.transcription.create(str(long_wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    original = ctx.engine.transcribe
    calls = {"n": 0}

    def watch(audio, language, *, prompt=None):
        calls["n"] += 1
        if calls["n"] >= 2:
            squeeze(governor, 0.5)  # and never released
        return original(audio, language, prompt=prompt)

    ctx.engine.transcribe = watch

    assert await app.pipeline.execute(ctx) is JobStatus.FAILED

    status = await app.reads.status(created.job_id)
    assert status.error_code == "memory_pressure"
    assert status.resumable is True, "a memory abort must always be resumable"
    assert status.partial is True

    committed = [
        c for c in await app.transcripts.chunks(ctx.job_id)
        if c.status is ChunkStatus.DONE
    ]
    assert committed, "everything transcribed before the abort survives"


async def test_the_model_is_evicted_on_abort(app, long_wav, governor) -> None:
    """On abort the machine needs the memory now, not in ten minutes."""
    created = await app.transcription.create(str(long_wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    original = ctx.engine.transcribe

    def watch(audio, language, *, prompt=None):
        squeeze(governor, 0.5)
        return original(audio, language, prompt=prompt)

    ctx.engine.transcribe = watch
    await app.pipeline.execute(ctx)

    assert ctx.engine.loaded is False
    assert ctx.engine.unload_calls > 0


async def test_an_aborted_job_resumes_and_finishes(app, long_wav, governor) -> None:
    """The whole point of a resumable partial."""
    created = await app.transcription.create(str(long_wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    original = ctx.engine.transcribe
    calls = {"n": 0}

    def watch(audio, language, *, prompt=None):
        calls["n"] += 1
        if calls["n"] >= 2:
            squeeze(governor, 0.5)
        return original(audio, language, prompt=prompt)

    ctx.engine.transcribe = watch
    assert await app.pipeline.execute(ctx) is JobStatus.FAILED
    aborted_after = calls["n"]

    # The next request for the same source resumes rather than starting over.
    squeeze(governor, 12.0)
    again = await app.transcription.create(str(long_wav))
    assert again.job_id == created.job_id
    assert again.reuse_reason == "resumed"

    app.engine.transcribe = original
    assert await app.runner.tick() is JobStatus.DONE

    total = len(await app.transcripts.chunks(ctx.job_id))
    assert (
        await app.reads.status(created.job_id)
    ).status == "done"
    assert aborted_after < total, "the abort really did leave work undone"


# ------------------------------------------------------- database outage


class Flaky:
    """A transcript store whose commits fail until told otherwise."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.broken = False
        self.attempts = 0

    async def commit_chunk(self, job_id, chunk, segments):
        self.attempts += 1
        if self.broken:
            raise OperationalError("commit", {}, Exception("connection lost"))
        return await self._inner.commit_chunk(job_id, chunk, segments)

    def __getattr__(self, name):
        return getattr(self._inner, name)


async def test_a_commit_during_an_outage_is_buffered_not_lost(
    app, transcripts, jobs, source_id
) -> None:
    from transcriptor.adapters.buffered_store import BufferedTranscriptStore
    from transcriptor.domain.source import CaptionPolicy, result_key

    claim = dict(model="fake-engine", language=None, captions=CaptionPolicy.MANUAL_ONLY)
    created = await jobs.claim_or_attach(
        result_key(identity="/tmp/x.wav", **claim), source_id, **claim
    )

    flaky = Flaky(transcripts)
    store = BufferedTranscriptStore(flaky, max_segments=500)

    flaky.broken = True
    await store.commit_chunk(
        created.job_id, Chunk(0, 0, 10), [Segment(0, 0, 5, "held in memory")]
    )
    assert store.buffered_chunks == 1
    assert await transcripts.segments(created.job_id) == []

    # Reconnect: the buffered chunk flushes on the next commit.
    flaky.broken = False
    await store.commit_chunk(
        created.job_id, Chunk(1, 10, 20), [Segment(1, 10, 15, "after reconnect")]
    )
    assert store.buffered_chunks == 0
    assert [s.text for s in await transcripts.segments(created.job_id)] == [
        "held in memory",
        "after reconnect",
    ]


async def test_a_full_buffer_fails_resumably_with_committed_chunks_intact(
    app, transcripts, jobs, source_id
) -> None:
    from transcriptor.adapters.buffered_store import BufferedTranscriptStore
    from transcriptor.domain.source import CaptionPolicy, result_key

    claim = dict(model="fake-engine", language=None, captions=CaptionPolicy.MANUAL_ONLY)
    created = await jobs.claim_or_attach(
        result_key(identity="/tmp/y.wav", **claim), source_id, **claim
    )

    flaky = Flaky(transcripts)
    store = BufferedTranscriptStore(flaky, max_segments=3)

    await store.commit_chunk(
        created.job_id, Chunk(0, 0, 10), [Segment(0, 0, 5, "safely committed")]
    )
    flaky.broken = True

    with pytest.raises(DatabaseLost) as excinfo:
        for idx in range(1, 6):
            await store.commit_chunk(
                created.job_id,
                Chunk(idx, idx * 10, idx * 10 + 10),
                [Segment(idx, idx * 10, idx * 10 + 5, f"lost {idx}")],
            )

    assert excinfo.value.code == "database_lost"
    assert excinfo.value.resumable is True
    assert [s.text for s in await transcripts.segments(created.job_id)] == [
        "safely committed"
    ], "what landed before the outage must be intact"


async def test_buffered_chunks_flush_in_order(app, transcripts, jobs, source_id) -> None:
    """A later chunk must never land before an earlier one, or covered_until_s
    would claim coverage that has gaps."""
    from transcriptor.adapters.buffered_store import BufferedTranscriptStore
    from transcriptor.domain.source import CaptionPolicy, result_key

    claim = dict(model="fake-engine", language=None, captions=CaptionPolicy.MANUAL_ONLY)
    created = await jobs.claim_or_attach(
        result_key(identity="/tmp/z.wav", **claim), source_id, **claim
    )

    flaky = Flaky(transcripts)
    store = BufferedTranscriptStore(flaky)
    flaky.broken = True
    for idx in range(4):
        await store.commit_chunk(
            created.job_id,
            Chunk(idx, idx * 10, idx * 10 + 10),
            [Segment(idx, idx * 10, idx * 10 + 5, f"chunk {idx}")],
        )
    assert store.buffered_chunks == 4

    flaky.broken = False
    await store._flush()
    assert [s.text for s in await transcripts.segments(created.job_id)] == [
        f"chunk {i}" for i in range(4)
    ]


# ------------------------------------------------------------- eviction


async def test_an_idle_model_is_evicted(app) -> None:
    await app.engine.load()
    app.evictor._base_idle_s = 0
    assert await app.evictor.tick() is True
    assert app.engine.loaded is False


async def test_a_model_in_use_is_never_evicted(app, long_wav) -> None:
    await app.engine.load()
    app.evictor._base_idle_s = 0
    app.runner._current = uuid.uuid4()  # pretend a job is running
    assert await app.evictor.tick() is False
    assert app.engine.loaded is True


async def test_eviction_tightens_under_pressure(app, governor) -> None:
    from transcriptor.domain.pressure import Pressure, idle_evict_seconds

    assert idle_evict_seconds(Pressure.NORMAL) == 600
    assert idle_evict_seconds(Pressure.ELEVATED) == 60
    assert idle_evict_seconds(Pressure.CRITICAL) == 0

    await app.engine.load()
    squeeze(governor, 0.5)  # critical -> evict immediately
    assert await app.evictor.tick() is True


def test_the_buffering_decorator_covers_the_whole_port() -> None:
    """A decorator with an explicit passthrough list silently loses every method
    added to the store afterwards — which is exactly how a real job came to fail
    with AttributeError on `copy_from`."""
    from transcriptor.adapters.buffered_store import BufferedTranscriptStore
    from transcriptor.adapters.sqlite_transcript_store import SqliteTranscriptStore

    wrapped = BufferedTranscriptStore(SqliteTranscriptStore.__new__(SqliteTranscriptStore))
    surface = [
        name
        for name in dir(SqliteTranscriptStore)
        if not name.startswith("_") and callable(getattr(SqliteTranscriptStore, name))
    ]
    assert surface
    for name in surface:
        assert hasattr(wrapped, name), f"BufferedTranscriptStore drops {name}"
