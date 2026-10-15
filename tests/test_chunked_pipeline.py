"""Chunking through the real pipeline: per-chunk commits, partials, resume.

Against a real Postgres and real ffmpeg, with FakeEngine standing in for MLX.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from transcriptor.domain.errors import Cancelled
from transcriptor.domain.job import JobStatus
from transcriptor.domain.transcript import ChunkStatus, Format, plan_chunks

pytestmark = pytest.mark.db


async def test_a_long_file_is_split_into_several_chunks(app, long_wav) -> None:
    created = await app.transcription.create(str(long_wav))
    assert await app.runner.tick() is JobStatus.DONE

    job_id = uuid.UUID(created.job_id)
    chunks = await app.transcripts.chunks(job_id)
    assert len(chunks) == len(plan_chunks(12.0, chunk_s=2.0, overlap_s=0.5))
    assert len(chunks) > 4, "the fixture must actually span several chunks"
    assert all(c.status is ChunkStatus.DONE for c in chunks)


async def test_every_chunk_commits_its_own_timing(app, long_wav) -> None:
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    chunks = await app.transcripts.chunks(uuid.UUID(created.job_id))
    assert all(c.engine_ms is not None for c in chunks)
    assert all(c.attempts == 1 for c in chunks)


async def test_segment_indices_are_unique_and_ordered(app, long_wav) -> None:
    """They are per job and monotonic, so a resumed job continues the numbering
    rather than colliding with what is already committed."""
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    segments = await app.transcripts.segments(uuid.UUID(created.job_id))
    indices = [s.idx for s in segments]
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))
    assert indices == list(range(len(indices)))


async def test_the_stitched_transcript_has_no_repeated_run(app, long_wav) -> None:
    """The failure this whole mechanism exists to prevent: the overlap's words
    appearing twice."""
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    txt = await app.reads.get(created.job_id, fmt="txt", mode="full")
    lines = [ln for ln in txt.content.splitlines() if ln.strip()]
    for previous, following in zip(lines, lines[1:]):
        assert previous != following, f"repeated line across a seam: {previous!r}"


async def test_segment_times_never_go_backwards(app, long_wav) -> None:
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    segments = await app.transcripts.segments(uuid.UUID(created.job_id))
    starts = [s.start for s in segments]
    assert starts == sorted(starts)
    assert segments[0].start == pytest.approx(0.0, abs=0.01)


async def test_no_orphan_segments_after_a_chunked_run(app, long_wav, db) -> None:
    """Invariant I3, over a job with many chunks rather than one."""
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    async with db.session() as session:
        orphans = (
            await session.execute(
                text(
                    "SELECT count(*) FROM segments s LEFT JOIN chunks c "
                    "ON c.job_id = s.job_id AND c.idx = s.chunk_idx "
                    "WHERE c.status IS DISTINCT FROM 'done'"
                )
            )
        ).scalar_one()
    assert orphans == 0


# ------------------------------------------------------------ partial reads


async def test_a_partial_transcript_is_readable_mid_job(app, long_wav) -> None:
    """Per-chunk commits are what make a running job queryable rather than
    opaque — and what turn a memory abort into 'finish it later'."""
    created = await app.transcription.create(str(long_wav))
    job_id = uuid.UUID(created.job_id)

    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)
    from transcriptor.services.stages.extract import ExtractStage
    from transcriptor.services.stages.resolve import ResolveStage
    from transcriptor.services.stages.transcribe import TranscribeStage

    await ResolveStage().run(ctx)
    await ctx.refresh()
    await ExtractStage().run(ctx)

    # Let two chunks commit, then stop — as an interrupted job would have.
    original = ctx.engine.transcribe
    calls = {"n": 0}

    def counting(audio, language, *, prompt=None):
        calls["n"] += 1
        if calls["n"] >= 2:
            ctx.cancel.cancel("stopping after a couple of chunks")
        return original(audio, language)

    ctx.engine.transcribe = counting
    with pytest.raises(Cancelled):
        await TranscribeStage().run(ctx)

    view = await app.reads.get(created.job_id, fmt="srt", mode="full")
    assert view.partial is True, "a job that stopped early reports partial"
    assert 0 < view.covered_until_s < 12.0, "and says how far it reached"
    assert view.content.count(" --> ") > 0, "with the committed cues readable"


# ------------------------------------------------------------ cancellation


async def test_cancelling_stops_at_a_chunk_boundary(app, long_wav) -> None:
    """Inference in a thread cannot be interrupted mid-call, so cancellation is
    cooperative and costs at most one chunk."""
    created = await app.transcription.create(str(long_wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)
    ctx.cancel.cancel("user asked to stop")

    status = await app.pipeline.execute(ctx)
    assert status is JobStatus.CANCELLED

    view = await app.reads.status(created.job_id)
    assert view.status == "cancelled"
    assert view.error_code is None, "cancelling is not a failure"


async def test_a_cancelled_job_keeps_what_it_committed(app, long_wav) -> None:
    created = await app.transcription.create(str(long_wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    from transcriptor.services.stages.extract import ExtractStage
    from transcriptor.services.stages.resolve import ResolveStage
    from transcriptor.services.stages.transcribe import TranscribeStage

    await ResolveStage().run(ctx)
    await ctx.refresh()
    await ExtractStage().run(ctx)

    # Let a couple of chunks land, then ask it to stop.
    stage = TranscribeStage()
    original = ctx.engine.transcribe
    calls = {"n": 0}

    def counting(audio, language, *, prompt=None):
        calls["n"] += 1
        if calls["n"] >= 2:
            ctx.cancel.cancel("enough")
        return original(audio, language)

    ctx.engine.transcribe = counting
    with pytest.raises(Cancelled):
        await stage.run(ctx)

    committed = await app.transcripts.chunks(uuid.UUID(created.job_id))
    done = [c for c in committed if c.status is ChunkStatus.DONE]
    assert done, "the chunks finished before the cancel must survive"
    assert len(done) < len(committed), "and it stopped before the end"


async def test_cancelling_before_any_work_is_still_clean(app, wav) -> None:
    created = await app.transcription.create(str(wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)
    ctx.cancel.cancel()

    assert await app.pipeline.execute(ctx) is JobStatus.CANCELLED
    assert (await app.reads.status(created.job_id)).status == "cancelled"


# ----------------------------------------------------------------- resume


async def test_resume_reruns_only_the_uncommitted_chunks(app, long_wav) -> None:
    """A 90-minute job that died at minute 80 re-runs ten minutes of audio, not
    ninety."""
    created = await app.transcription.create(str(long_wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    from transcriptor.services.stages.extract import ExtractStage
    from transcriptor.services.stages.resolve import ResolveStage
    from transcriptor.services.stages.transcribe import TranscribeStage

    await ResolveStage().run(ctx)
    await ctx.refresh()
    await ExtractStage().run(ctx)

    stage = TranscribeStage()
    original = ctx.engine.transcribe
    calls = {"n": 0}

    def counting(audio, language, *, prompt=None):
        calls["n"] += 1
        if calls["n"] >= 3:
            ctx.cancel.cancel("simulated crash")
        return original(audio, language)

    ctx.engine.transcribe = counting
    with pytest.raises(Cancelled):
        await stage.run(ctx)

    first_pass = calls["n"]
    committed = len([c for c in await app.transcripts.chunks(ctx.job_id)
                     if c.status is ChunkStatus.DONE])
    assert committed > 0

    # Resume: the same stage over the same job, engine restored.
    ctx.engine.transcribe = original
    ctx.cancel = type(ctx.cancel)()
    ctx.engine.transcribe_calls.clear()
    await ctx.refresh()
    await stage.run(ctx)

    total_chunks = len(await app.transcripts.chunks(ctx.job_id))
    second_pass = len(ctx.engine.transcribe_calls)
    assert second_pass == total_chunks - committed, (
        f"resume re-ran {second_pass} chunks; only the "
        f"{total_chunks - committed} uncommitted ones should have run"
    )

    segments = await app.transcripts.segments(ctx.job_id)
    indices = [s.idx for s in segments]
    assert len(indices) == len(set(indices)), "resume must not collide indices"


# ------------------------------------------------------ context across seams


async def test_the_first_chunk_gets_no_prompt(app, long_wav) -> None:
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()
    assert app.engine.prompts[0] is None, "there is nothing committed to continue from"


async def test_every_later_chunk_continues_the_previous_text(app, long_wav) -> None:
    """Whisper starts a chunk cold without this and loses casing and punctuation
    for the rest of it — which on a long file is most of the transcript."""
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    assert len(app.engine.prompts) > 3
    for prompt in app.engine.prompts[1:]:
        assert prompt, "every chunk after the first must carry context"
        assert len(prompt) <= 120, "the prompt shares Whisper's context window"


async def test_the_prompt_is_the_tail_of_what_was_committed(app, long_wav) -> None:
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    segments = await app.transcripts.segments(uuid.UUID(created.job_id))
    committed = " ".join(s.text for s in segments)
    # The second chunk's prompt must be real committed text, not invented.
    second = app.engine.prompts[1]
    assert second.split()[-1] in committed


async def test_a_chunked_transcript_repeats_no_phrase(app, long_wav) -> None:
    """The regression a long seam prompt caused: Whisper echoing the prompt back
    into the transcript, past the seam where stitching never looks."""
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    view = await app.reads.get(created.job_id, fmt="txt", mode="full")
    words = view.content.split()
    seen: dict[tuple, int] = {}
    for i in range(len(words) - 4):
        gram = tuple(w.lower().strip(".,!?") for w in words[i : i + 5])
        assert gram not in seen, (
            f"5-gram repeated at word {i} (first at {seen.get(gram)}): "
            f"{' '.join(words[i : i + 5])}"
        )
        seen[gram] = i
