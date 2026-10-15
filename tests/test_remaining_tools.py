"""search, segments, list, export, cancel, delete — and the render cache."""

from __future__ import annotations

import json
import uuid

import pytest
from fastmcp import Client, FastMCP

from transcriptor import tools
from transcriptor.domain.job import JobStatus
from transcriptor.domain.transcript import Format

pytestmark = pytest.mark.db


@pytest.fixture
def mcp(app):
    server = FastMCP(name="transcriptor-test")
    tools.register(
        server,
        tools.Deps(
            transcription=app.transcription, reads=app.reads, health=app.health,
            lifecycle=app.lifecycle, clips=app.clips, requests=None,
        ),
    )
    return server


def payload(result):
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


@pytest.fixture
async def done_job(app, long_wav):
    created = await app.transcription.create(str(long_wav))
    assert await app.runner.tick() is JobStatus.DONE
    return created.job_id


# ------------------------------------------------------------------ search


async def test_search_finds_a_phrase_with_timestamps(app, done_job) -> None:
    hits = await app.reads.search(done_job, "quick brown fox")
    assert hits
    assert hits[0]["start"] >= 0
    assert "quick" in hits[0]["text"].lower()
    assert hits[0]["context_start"] <= hits[0]["start"]
    assert hits[0]["context_end"] >= hits[0]["end"]


async def test_search_falls_back_to_fuzzy_matching(app, done_job) -> None:
    """What actually helps when someone half-remembers a phrase."""
    exact = await app.reads.search(done_job, "quick brown fox")
    fuzzy = await app.reads.search(done_job, "quik brown foks")
    assert exact, "the sanity case must match"
    assert fuzzy, "a misspelling must still find something"


async def test_search_returns_nothing_for_an_absent_phrase(app, done_job) -> None:
    assert await app.reads.search(done_job, "zzzznotpresentzzzz") == []


async def test_search_works_on_a_running_job(app, long_wav) -> None:
    """Per-chunk commits are what make a running job queryable, not opaque."""
    created = await app.transcription.create(str(long_wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    from transcriptor.services.stages.detect import DetectStage
    from transcriptor.services.stages.extract import ExtractStage
    from transcriptor.services.stages.fetch import FetchStage
    from transcriptor.services.stages.resolve import ResolveStage
    from transcriptor.services.stages.transcribe import TranscribeStage

    await ResolveStage().run(ctx)
    await ctx.refresh()
    await FetchStage().run(ctx)
    await ExtractStage().run(ctx)
    await DetectStage(detector=app.detector).run(ctx)

    original = ctx.engine.transcribe
    calls = {"n": 0}

    def stop_early(audio, language, *, prompt=None):
        calls["n"] += 1
        if calls["n"] >= 2:
            ctx.cancel.cancel("enough")
        return original(audio, language, prompt=prompt)

    ctx.engine.transcribe = stop_early
    from transcriptor.domain.errors import Cancelled

    with pytest.raises(Cancelled):
        await TranscribeStage().run(ctx)

    assert (await app.jobs.get(ctx.job_id)).status is JobStatus.RUNNING
    assert await app.reads.search(created.job_id, "quick brown fox")


async def test_search_rejects_an_empty_query(app, done_job) -> None:
    from transcriptor.domain.errors import TranscriptorError

    with pytest.raises(TranscriptorError):
        await app.reads.search(done_job, "   ")


# ---------------------------------------------------------------- segments


async def test_segments_returns_a_window(app, done_job) -> None:
    window = await app.reads.segments(done_job, 2.0, 5.0)
    assert window
    for seg in window:
        assert seg["end"] > 2.0 and seg["start"] < 5.0


async def test_segments_omits_words_by_default(app, done_job) -> None:
    assert all("words" not in s for s in await app.reads.segments(done_job, 0, 12))


async def test_segments_rejects_a_reversed_window(app, done_job) -> None:
    from transcriptor.domain.errors import TranscriptorError

    with pytest.raises(TranscriptorError):
        await app.reads.segments(done_job, 10.0, 2.0)


# -------------------------------------------------------------------- list


async def test_list_returns_recent_jobs(app, done_job) -> None:
    jobs = await app.reads.list_jobs()
    assert jobs
    assert jobs[0].job_id == done_job
    assert jobs[0].status == "done"
    assert jobs[0].source


async def test_list_filters_by_status(app, done_job, wav) -> None:
    await app.transcription.create(str(wav))
    assert [j.status for j in await app.reads.list_jobs(status="queued")] == ["queued"]
    assert [j.status for j in await app.reads.list_jobs(status="done")] == ["done"]


async def test_list_rejects_an_unknown_status(app) -> None:
    from transcriptor.domain.errors import TranscriptorError

    with pytest.raises(TranscriptorError):
        await app.reads.list_jobs(status="nearly-done")


# ------------------------------------------------------------------ export


async def test_export_writes_the_file(app, done_job, tmp_path) -> None:
    dest = tmp_path / "out" / "transcript.srt"
    result = await app.lifecycle.export(done_job, fmt="srt", dest_path=str(dest))

    assert dest.exists()
    assert result.bytes == dest.stat().st_size
    assert " --> " in dest.read_text()


async def test_export_refuses_a_relative_path(app, done_job) -> None:
    from transcriptor.domain.errors import TranscriptorError

    with pytest.raises(TranscriptorError) as excinfo:
        await app.lifecycle.export(done_job, fmt="srt", dest_path="out/x.srt")
    assert excinfo.value.code == "relative_path_not_allowed"


async def test_a_failed_export_leaves_nothing_partial(app, done_job, tmp_path) -> None:
    from transcriptor.adapters.local_files import ExportPathUnwritable

    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory")
    with pytest.raises(ExportPathUnwritable):
        await app.lifecycle.export(
            done_job, fmt="srt", dest_path=str(blocked / "x.srt")
        )
    assert blocked.read_text() == "i am a file, not a directory"


# ------------------------------------------------------------------ cancel


async def test_cancelling_a_queued_job_stops_it_immediately(app, wav) -> None:
    created = await app.transcription.create(str(wav))
    result = await app.lifecycle.cancel(created.job_id)

    assert result["cancelled"] is True
    assert (await app.reads.status(created.job_id)).status == "cancelled"


async def test_cancelling_a_finished_job_is_a_no_op(app, done_job) -> None:
    result = await app.lifecycle.cancel(done_job)
    assert result["cancelled"] is False
    assert "already done" in result["detail"]


async def test_cancelling_an_unknown_job_says_so(app) -> None:
    from transcriptor.domain.errors import JobNotFound

    with pytest.raises(JobNotFound):
        await app.lifecycle.cancel(str(uuid.uuid4()))


# ------------------------------------------------------------------ delete


async def test_delete_removes_the_transcript(app, done_job) -> None:
    uid = uuid.UUID(done_job)
    assert await app.transcripts.segments(uid)

    result = await app.lifecycle.delete(done_job)
    assert result.deleted is True
    assert await app.jobs.get(uid) is None
    assert await app.transcripts.segments(uid) == []
    assert await app.transcripts.chunks(uid) == []


async def test_delete_refuses_while_the_job_is_running(app, wav) -> None:
    from transcriptor.domain.errors import TranscriptorError

    created = await app.transcription.create(str(wav))
    await app.jobs.next_queued()

    with pytest.raises(TranscriptorError) as excinfo:
        await app.lifecycle.delete(created.job_id)
    assert excinfo.value.code == "job_in_flight"


async def test_delete_can_free_the_scratch_audio(app, long_wav) -> None:
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
    result = await app.lifecycle.delete(created.job_id, delete_media=True)
    assert result.freed_bytes > 0
    assert not ctx.job_scratch.exists()


# ----------------------------------------------------------- render cache


async def test_a_finished_render_is_cached(app, done_job) -> None:
    uid = uuid.UUID(done_job)
    assert await app.transcripts.cached_render(uid, Format.SRT) is None

    first = await app.transcripts.render(uid, Format.SRT)
    assert await app.transcripts.cached_render(uid, Format.SRT) == first
    assert await app.transcripts.render(uid, Format.SRT) == first


async def test_a_running_job_is_never_cached(app, long_wav) -> None:
    """A running job gains segments with every chunk, so a cached partial would
    be stale the moment it was written."""
    created = await app.transcription.create(str(long_wav))
    await app.jobs.next_queued()
    uid = uuid.UUID(created.job_id)

    await app.transcripts.render(uid, Format.TXT)
    assert await app.transcripts.cached_render(uid, Format.TXT) is None


async def test_renders_are_always_regenerable(app, done_job) -> None:
    """Invariant I7: delete every render, re-render, byte-compare."""
    uid = uuid.UUID(done_job)
    before = {fmt: await app.transcripts.render(uid, fmt) for fmt in Format}

    assert await app.transcripts.drop_renders(uid) == len(Format)
    for fmt in Format:
        assert await app.transcripts.cached_render(uid, fmt) is None

    after = {fmt: await app.transcripts.render(uid, fmt) for fmt in Format}
    assert after == before


# ------------------------------------------------------- over the wire


async def test_every_tool_answers_over_the_wire(mcp, app, done_job, tmp_path) -> None:
    async with Client(mcp) as client:
        listed = {t.name for t in await client.list_tools()}
        assert len(listed) == 13

        assert payload(await client.call_tool("transcript_list", {}))["result"]
        hits = payload(
            await client.call_tool(
                "transcript_search", {"job_id": done_job, "query": "quick brown fox"}
            )
        )["result"]
        assert hits

        window = payload(
            await client.call_tool(
                "transcript_segments",
                {"job_id": done_job, "start_s": 0.0, "end_s": 5.0},
            )
        )["result"]
        assert window

        dest = tmp_path / "wire.srt"
        exported = payload(
            await client.call_tool(
                "transcript_export", {"job_id": done_job, "dest_path": str(dest)}
            )
        )
        assert exported["ok"] and dest.exists()

        cancelled = payload(
            await client.call_tool("transcript_cancel", {"job_id": done_job})
        )
        assert cancelled["cancelled"] is False  # already done

        deleted = payload(
            await client.call_tool("transcript_delete", {"job_id": done_job})
        )
        assert deleted["ok"] and deleted["deleted"] is True
