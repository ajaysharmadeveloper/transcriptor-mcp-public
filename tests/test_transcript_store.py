"""SqliteTranscriptStore: per-chunk commits and the invariants they buy."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from transcriptor.domain.job import JobStatus
from transcriptor.domain.source import CaptionPolicy, result_key
from transcriptor.domain.transcript import Chunk, ChunkStatus, Format, Segment, Word

pytestmark = pytest.mark.db

CLAIM = dict(model="whisper-8bit", language=None, captions=CaptionPolicy.MANUAL_ONLY)


@pytest.fixture
async def job_id(jobs, source_id):
    created = await jobs.claim_or_attach(
        result_key(identity="/tmp/a.mp4", **CLAIM), source_id, **CLAIM
    )
    return created.job_id


def _segments(start_idx: int, a: float, b: float) -> list[Segment]:
    return [
        Segment(start_idx, a, (a + b) / 2, "first half"),
        Segment(start_idx + 1, (a + b) / 2, b, "second half"),
    ]


async def test_commit_chunk_writes_chunk_segments_and_progress(
    transcripts, jobs, db, job_id
) -> None:
    async with db.begin() as session:
        await session.execute(
            text("UPDATE jobs SET duration_s = 120 WHERE id = :id"), {"id": job_id.hex}
        )

    chunk = Chunk(idx=0, start_s=0.0, end_s=60.0, engine_ms=1234)
    await transcripts.commit_chunk(job_id, chunk, _segments(0, 0.0, 60.0))

    stored = await transcripts.chunks(job_id)
    assert [c.status for c in stored] == [ChunkStatus.DONE]
    assert stored[0].engine_ms == 1234

    assert len(await transcripts.segments(job_id)) == 2

    job = await jobs.get(job_id)
    assert job.covered_until_s == 60.0
    assert job.last_chunk_idx == 0
    assert job.partial is True
    assert job.progress_pct == pytest.approx(50.0)


async def test_no_segment_exists_without_a_done_chunk(
    transcripts, db, job_id
) -> None:
    """Invariant I3, asserted as a post-condition query over the whole table."""
    await transcripts.commit_chunk(
        job_id, Chunk(idx=0, start_s=0.0, end_s=60.0), _segments(0, 0.0, 60.0)
    )
    await transcripts.commit_chunk(
        job_id, Chunk(idx=1, start_s=60.0, end_s=120.0), _segments(2, 60.0, 120.0)
    )

    async with db.session() as session:
        orphans = (
            await session.execute(
                text(
                    """
                    SELECT count(*) FROM segments s
                    LEFT JOIN chunks c
                      ON c.job_id = s.job_id AND c.idx = s.chunk_idx
                    WHERE c.status IS DISTINCT FROM 'done'
                    """
                )
            )
        ).scalar_one()
    assert orphans == 0


async def test_progress_never_goes_backwards_on_resume(
    transcripts, jobs, db, job_id
) -> None:
    """A resumed job re-committing an earlier chunk must not drag
    covered_until_s back."""
    async with db.begin() as session:
        await session.execute(
            text("UPDATE jobs SET duration_s = 120 WHERE id = :id"), {"id": job_id.hex}
        )

    await transcripts.commit_chunk(
        job_id, Chunk(idx=1, start_s=60.0, end_s=120.0), _segments(2, 60.0, 120.0)
    )
    await transcripts.commit_chunk(
        job_id, Chunk(idx=0, start_s=0.0, end_s=60.0), _segments(0, 0.0, 60.0)
    )

    job = await jobs.get(job_id)
    assert job.covered_until_s == 120.0
    assert job.last_chunk_idx == 1


async def test_recommitting_a_chunk_is_idempotent(transcripts, job_id) -> None:
    chunk = Chunk(idx=0, start_s=0.0, end_s=60.0)
    await transcripts.commit_chunk(job_id, chunk, _segments(0, 0.0, 60.0))
    await transcripts.commit_chunk(job_id, chunk, _segments(0, 0.0, 60.0))

    assert len(await transcripts.segments(job_id)) == 2
    assert len(await transcripts.chunks(job_id)) == 1


async def test_plan_chunks_does_not_reset_committed_work(
    transcripts, job_id
) -> None:
    """Replanning a resumed job must leave finished chunks alone."""
    await transcripts.commit_chunk(
        job_id, Chunk(idx=0, start_s=0.0, end_s=60.0), _segments(0, 0.0, 60.0)
    )
    await transcripts.plan_chunks(
        job_id,
        [
            Chunk(idx=0, start_s=0.0, end_s=60.0),
            Chunk(idx=1, start_s=60.0, end_s=120.0),
        ],
    )
    stored = {c.idx: c.status for c in await transcripts.chunks(job_id)}
    assert stored == {0: ChunkStatus.DONE, 1: ChunkStatus.PENDING}


async def test_words_round_trip_only_when_asked(transcripts, job_id) -> None:
    segments = [
        Segment(
            0,
            0.0,
            1.0,
            "hello world",
            words=[Word(0.0, 0.5, "hello"), Word(0.5, 1.0, "world")],
        )
    ]
    await transcripts.commit_chunk(job_id, Chunk(0, 0.0, 1.0), segments)

    assert (await transcripts.segments(job_id))[0].words is None
    with_words = await transcripts.segments(job_id, include_words=True)
    assert [w.text for w in with_words[0].words] == ["hello", "world"]


async def test_segments_between_returns_overlaps_not_just_containment(
    transcripts, job_id
) -> None:
    await transcripts.commit_chunk(
        job_id,
        Chunk(0, 0.0, 30.0),
        [
            Segment(0, 0.0, 10.0, "a"),
            Segment(1, 10.0, 20.0, "b"),
            Segment(2, 20.0, 30.0, "c"),
        ],
    )
    window = await transcripts.segments_between(job_id, 9.0, 11.0)
    assert [s.text for s in window] == ["a", "b"], "a straddling segment must appear"


async def test_partial_transcript_is_readable_while_the_job_runs(
    transcripts, jobs, job_id
) -> None:
    """A running job is queryable, not opaque — this falls out of per-chunk
    commits for free."""
    await jobs.transition(job_id, JobStatus.RUNNING)
    await transcripts.commit_chunk(
        job_id, Chunk(0, 0.0, 60.0), [Segment(0, 0.0, 2.0, "already committed")]
    )

    assert (await jobs.get(job_id)).status is JobStatus.RUNNING
    srt = await transcripts.render(job_id, Format.SRT)
    assert "already committed" in srt


async def test_render_srt_from_segments(transcripts, job_id) -> None:
    await transcripts.commit_chunk(
        job_id,
        Chunk(0, 0.0, 6.0),
        [Segment(0, 0.0, 2.5, "Hello world"), Segment(1, 2.5, 5.25, "Second line")],
    )
    srt = await transcripts.render(job_id, Format.SRT)
    assert srt == (
        "1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n"
        "2\n00:00:02,500 --> 00:00:05,250\nSecond line\n"
    )


async def test_render_txt_from_segments(transcripts, job_id) -> None:
    await transcripts.commit_chunk(
        job_id,
        Chunk(0, 0.0, 6.0),
        [Segment(0, 0.0, 2.5, "Hello world"), Segment(1, 2.5, 5.25, "Second line")],
    )
    assert await transcripts.render(job_id, Format.TXT) == "Hello world\nSecond line\n"


async def test_word_count(transcripts, job_id) -> None:
    await transcripts.commit_chunk(
        job_id,
        Chunk(0, 0.0, 6.0),
        [Segment(0, 0.0, 2.5, "one two three"), Segment(1, 2.5, 5.0, "four five")],
    )
    assert await transcripts.word_count(job_id) == 5


async def test_deleting_a_job_cascades_to_its_transcript(
    transcripts, db, job_id
) -> None:
    await transcripts.commit_chunk(
        job_id, Chunk(0, 0.0, 6.0), [Segment(0, 0.0, 2.5, "x")]
    )
    async with db.begin() as session:
        await session.execute(text("DELETE FROM jobs WHERE id = :id"), {"id": job_id.hex})
        assert (
            await session.execute(text("SELECT count(*) FROM segments"))
        ).scalar_one() == 0
        assert (
            await session.execute(text("SELECT count(*) FROM chunks"))
        ).scalar_one() == 0
