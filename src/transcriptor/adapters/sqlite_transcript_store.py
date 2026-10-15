"""SqliteTranscriptStore — SQLAlchemy 2.0 async ORM over SQLite.

Segments are the source of truth.  Every format is a pure function of them, and
``renders`` is a lazy cache that any correction invalidates (invariant I7).

Full-text search is the one place this store reaches past the ORM into raw SQL,
because FTS5 is two virtual tables maintained by triggers rather than a column
on a row — see :meth:`search` and ``migrations/001_initial.sql``.
"""

from __future__ import annotations

import uuid

import logging

from sqlalchemy import delete, func, select, update
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import OperationalError

from transcriptor.adapters import orm, renderers
from transcriptor.adapters.db import Database
from transcriptor.adapters.orm import _utcnow
from transcriptor.domain.transcript import Chunk, ChunkStatus, Format, Segment, Word

log = logging.getLogger(__name__)


def _to_segment(row: orm.Segment, *, include_words: bool) -> Segment:
    words = None
    if include_words and row.words:
        words = [
            Word(
                start=w["start"],
                end=w["end"],
                text=w["text"],
                probability=w.get("probability"),
            )
            for w in row.words
        ]
    return Segment(
        idx=row.idx, start=row.start_s, end=row.end_s, text=row.text, words=words
    )


class SqliteTranscriptStore:
    """Implements :class:`transcriptor.ports.stores.TranscriptStore`."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # -------------------------------------------------------------- chunks

    async def plan_chunks(self, job_id: uuid.UUID, chunks: list[Chunk]) -> None:
        """Write the pending chunk rows.

        Idempotent, so replanning a resumed job leaves already-committed chunks
        untouched rather than resetting their status.
        """
        if not chunks:
            return
        async with self._db.begin() as session:
            await session.execute(
                sqlite_insert(orm.Chunk).on_conflict_do_nothing(
                    index_elements=[orm.Chunk.job_id, orm.Chunk.idx]
                ),
                [
                    {
                        "job_id": job_id,
                        "idx": c.idx,
                        "start_s": c.start_s,
                        "end_s": c.end_s,
                        "status": c.status.value,
                    }
                    for c in chunks
                ],
            )

    async def commit_chunk(
        self, job_id: uuid.UUID, chunk: Chunk, segments: list[Segment]
    ) -> None:
        """Chunk row, its segments, and job progress — in **one transaction**.

        This is the checkpoint unit.  Because the three writes land together, a
        partial chunk never exists, so ``segments`` can never contain rows whose
        chunk is not ``done`` (invariant I3).  It is also what makes partial
        results readable while a job runs, and resume possible after a crash.
        """
        async with self._db.begin() as session:
            chunk_stmt = sqlite_insert(orm.Chunk).values(
                job_id=job_id,
                idx=chunk.idx,
                start_s=chunk.start_s,
                end_s=chunk.end_s,
                status=ChunkStatus.DONE.value,
                attempts=chunk.attempts,
                engine_ms=chunk.engine_ms,
                committed_at=_utcnow(),
            )
            await session.execute(
                chunk_stmt.on_conflict_do_update(
                    index_elements=[orm.Chunk.job_id, orm.Chunk.idx],
                    set_={
                        "status": ChunkStatus.DONE.value,
                        "attempts": chunk_stmt.excluded.attempts,
                        "engine_ms": chunk_stmt.excluded.engine_ms,
                        "committed_at": _utcnow(),
                    },
                )
            )

            if segments:
                seg_stmt = sqlite_insert(orm.Segment)
                await session.execute(
                    seg_stmt.on_conflict_do_update(
                        index_elements=[orm.Segment.job_id, orm.Segment.idx],
                        set_={
                            "chunk_idx": seg_stmt.excluded.chunk_idx,
                            "start_s": seg_stmt.excluded.start_s,
                            "end_s": seg_stmt.excluded.end_s,
                            "text": seg_stmt.excluded.text,
                            "words": seg_stmt.excluded.words,
                        },
                    ),
                    [
                        {
                            "job_id": job_id,
                            "idx": s.idx,
                            "chunk_idx": chunk.idx,
                            "start_s": s.start,
                            "end_s": s.end,
                            "text": s.text,
                            "words": [
                                {
                                    "start": w.start,
                                    "end": w.end,
                                    "text": w.text,
                                    "probability": w.probability,
                                }
                                for w in s.words
                            ]
                            if s.words
                            else None,
                        }
                        for s in segments
                    ],
                )

            # Progress advances only forwards: a re-committed chunk on a resumed
            # job must not drag covered_until_s backwards.
            covered = func.max(orm.Job.covered_until_s, chunk.end_s)
            await session.execute(
                update(orm.Job)
                .where(orm.Job.id == job_id)
                .values(
                    last_chunk_idx=func.max(orm.Job.last_chunk_idx, chunk.idx),
                    covered_until_s=covered,
                    partial=True,
                    progress_pct=func.coalesce(
                        func.min(
                            100.0,
                            covered
                            / func.nullif(func.coalesce(orm.Job.duration_s, 0.0), 0.0)
                            * 100.0,
                        ),
                        orm.Job.progress_pct,
                    ),
                )
            )

    async def chunks(self, job_id: uuid.UUID) -> list[Chunk]:
        async with self._db.session() as session:
            rows = (
                await session.execute(
                    select(orm.Chunk)
                    .where(orm.Chunk.job_id == job_id)
                    .order_by(orm.Chunk.idx)
                )
            ).scalars().all()
        return [
            Chunk(
                idx=r.idx,
                start_s=r.start_s,
                end_s=r.end_s,
                status=ChunkStatus(r.status),
                attempts=r.attempts,
                engine_ms=r.engine_ms,
            )
            for r in rows
        ]

    # ------------------------------------------------------------ segments

    async def segments(
        self, job_id: uuid.UUID, *, include_words: bool = False
    ) -> list[Segment]:
        async with self._db.session() as session:
            rows = (
                await session.execute(
                    select(orm.Segment)
                    .where(orm.Segment.job_id == job_id)
                    .order_by(orm.Segment.idx)
                )
            ).scalars().all()
        return [_to_segment(r, include_words=include_words) for r in rows]

    async def segments_between(
        self, job_id: uuid.UUID, a: float, b: float, *, include_words: bool = False
    ) -> list[Segment]:
        """Segments overlapping ``[a, b)`` — not merely starting inside it, so a
        segment straddling the window boundary is still returned."""
        async with self._db.session() as session:
            rows = (
                await session.execute(
                    select(orm.Segment)
                    .where(
                        orm.Segment.job_id == job_id,
                        orm.Segment.end_s > a,
                        orm.Segment.start_s < b,
                    )
                    .order_by(orm.Segment.idx)
                )
            ).scalars().all()
        return [_to_segment(r, include_words=include_words) for r in rows]

    async def next_segment_idx(self, job_id: uuid.UUID) -> int:
        """The next free segment index for this job.

        Segment indices are per job and monotonic, so a resumed job continues
        the numbering instead of colliding with what is already committed.
        """
        async with self._db.session() as session:
            highest = (
                await session.execute(
                    select(func.max(orm.Segment.idx)).where(
                        orm.Segment.job_id == job_id
                    )
                )
            ).scalar_one_or_none()
        return 0 if highest is None else highest + 1

    async def word_count(self, job_id: uuid.UUID) -> int:
        """Total words across the job's segments.

        Postgres counted this in SQL, with ``regexp_split_to_array``. SQLite
        has no regular expressions and no arrays, so the split happens here —
        and only the ``text`` column is selected, so a two-hour transcript
        still does not pull its word timings into memory to be tallied.

        ``str.split()`` with no argument, rather than splitting on a space:
        it splits on any run of whitespace and discards empties, which is
        exactly what the regexp split it replaces did.
        """
        async with self._db.session() as session:
            rows = (
                await session.execute(
                    select(orm.Segment.text).where(orm.Segment.job_id == job_id)
                )
            ).scalars()
            return sum(len(text.split()) for text in rows)

    async def replace_segments(
        self, job_id: uuid.UUID, segments: list[Segment]
    ) -> int:
        """Rewrite segment text in place, keeping chunk and timing intact.

        A correction to the source of truth, which project-details §12 supports
        explicitly — and which is why callers must drop the render cache
        afterwards.
        """
        if not segments:
            return 0
        async with self._db.begin() as session:
            for segment in segments:
                await session.execute(
                    update(orm.Segment)
                    .where(
                        orm.Segment.job_id == job_id, orm.Segment.idx == segment.idx
                    )
                    .values(text=segment.text)
                )
        return len(segments)

    async def copy_from(self, source_job_id: uuid.UUID, dest_job_id: uuid.UUID) -> int:
        """Copy a finished transcript onto another job.

        Layer-2 dedupe found byte-identical audio, so re-running inference would
        produce the same words at the same times for the same cost.  The rows
        are duplicated rather than shared because every read path is keyed by
        ``job_id`` — and a transcript is about 120KB per hour, so the copy is
        cheaper than the indirection would be.
        """
        chunks = await self.chunks(source_job_id)
        segments = await self.segments(source_job_id, include_words=True)
        if not chunks:
            return 0

        span = max(c.end_s for c in chunks)
        await self.commit_chunk(
            dest_job_id,
            Chunk(idx=0, start_s=0.0, end_s=span, status=ChunkStatus.DONE, attempts=0),
            [
                Segment(idx=n, start=s.start, end=s.end, text=s.text, words=s.words)
                for n, s in enumerate(segments)
            ],
        )
        return len(segments)

    # -------------------------------------------------------------- render

    async def render(self, job_id: uuid.UUID, fmt: Format) -> str:
        """Render a format from segments, caching the result.

        The cache is consulted only for a *finished* job: a running one gains
        segments with every chunk, so a cached partial would be stale the moment
        it was written — and a running job being queryable is the whole point of
        per-chunk commits.
        """
        final = await self._job_is_final(job_id)
        if final:
            cached = await self.cached_render(job_id, fmt)
            if cached is not None:
                return cached

        segments = await self.segments(job_id, include_words=fmt is Format.JSON)
        content = renderers.render(segments, fmt)

        if final:
            stmt = sqlite_insert(orm.Render).values(
                job_id=job_id,
                format=fmt.value,
                content=content,
                bytes=len(content.encode()),
            )
            async with self._db.begin() as session:
                await session.execute(
                    stmt.on_conflict_do_update(
                        index_elements=[orm.Render.job_id, orm.Render.format],
                        set_={
                            "content": stmt.excluded.content,
                            "bytes": stmt.excluded.bytes,
                            "generated_at": _utcnow(),
                        },
                    )
                )
        return content

    async def _job_is_final(self, job_id: uuid.UUID) -> bool:
        async with self._db.session() as session:
            status = (
                await session.execute(
                    select(orm.Job.status).where(orm.Job.id == job_id)
                )
            ).scalar_one_or_none()
        return status in ("done", "failed", "cancelled", "interrupted")

    async def cached_render(self, job_id: uuid.UUID, fmt: Format) -> str | None:
        async with self._db.session() as session:
            return (
                await session.execute(
                    select(orm.Render.content).where(
                        orm.Render.job_id == job_id, orm.Render.format == fmt.value
                    )
                )
            ).scalar_one_or_none()

    async def drop_renders(self, job_id: uuid.UUID) -> int:
        """Invalidate the cache.  Any correction to a segment calls this."""
        async with self._db.begin() as session:
            result = await session.execute(
                delete(orm.Render).where(orm.Render.job_id == job_id)
            )
        return result.rowcount or 0

    async def search(self, job_id: uuid.UUID, query: str, *, limit: int = 10):
        """Whole words, then substrings, then near-misses.

        This replaces a Postgres ``tsvector`` search with a ``pg_trgm``
        fallback, and it keeps the same shape for the same reason: an
        exact-word index finds what someone actually remembers, and something
        character-level finds what they half-remember.

        ``segments_fts`` is tokenised ``unicode61 remove_diacritics 0``, the
        direct equivalent of the ``'simple'`` configuration this used before:
        **no stemming**, because across 99 languages no-stemming is the only
        configuration that behaves consistently, and **no diacritic
        stripping**, because stripping them mangles Hindi, Arabic and
        Vietnamese rather than helping.

        The third tier is the interesting one. ``segments_trgm`` matches
        *substrings*, which is not the same thing pg_trgm did: pg_trgm scored
        **similarity**, so it tolerated a genuine misspelling, and a trigram
        index does not — "quik brown foks" shares too few whole trigrams with
        "quick brown fox" to match as a substring. Losing that would have been
        a real regression in the thing fuzzy search is *for*, so the similarity
        coefficient is computed here instead, with the same formula pg_trgm
        uses: shared trigrams over total distinct trigrams.

        It only runs when both indexed tiers miss, and search is always scoped
        to one job, so the work is bounded by one transcript rather than by the
        database.
        """
        phrase = _fts_phrase(query)
        rows = await self._fts(job_id, "segments_fts", phrase, limit)
        if not rows:
            rows = await self._fts(job_id, "segments_trgm", phrase, limit)
        if rows:
            return [
                {
                    "idx": r.idx,
                    "start": r.start_s,
                    "end": r.end_s,
                    "text": r.text,
                    # bm25() returns a negative number, better matches being
                    # more negative. Flipped so "higher is better" holds for
                    # callers, exactly as ts_rank and similarity did.
                    "score": round(-float(r.score), 4),
                }
                for r in rows
            ]
        return await self._similar(job_id, query, limit)

    async def _fts(self, job_id: uuid.UUID, table: str, match: str, limit: int):
        """One FTS5 query, joined back to the row it indexes.

        The table name is interpolated because SQLite cannot bind an identifier
        — and it is safe here because it is never user input: both call sites
        pass a literal naming one of the two indexes the migration creates.
        The query itself is bound.
        """
        if not match:
            return []
        sql = sa_text(
            f"""
            SELECT s.idx, s.start_s, s.end_s, s.text, bm25({table}) AS score
              FROM {table} f
              JOIN segments s ON s.rowid = f.rowid
             WHERE {table} MATCH :match
               AND s.job_id = :job_id
             ORDER BY score, s.start_s
             LIMIT :limit
            """
        )
        async with self._db.session() as session:
            try:
                return (
                    await session.execute(
                        sql,
                        {
                            "match": match,
                            "job_id": _uuid_param(job_id),
                            "limit": limit,
                        },
                    )
                ).all()
            except OperationalError as exc:
                # A malformed MATCH expression is a bad *query*, not a broken
                # database. Someone searching for `AND` or an unbalanced quote
                # should get no results, not a stack trace.
                log.info("fts query rejected", extra={"error": str(exc)})
                return []

    async def _similar(self, job_id: uuid.UUID, query: str, limit: int):
        """Trigram similarity, the way pg_trgm defines it.

        The last tier, reached only when nothing indexed matched. Scored in
        Python because SQLite has no equivalent — and the alternative, dropping
        misspelling tolerance, would quietly remove the reason fuzzy search
        exists.
        """
        wanted = _trigrams(query)
        if not wanted:
            return []

        async with self._db.session() as session:
            rows = (
                await session.execute(
                    select(
                        orm.Segment.idx,
                        orm.Segment.start_s,
                        orm.Segment.end_s,
                        orm.Segment.text,
                    ).where(orm.Segment.job_id == job_id)
                )
            ).all()

        scored = []
        for row in rows:
            score = _similarity(wanted, _trigrams(row.text))
            # The same threshold pg_trgm defaulted to. Below it the "matches"
            # are noise, and returning noise is worse than returning nothing.
            if score > 0.15:
                scored.append(
                    {
                        "idx": row.idx,
                        "start": row.start_s,
                        "end": row.end_s,
                        "text": row.text,
                        "score": round(score, 4),
                    }
                )

        scored.sort(key=lambda r: (-r["score"], r["start"]))
        return scored[:limit]


def _trigrams(value: str) -> set[str]:
    """The trigram set of a string, as pg_trgm forms it.

    Lowercased, padded with two leading and one trailing space, so that word
    beginnings and endings carry their own trigrams — which is what makes a
    short word match at all.
    """
    cleaned = " ".join(value.lower().split())
    if not cleaned:
        return set()
    grams: set[str] = set()
    for word in cleaned.split(" "):
        padded = f"  {word} "
        grams.update(padded[i : i + 3] for i in range(len(padded) - 2))
    return grams


def _similarity(a: set[str], b: set[str]) -> float:
    """Shared trigrams over total distinct trigrams — pg_trgm's coefficient."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _fts_phrase(query: str) -> str:
    """Turn a user's words into an FTS5 phrase query.

    Quoted as a single phrase, with embedded quotes doubled. Without this,
    FTS5 reads bare input as its own query language — so a search for ``AND``
    or ``foo*`` or ``NEAR`` would either error or silently mean something the
    user did not ask for.
    """
    cleaned = query.strip()
    if not cleaned:
        return ""
    return '"' + cleaned.replace('"', '""') + '"'


def _uuid_param(value: uuid.UUID) -> str:
    """A UUID in the form the ORM stores.

    SQLAlchemy's ``Uuid`` type writes 32 hex characters with no dashes on
    SQLite. Hand-written SQL has to match that or it silently finds nothing —
    the sort of mismatch that returns an empty result rather than an error.
    """
    return value.hex
