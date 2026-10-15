"""Punctuate — put the sentence boundaries back after transcription.

Disabling `condition_on_previous_text` bought roughly 4x throughput and all but
eliminated repetition, at the cost of punctuation: only the first 30s window of
each Whisper call is well segmented, so later windows arrive as run-ons. Every
word is correct; the full stops are missing.

This stage restores them from the text, because they cannot be recovered from
timing — the degraded segment boundaries cut mid-sentence.

**The ASR model is evicted first.** Whisper peaks at 3.04GB and the punctuator
needs about 1.1GB; together they exceed the 4GB self cap, and transcription is
finished by the time this runs.
"""

from __future__ import annotations

import asyncio
import logging

from transcriptor.domain.job import Stage as JobStage
from transcriptor.domain.transcript import Segment
from transcriptor.services.pipeline import JobContext

log = logging.getLogger(__name__)


class PunctuateStage:
    name = "punctuate"
    job_stage = JobStage.RENDERING

    def __init__(self, *, punctuator=None) -> None:
        # Optional: without one the transcript is left exactly as transcribed.
        self._punctuator = punctuator

    async def satisfied(self, ctx: JobContext) -> bool:
        """Shares Finalize's checkpoint.

        There is no column of its own, and none is needed: restoring punctuation
        is idempotent — already-punctuated text comes back unchanged — so a
        crash between this stage and Finalize costs a re-run, not a corruption.
        """
        if self._punctuator is None:
            return True
        return ctx.job.word_count is not None

    async def run(self, ctx: JobContext) -> None:
        # Finalize writes `job.language`, and it runs *after* this stage, so the
        # detected language has to come from the engine. A layer-2 reuse never
        # called the engine but did copy the language onto the job, so that is
        # checked first.
        language = (
            ctx.job.language
            or ctx.language
            or getattr(ctx.engine, "detected_language", None)
        )
        if not self._punctuator.supports(language):
            # Transcriptor targets 99 languages and no punctuation model comes
            # close. Leaving the transcript alone beats corrupting it.
            log.info(
                "punctuation skipped: language not supported",
                extra={"job_id": str(ctx.job_id), "language": language},
            )
            return

        segments = await ctx.transcripts.segments(ctx.job_id)
        if not segments:
            return

        # Whisper and the punctuator do not fit under the self cap together.
        await self._evict_engine(ctx)
        await self._punctuator.load()

        loop = asyncio.get_running_loop()
        restored: list[Segment] = []
        changed = 0
        for segment in segments:
            text = await loop.run_in_executor(
                ctx.inference_executor, self._punctuator.restore, segment.text
            )
            if text != segment.text:
                changed += 1
            restored.append(
                Segment(
                    idx=segment.idx,
                    start=segment.start,
                    end=segment.end,
                    text=text,
                    words=segment.words,
                )
            )
            if segment.idx % 40 == 0:
                await ctx.jobs.heartbeat(ctx.job_id)

        if not changed:
            return

        await ctx.transcripts.replace_segments(ctx.job_id, restored)
        # Segments are the source of truth; any correction to them invalidates
        # every cached render.
        await ctx.transcripts.drop_renders(ctx.job_id)
        log.info(
            "punctuation restored",
            extra={
                "job_id": str(ctx.job_id),
                "segments": len(restored),
                "changed": changed,
            },
        )

    @staticmethod
    async def _evict_engine(ctx: JobContext) -> None:
        if not ctx.engine.loaded:
            return
        try:
            await ctx.engine.unload()
        except Exception:  # noqa: BLE001 - eviction must not fail the job
            log.exception("could not evict the ASR model before punctuating")
