"""Finalize — job totals.

Satisfied when ``word_count`` is set.  The pipeline makes the terminal
transition to ``done``; this stage only writes the numbers that transition
reports.
"""

from __future__ import annotations

from transcriptor.domain.job import Stage as JobStage
from transcriptor.services.pipeline import JobContext


class FinalizeStage:
    name = "finalize"
    job_stage = JobStage.RENDERING

    async def satisfied(self, ctx: JobContext) -> bool:
        return ctx.job.word_count is not None

    async def run(self, ctx: JobContext) -> None:
        word_count = await ctx.transcripts.word_count(ctx.job_id)
        await ctx.jobs.set_job_fields(
            ctx.job_id,
            word_count=word_count,
            language=ctx.language or ctx.engine.detected_language,
            covered_until_s=ctx.job.duration_s or ctx.job.covered_until_s,
        )
        await ctx.refresh()
