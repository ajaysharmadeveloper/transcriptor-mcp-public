"""Extract — decode to 16kHz mono WAV and hash the audio.

Satisfied when the source has an ``audio_hash`` *and* the scratch WAV is still
present.  Both halves matter: after a crash the hash survives in Postgres but
the scratch directory is wiped at startup, so this stage correctly re-runs.
Re-fetching ten minutes of audio beats redoing hours of inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

from transcriptor.domain.job import Origin
from transcriptor.domain.job import Stage as JobStage
from transcriptor.services.pipeline import JobContext

log = logging.getLogger(__name__)

WAV_NAME = "audio.wav"


class ExtractStage:
    name = "extract"
    job_stage = JobStage.EXTRACTING

    async def satisfied(self, ctx: JobContext) -> bool:
        if ctx.from_captions:
            return True
        wav = ctx.job_scratch / WAV_NAME
        if not wav.exists():
            return False
        source = ctx.source or await ctx.jobs.get_source(ctx.job.source_id) or {}
        if not source.get("audio_hash"):
            return False
        ctx.source = source
        ctx.wav = wav
        return True

    async def run(self, ctx: JobContext) -> None:
        ctx.source = ctx.source or await ctx.jobs.get_source(ctx.job.source_id) or {}
        src = ctx.media_path or Path(ctx.source["source_key"])
        dest = ctx.job_scratch / WAV_NAME

        ctx.wav = await ctx.extractor.to_wav16k(src, dest, cancel=ctx.cancel)

        # blake2b over the decoded PCM: identical regardless of container,
        # bitrate or where the file came from. Hashed on the cpu executor,
        # because tens of megabytes is not "fast enough" for the event loop.
        audio_hash = await ctx.extractor.audio_hash(ctx.wav)
        await ctx.jobs.set_audio_hash(ctx.job.source_id, audio_hash)
        ctx.source["audio_hash"] = audio_hash

        await self._reuse_identical_audio(ctx, audio_hash)

    @staticmethod
    async def _reuse_identical_audio(ctx: JobContext, audio_hash: str) -> None:
        """Layer 2 — if this exact audio is already transcribed, take it.

        Catches what layer 1 cannot: the same video from a shortened URL and a
        full one, or a local copy of something already pulled from a platform.
        The freshly fetched audio is simply discarded.

        Skipped entirely when the caller passed ``force``. Layer 1 could honour
        that at create time; this one runs a stage later, so the flag has to be
        read back off the job.
        """
        if ctx.job.forced:
            log.info(
                "layer-2 dedupe skipped: force was requested",
                extra={"job_id": str(ctx.job_id)},
            )
            return

        twin = await ctx.jobs.find_done_by_audio_hash(
            audio_hash,
            model=ctx.job.model or "",
            language=ctx.job.requested_language,
            exclude_job_id=ctx.job_id,
        )
        if twin is None:
            return

        copied = await ctx.transcripts.copy_from(twin, ctx.job_id)
        if not copied:
            return

        # Carry the detected language across too. The transcript is byte-identical
        # work, so reporting `language: null` for it would be a worse answer than
        # the one the original job already produced.
        source_job = await ctx.jobs.get(twin)
        await ctx.jobs.set_job_fields(
            ctx.job_id,
            origin=Origin.ASR.value,
            language=source_job.language if source_job else None,
        )
        await ctx.jobs.set_stage(ctx.job_id, JobStage.TRANSCRIBING)
        await ctx.refresh()
        log.info(
            "identical audio already transcribed — reused",
            extra={
                "job_id": str(ctx.job_id),
                "reused_from": str(twin),
                "segments": copied,
                "audio_hash": audio_hash[:16],
            },
        )
