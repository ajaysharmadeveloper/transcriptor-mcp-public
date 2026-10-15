"""Detect — find the speech and drop everything else.

Mandatory rather than an optimisation.  Whisper hallucinates confident invented
sentences over silence and drifts on long-form audio, so the non-speech is
removed before transcription and the timings are mapped back afterwards.

Satisfied when the timeline is stored *and* the speech-only WAV is present.  If
the timeline survived a crash but scratch did not, the WAV is rebuilt from the
stored regions without re-running Silero — which is the whole reason the
timeline is a column rather than a scratch file.
"""

from __future__ import annotations

import asyncio
import logging

from transcriptor.domain.errors import NoSpeechDetected
from transcriptor.domain.job import Stage as JobStage
from transcriptor.domain.transcript import Timeline
from transcriptor.services.pipeline import JobContext

log = logging.getLogger(__name__)

SPEECH_WAV_NAME = "speech.wav"


class DetectStage:
    name = "detect"
    job_stage = JobStage.VAD

    def __init__(self, *, detector=None) -> None:
        # Optional: without a detector the whole recording is treated as speech,
        # which is what the phase 1 behaviour was.
        self._detector = detector

    async def satisfied(self, ctx: JobContext) -> bool:
        if ctx.from_captions:
            return True
        if ctx.job.speech_timeline is None:
            return False
        speech = ctx.job_scratch / SPEECH_WAV_NAME
        if not speech.exists():
            return False
        ctx.timeline = ctx.job.speech_timeline
        ctx.speech_wav = speech
        return True

    async def run(self, ctx: JobContext) -> None:
        timeline = ctx.job.speech_timeline

        if timeline is None:
            timeline = await self._detect(ctx)
            await ctx.jobs.set_job_fields(
                ctx.job_id, speech_timeline=timeline.to_json()
            )
            await ctx.refresh()
        else:
            # Regions already known; rebuild the audio without paying for VAD
            # again.  This is the resume path.
            log.info(
                "reusing stored speech timeline",
                extra={"job_id": str(ctx.job_id), "regions": len(timeline.regions)},
            )

        if timeline.is_empty:
            # Reported honestly rather than as a hallucinated transcript over
            # silence.
            raise NoSpeechDetected(
                "no speech was detected in this audio",
                job_id=str(ctx.job_id),
                duration_s=ctx.job.duration_s,
            )

        ctx.timeline = timeline
        ctx.speech_wav = await ctx.extractor.write_speech_only(
            ctx.wav, timeline, ctx.job_scratch / SPEECH_WAV_NAME
        )

    async def _detect(self, ctx: JobContext) -> Timeline:
        if self._detector is None:
            return Timeline.whole(ctx.job.duration_s or 0.0)

        loop = asyncio.get_running_loop()
        # Silero is neither fast nor interruptible, so it shares the inference
        # executor with MLX — same single thread, same reason.
        timeline = await loop.run_in_executor(
            ctx.inference_executor, self._detector.speech_regions, ctx.wav
        )
        log.info(
            "speech detected",
            extra={
                "job_id": str(ctx.job_id),
                "regions": len(timeline.regions),
                "speech_s": round(timeline.speech_seconds, 2),
                "audio_s": ctx.job.duration_s,
            },
        )
        return timeline
