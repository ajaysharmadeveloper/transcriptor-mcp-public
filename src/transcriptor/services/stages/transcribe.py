"""Transcribe — the only stage that can take hours.

Which is exactly why the chunk is the checkpoint unit: each chunk's segments are
committed as it completes, so an abort, a crash or a machine restart keeps
everything transcribed so far.  That single property is what makes partial
results readable, resume possible, and a memory abort survivable.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
import time

from transcriptor.domain.errors import (
    Cancelled,
    InsufficientMemory,
    MemoryPressure,
    NoSpeechDetected,
)
from transcriptor.domain.job import JobStatus
from transcriptor.domain.job import Stage as JobStage
from transcriptor.domain.pressure import Pressure
from transcriptor.domain.transcript import (
    AudioSlice,
    Chunk,
    ChunkStatus,
    Segment,
    Timeline,
    plan_chunks,
    stitch_overlap,
)
from transcriptor.services.pipeline import JobContext

log = logging.getLogger(__name__)


class TranscribeStage:
    name = "transcribe"
    job_stage = JobStage.TRANSCRIBING

    def __init__(self, *, governor=None) -> None:
        # Optional so the stage is testable without a machine to measure.
        self._governor = governor

    async def satisfied(self, ctx: JobContext) -> bool:
        if ctx.from_captions:
            return True
        chunks = await ctx.transcripts.chunks(ctx.job_id)
        return bool(chunks) and all(c.status is ChunkStatus.DONE for c in chunks)

    async def run(self, ctx: JobContext) -> None:
        # Everything here runs on the SPEECH-ONLY clock: chunk boundaries,
        # slices and the engine's output. Timings are mapped back onto the
        # original recording before anything is committed, so nothing outside
        # this method ever sees the compressed timeline.
        timeline = ctx.timeline or Timeline.whole(
            ctx.job.duration_s or (ctx.meta.duration_s if ctx.meta else 0.0)
        )
        audio_path = ctx.speech_wav or ctx.wav
        planned = plan_chunks(
            timeline.speech_seconds, chunk_s=ctx.chunk_s, overlap_s=ctx.overlap_s
        )
        await ctx.transcripts.plan_chunks(ctx.job_id, planned)

        existing = await ctx.transcripts.chunks(ctx.job_id)
        done = {c.idx for c in existing if c.status is ChunkStatus.DONE}

        # A resumed job's committed chunks already hold real text, so it must
        # not be reported as silent just because this pass transcribed nothing.
        produced_any = bool(done)
        next_idx = await ctx.transcripts.next_segment_idx(ctx.job_id)

        await self._ensure_loaded(ctx)
        loop = asyncio.get_running_loop()

        for chunk in planned:
            if chunk.idx in done:
                # A resumed job re-runs only what was never committed.
                log.info(
                    "chunk already committed",
                    extra={"job_id": str(ctx.job_id), "chunk": chunk.idx},
                )
                continue

            await self._at_boundary(ctx)

            slice_ = AudioSlice(
                path=audio_path, start_s=chunk.start_s, end_s=chunk.end_s
            )

            # The previous chunk's overlap, fetched on the ORIGINAL clock
            # because that is how it was committed.
            tail = await ctx.transcripts.segments_between(
                ctx.job_id,
                max(timeline.to_original(chunk.start_s) - 1.0, 0.0),
                timeline.to_original(chunk.end_s),
            )
            started = time.monotonic()

            # The one place inference happens, and it happens off the loop.
            # The executor is max_workers=1, so this *is* the concurrency
            # control — no semaphore to forget at a new call site (I5).
            raw = await loop.run_in_executor(
                ctx.inference_executor,
                functools.partial(
                    ctx.engine.transcribe,
                    slice_,
                    ctx.language,
                    prompt=_prompt_from(tail),
                ),
            )
            engine_ms = int((time.monotonic() - started) * 1000)
            ctx.engine_ms += engine_ms

            # Remove what the previous chunk's overlap already covered. Without
            # the overlap you lose words at every seam; without this you get
            # them twice.
            # Back onto the original clock before anything else looks at it —
            # stitching compares against committed segments, which are already
            # there.
            stitched = stitch_overlap(tail, timeline.remap(raw))
            if stitched.dropped or stitched.trimmed:
                log.info(
                    "overlap stitched",
                    extra={
                        "job_id": str(ctx.job_id),
                        "chunk": chunk.idx,
                        "dropped": stitched.dropped,
                        "trimmed": stitched.trimmed,
                        "matched_tokens": stitched.matched_tokens,
                    },
                )

            # Segment indices are per job and monotonic, so a resumed job
            # continues the numbering rather than colliding with what exists.
            renumbered = [
                Segment(
                    idx=next_idx + n,
                    start=s.start,
                    end=s.end,
                    text=s.text,
                    words=s.words,
                )
                for n, s in enumerate(stitched.segments)
            ]
            next_idx += len(renumbered)
            produced_any = produced_any or any(s.text.strip() for s in renumbered)

            await ctx.transcripts.commit_chunk(
                ctx.job_id,
                Chunk(
                    idx=chunk.idx,
                    # Recorded on the original clock, so `covered_until_s` and
                    # the progress percentage mean what a caller expects.
                    start_s=timeline.to_original(chunk.start_s),
                    end_s=timeline.to_original(chunk.end_s),
                    status=ChunkStatus.DONE,
                    attempts=1,
                    engine_ms=engine_ms,
                ),
                renumbered,
            )
            await ctx.jobs.heartbeat(ctx.job_id)

        if not produced_any:
            # Reported honestly rather than as a hallucinated transcript over
            # silence.
            raise NoSpeechDetected(
                "no speech was found in this audio", job_id=str(ctx.job_id)
            )

    # ------------------------------------------------------------- boundary

    async def _at_boundary(self, ctx: JobContext) -> None:
        """Cancellation and the escalation ladder, checked between chunks.

        The ladder's second consumer: admission decided whether to *begin*, this
        decides how to behave now that we are running.  Conflating them is how
        systems start a job they immediately have to abort.

            elevated          clear the MLX buffer cache, carry on
            critical          pause, up to `pause_max_s`, waiting for relief
            critical, expired  commit what is done, evict, fail(resumable)

        Never `kill -9`, never `setrlimit`, never crash.  Every rung is
        recoverable.
        """
        if ctx.cancel.cancelled:
            raise Cancelled(ctx.cancel.reason or "cancelled")

        if self._governor is None:
            return

        pressure = self._governor.refresh().pressure
        if pressure is Pressure.NORMAL:
            return

        if pressure is Pressure.ELEVATED:
            # Give the buffer cache back between chunks rather than letting
            # resident memory drift upwards across a long job.
            self._clear_cache(ctx)
            return

        await self._pause_for_relief(ctx)

    async def _pause_for_relief(self, ctx: JobContext) -> None:
        """Hold the job under critical pressure, then abort gracefully.

        Pausing is not a stall: the machine is being given the chance to recover
        on its own, which on a shared desktop it usually does within seconds —
        Chrome finishes a render, a build ends.  Aborting immediately would
        throw away a job that only needed to wait.
        """
        self._clear_cache(ctx)
        sample = self._governor.snapshot.sample

        await ctx.jobs.transition(
            ctx.job_id,
            JobStatus.PAUSED,
            note=f"memory_pressure: {sample.available_gb:.1f}GB available",
        )
        log.warning(
            "paused under critical memory pressure",
            extra={
                "job_id": str(ctx.job_id),
                "available_gb": round(sample.available_gb, 2),
                "swap_delta_gb": round(sample.swap_delta_gb, 2),
                "waiting_up_to_s": ctx.pause_max_s,
            },
        )

        waited = 0.0
        while waited < ctx.pause_max_s:
            if ctx.cancel.cancelled:
                raise Cancelled(ctx.cancel.reason or "cancelled while paused")

            await asyncio.sleep(ctx.pressure_poll_s)
            waited += ctx.pressure_poll_s
            await ctx.jobs.heartbeat(ctx.job_id)

            if self._governor.refresh().pressure is not Pressure.CRITICAL:
                await ctx.jobs.transition(
                    ctx.job_id,
                    JobStatus.RUNNING,
                    stage=self.job_stage,
                    note=f"pressure cleared after {waited:.0f}s",
                )
                log.info(
                    "resumed after pressure cleared",
                    extra={"job_id": str(ctx.job_id), "paused_s": waited},
                )
                return

        # Past the limit: give up gracefully. Everything committed so far
        # survives and the job is resumable, which turns the worst-case memory
        # outcome from "lost 40 minutes of work" into "finish it later".
        await self._evict(ctx)
        sample = self._governor.snapshot.sample
        raise MemoryPressure(
            f"memory pressure stayed critical for {ctx.pause_max_s:.0f}s: "
            f"{sample.available_gb:.1f}GB available, swap grew "
            f"{sample.swap_delta_gb:.2f}GB — stopped with the transcript "
            f"committed so far",
            available_gb=sample.available_gb,
            swap_delta_gb=sample.swap_delta_gb,
            paused_s=ctx.pause_max_s,
        )

    @staticmethod
    async def _evict(ctx: JobContext) -> None:
        """Release the model immediately, ignoring the idle timer.

        On abort the machine needs the memory now, not in ten minutes.
        """
        try:
            await ctx.engine.unload()
        except Exception:  # noqa: BLE001 - eviction must not mask the abort
            log.exception("could not evict the model", extra={"job_id": str(ctx.job_id)})

    @staticmethod
    def _clear_cache(ctx: JobContext) -> None:
        clear = getattr(ctx.engine, "clear_cache", None)
        if clear is not None:
            clear()

    async def _ensure_loaded(self, ctx: JobContext) -> None:
        if ctx.engine.loaded:
            return
        vram_gb = getattr(ctx.engine, "vram_estimate_gb", 0.0)
        if self._governor is not None and not self._governor.may_load_model(
            ctx.engine.memory_estimate_gb, vram_gb
        ):
            snapshot = self._governor.snapshot
            raise InsufficientMemory(
                f"not enough memory to load {ctx.engine.name}: "
                f"{snapshot.sample.available_gb:.1f}GB available, needs "
                f"{ctx.engine.memory_estimate_gb:.1f}GB plus the system floor",
                available_gb=snapshot.sample.available_gb,
                needs_gb=ctx.engine.memory_estimate_gb,
            )
        await ctx.engine.load()


# One sentence, capped. Measured on the 77s fixture at the 50s seam:
#
#   prompt          words  capitals  full stops  echoed 5-grams
#   none               72         0           0               0
#   220 chars          73        15           8               9   <- echoed back
#   last sentence      72        15           8               0
#   ~90 chars          61        15           6               0   <- lost 11 words
#
# A long prompt buys nothing extra and makes Whisper regurgitate it into the
# transcript, where the echo lands past the seam and so escapes stitching. One
# sentence re-establishes casing and punctuation with none of that.
_PROMPT_MAX_CHARS = 120

# Sentence terminators across the scripts this has to work in: Latin, the
# Devanagari danda, and the CJK/Arabic full stops.
_SENTENCE_END = re.compile(r"[.!?。！？।؟]+[\s\"'”’)\]]*")


def _prompt_from(previous: list[Segment]) -> str | None:
    """The last sentence of what is already committed, for the next chunk to
    continue from.

    Without it Whisper starts each chunk cold and drops casing and punctuation
    for the rest of that chunk — which on a long file means most of the
    transcript.
    """
    if not previous:
        return None
    text = " ".join(s.text.strip() for s in previous if s.text.strip()).strip()
    if not text:
        return None

    boundaries = [m.end() for m in _SENTENCE_END.finditer(text)]
    # The last boundary that still leaves a sentence after it; otherwise the
    # whole tail, which is what an unpunctuated language or a first chunk gives.
    starts = [b for b in boundaries if b < len(text)]
    sentence = text[starts[-1] :] if starts else text
    return sentence[-_PROMPT_MAX_CHARS:].strip() or None
