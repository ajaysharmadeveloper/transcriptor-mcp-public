"""Pipeline, the Stage protocol, and resume.

Execution is: for each stage, skip if ``satisfied``, else run.  **Resume is not
a special code path** — it is the same loop over a job whose early stages
already report satisfied.  A resumed job re-enters at the first unsatisfied
stage with no branching, which is why resume is likely to actually work rather
than being the untested path it usually is.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from transcriptor.domain.errors import Cancelled, TranscriptorError
from transcriptor.domain.job import CancelToken, Job, JobStatus, Origin
from transcriptor.domain.job import Stage as JobStage
from transcriptor.domain.source import CaptionPolicy, SourceMeta, SourceRef
from transcriptor.domain.transcript import Timeline
from transcriptor.ports.engine import ASREngine
from transcriptor.ports.media import AudioExtractor
from transcriptor.ports.stores import JobStore, TranscriptStore

log = logging.getLogger(__name__)


@dataclass
class JobContext:
    """Everything a stage needs, and the working state it hands on.

    Stages communicate through this rather than through return values, so a
    stage that is skipped costs nothing and the ones after it still find what
    they need — by re-reading the checkpoint the skipped stage would have
    written.
    """

    job: Job
    jobs: JobStore
    transcripts: TranscriptStore
    engine: ASREngine
    extractor: AudioExtractor
    inference_executor: Executor
    cpu_executor: Executor
    scratch_dir: Path
    max_duration_s: int
    language: str | None = None
    chunk_s: float = 120.0
    overlap_s: float = 15.0
    # How long a job may sit paused under critical pressure before it gives up
    # and commits what it has.
    pause_max_s: float = 60.0
    pressure_poll_s: float = 2.0
    # Cooperative cancellation, honoured at chunk boundaries.
    cancel: CancelToken = field(default_factory=CancelToken)

    # Working state, filled in as stages run.
    captions: CaptionPolicy | None = None

    source: dict[str, Any] = field(default_factory=dict)
    ref: SourceRef | None = None
    # The local media file audio is extracted from: the source itself for a
    # file, the downloaded bestaudio for a URL.
    media_path: Path | None = None
    meta: SourceMeta | None = None
    wav: Path | None = None
    # Speech regions on the original clock, and the audio with the silence
    # removed.  Chunking and inference happen on the speech-only clock; every
    # timestamp is mapped back through `timeline` before it is committed.
    timeline: Timeline | None = None
    speech_wav: Path | None = None
    engine_ms: int = 0

    @property
    def job_id(self) -> uuid.UUID:
        return self.job.id

    @property
    def job_scratch(self) -> Path:
        return self.scratch_dir / str(self.job.id)

    @property
    def source_path(self) -> Path | None:
        key = self.source.get("source_key")
        return Path(key) if key and self.source.get("kind") == "file" else None

    @property
    def from_captions(self) -> bool:
        """Captions answered this job, so there is nothing to decode or infer.

        Read from the job row rather than a flag, so it survives a restart and
        a resumed job still skips the same stages.
        """
        return self.job.origin in (Origin.CAPTIONS_MANUAL, Origin.CAPTIONS_AUTO)

    async def refresh(self) -> None:
        """Re-read the job row.  Stages checkpoint into it, so later stages must
        not reason about a stale copy."""
        fresh = await self.jobs.get(self.job.id)
        if fresh is not None:
            self.job = fresh


@runtime_checkable
class Stage(Protocol):
    """One step, with an idempotency probe."""

    name: str
    job_stage: JobStage

    async def satisfied(self, ctx: JobContext) -> bool:
        """Has this stage's checkpoint already been written?"""
        ...

    async def run(self, ctx: JobContext) -> None: ...


class Pipeline:
    def __init__(self, stages: list[Stage]) -> None:
        self._stages = stages

    @property
    def stages(self) -> list[Stage]:
        return list(self._stages)

    async def execute(self, ctx: JobContext) -> JobStatus:
        """Run the job to a terminal state.  Never raises for job failure —
        a failed job is a recorded outcome, not an exception for the caller.

        ``asyncio.CancelledError`` is deliberately not caught: shutdown must not
        be mistaken for a job failure.
        """
        started = time.monotonic()
        try:
            for stage in self._stages:
                await ctx.refresh()
                if await stage.satisfied(ctx):
                    log.info(
                        "stage skipped (already satisfied)",
                        extra={"job_id": str(ctx.job_id), "stage": stage.name},
                    )
                    continue
                await ctx.jobs.set_stage(ctx.job_id, stage.job_stage)
                log.info(
                    "stage running",
                    extra={"job_id": str(ctx.job_id), "stage": stage.name},
                )
                await stage.run(ctx)

            await ctx.jobs.transition(
                ctx.job_id,
                JobStatus.DONE,
                note="pipeline complete",
                progress_pct=100.0,
                partial=False,
                # Only claim ASR if nothing else already said where this came
                # from — the captions shortcut sets it in Fetch.
                origin=(ctx.job.origin or Origin.ASR).value,
                transcribe_ms=int((time.monotonic() - started) * 1000),
            )
            return JobStatus.DONE

        except Cancelled as exc:
            # A honoured request, not a failure. Everything committed so far
            # stays; cancelling costs at most one chunk.
            log.info("job cancelled", extra={"job_id": str(ctx.job_id)})
            await ctx.jobs.transition(
                ctx.job_id, JobStatus.CANCELLED, note=str(exc) or "cancelled"
            )
            return JobStatus.CANCELLED

        except TranscriptorError as exc:
            # The error decided its own code and resumability where it was
            # raised, by the code that actually knew.
            log.warning(
                "job failed",
                extra={
                    "job_id": str(ctx.job_id),
                    "error_code": exc.code,
                    "resumable": exc.resumable,
                    **exc.context,
                },
            )
            await ctx.jobs.transition(
                ctx.job_id,
                JobStatus.FAILED,
                note=exc.code,
                error=exc.user_message,
                error_code=exc.code,
                resumable=exc.resumable,
            )
            return JobStatus.FAILED

        except Exception as exc:  # noqa: BLE001 - mapped once, logged with traceback
            log.exception("job failed unexpectedly", extra={"job_id": str(ctx.job_id)})
            await ctx.jobs.transition(
                ctx.job_id,
                JobStatus.FAILED,
                note="internal_error",
                error=f"{type(exc).__name__}: {exc}",
                error_code="internal_error",
                resumable=False,
            )
            return JobStatus.FAILED

    @staticmethod
    def cleanup(ctx: JobContext, *, keep: bool = False) -> None:
        """Remove the job's scratch directory.

        Nothing durable lives there — model weights and in-flight audio only —
        so it goes when the job ends, and is wiped again at startup so a crash
        cannot leave orphans behind.
        """
        if keep:
            return
        shutil.rmtree(ctx.job_scratch, ignore_errors=True)
