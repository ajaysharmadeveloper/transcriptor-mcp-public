"""Job, JobStatus, Stage, and the transition table.

The state machine lives here, in pure code, so every illegal transition is
rejected in one place and can be tested without a database.  Terminal states
change only through resume (invariant I6).
"""

from __future__ import annotations

import datetime as dt
import enum
import logging
import uuid
from contextlib import suppress
from dataclasses import dataclass, field

from transcriptor.domain.errors import IllegalTransition
from transcriptor.domain.transcript import Timeline


class JobStatus(str, enum.Enum):
    """Where the job stands.  Seven values, and nothing else is valid."""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL

    @property
    def is_active(self) -> bool:
        """Occupies the one active slot for a result_key.

        Must stay in step with the ``jobs_one_active_idx`` predicate.
        """
        return self in _ACTIVE


class Stage(str, enum.Enum):
    """The pipeline step.  Meaningful only while ``running``."""

    RESOLVING = "resolving"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    VAD = "vad"
    TRANSCRIBING = "transcribing"
    RENDERING = "rendering"


class Origin(str, enum.Enum):
    """What actually produced the transcript.

    Reported on every result so the caller never has to infer it from quality.
    """

    ASR = "asr"
    CAPTIONS_MANUAL = "captions_manual"
    CAPTIONS_AUTO = "captions_auto"


_TERMINAL = frozenset(
    {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED}
)

# Mirrors the partial unique index `jobs_one_active_idx`.
_ACTIVE = frozenset({JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSED})

# The legal transitions, from project-details §13.  Anything absent is a bug in
# the caller, not a condition to handle.
TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.CANCELLED}),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.DONE,
            JobStatus.PAUSED,
            JobStatus.FAILED,
            JobStatus.INTERRUPTED,
            JobStatus.CANCELLED,
        }
    ),
    JobStatus.PAUSED: frozenset(
        {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.INTERRUPTED, JobStatus.CANCELLED}
    ),
    # Terminal states change only via resume, which moves the job back to
    # `queued` on the same row.
    JobStatus.DONE: frozenset(),
    JobStatus.FAILED: frozenset({JobStatus.QUEUED}),
    JobStatus.INTERRUPTED: frozenset({JobStatus.QUEUED}),
    JobStatus.CANCELLED: frozenset(),
}


def may_transition(frm: JobStatus, to: JobStatus) -> bool:
    return to in TRANSITIONS[frm]


def assert_transition(frm: JobStatus, to: JobStatus) -> None:
    """Raise :class:`IllegalTransition` unless the move is legal."""
    if not may_transition(frm, to):
        raise IllegalTransition(
            f"cannot move a job from {frm.value} to {to.value}",
            from_status=frm.value,
            to_status=to.value,
        )


def is_resume(frm: JobStatus, to: JobStatus) -> bool:
    """A resume is the only way out of a terminal state."""
    return frm in (JobStatus.FAILED, JobStatus.INTERRUPTED) and to is JobStatus.QUEUED


@dataclass(slots=True)
class Job:
    """One transcription attempt.  A row in ``jobs``."""

    id: uuid.UUID
    source_id: uuid.UUID
    result_key: str
    status: JobStatus
    stage: Stage | None = None
    origin: Origin | None = None
    resumable: bool = False
    forced: bool = False
    heartbeat_at: dt.datetime | None = None
    progress_pct: float = 0.0
    partial: bool = False
    covered_until_s: float = 0.0
    last_chunk_idx: int = -1
    error: str | None = None
    error_code: str | None = None
    language: str | None = None
    requested_language: str | None = None
    model: str | None = None
    duration_s: float | None = None
    word_count: int | None = None
    transcribe_ms: int | None = None
    speech_timeline: Timeline | None = None
    created_at: dt.datetime | None = None
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def is_active(self) -> bool:
        return self.status.is_active


@dataclass(frozen=True, slots=True)
class JobEvent:
    """One row of the transition log.

    ``job_events`` gives the audit trail without making current state a fold
    over history.
    """

    job_id: uuid.UUID
    to_status: JobStatus
    from_status: JobStatus | None = None
    stage: Stage | None = None
    note: str | None = None
    at: dt.datetime | None = None


# --------------------------------------------------------------------------
# Claim results.  A sum type rather than `Optional[UUID]`, so the caller cannot
# forget to handle a dedupe case.


class ReuseReason(str, enum.Enum):
    EXACT_SOURCE = "exact_source"      # layer 1 — same URL or unchanged file
    SAME_AUDIO = "same_audio"          # layer 2 — different source, identical audio
    ALREADY_RUNNING = "already_running"  # layer 3 — attached to a job in flight
    RESUMED = "resumed"                # an interrupted job picked up mid-way


@dataclass(frozen=True, slots=True)
class Created:
    """A new job was inserted.  Work is about to start."""

    job_id: uuid.UUID


@dataclass(frozen=True, slots=True)
class Attached:
    """An active job already owned this result_key."""

    job_id: uuid.UUID
    reason: ReuseReason = ReuseReason.ALREADY_RUNNING


@dataclass(frozen=True, slots=True)
class Reused:
    """A completed or resumable job answered the request."""

    job_id: uuid.UUID
    reason: ReuseReason


ClaimResult = Created | Attached | Reused


def claimed_job_id(result: ClaimResult) -> uuid.UUID:
    return result.job_id


def reuse_reason(result: ClaimResult) -> ReuseReason | None:
    return None if isinstance(result, Created) else result.reason


@dataclass(slots=True)
class CancelToken:
    """Cancellation, cooperative for inference and immediate for subprocesses.

    Inference happens in a thread and cannot be interrupted mid-call, so it is
    checked at chunk boundaries.  Known bound, stated rather than hidden:
    cancelling a transcription costs up to one chunk — about two minutes.
    Shortening it means shorter chunks, which costs accuracy at the seams; two
    minutes is the accepted trade.

    ffmpeg and yt-dlp are different: they are child processes, so they stop the
    moment they are killed.  Adapters register a killer with :meth:`on_cancel`
    and it fires immediately — a download does not have to run to completion
    just because someone changed their mind.
    """

    _cancelled: bool = False
    reason: str | None = None
    _on_cancel: list = field(default_factory=list)

    def cancel(self, reason: str = "cancelled by request") -> None:
        self._cancelled = True
        self.reason = reason
        for callback in list(self._on_cancel):
            _fire(callback)
        self._on_cancel.clear()

    def on_cancel(self, callback) -> None:
        """Run ``callback`` when cancelled — or now, if that already happened."""
        if self._cancelled:
            _fire(callback)
            return
        self._on_cancel.append(callback)

    def forget(self, callback) -> None:
        """Stop tracking a callback whose subject has finished normally."""
        with suppress(ValueError):
            self._on_cancel.remove(callback)

    @property
    def cancelled(self) -> bool:
        return self._cancelled


def _fire(callback) -> None:
    """A killer that fails must not stop the others running."""
    try:
        callback()
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("cancellation callback failed")
