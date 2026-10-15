"""The error hierarchy.

One hierarchy, one mapping point, and **no error strings are interpreted
anywhere**.  Adapters raise typed errors; the pipeline catches
:class:`TranscriptorError` and writes ``error_code`` plus ``resumable`` onto the
job.  A bare :class:`Exception` becomes ``internal_error, resumable=False`` and
is logged with a traceback.

``resumable`` is a stored boolean rather than a lookup table because clients
must never parse messages or match on codes to decide whether to retry.  The
decision is made once, where the error is raised, by the code that actually
knows.
"""

from __future__ import annotations

from typing import Any


class TranscriptorError(Exception):
    """Base of the hierarchy.  Never raised directly."""

    code: str = "internal_error"
    resumable: bool = False

    def __init__(
        self,
        user_message: str,
        *,
        code: str | None = None,
        resumable: bool | None = None,
        **context: Any,
    ) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        if code is not None:
            self.code = code
        if resumable is not None:
            self.resumable = resumable
        self.context: dict[str, Any] = context

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(code={self.code!r}, resumable={self.resumable})"


class SourceError(TranscriptorError):
    """The source cannot be used.  Retrying the same input will not help."""

    code = "source_error"
    resumable = False


class AuthError(TranscriptorError):
    """The source is login-walled.  Cookies would fix it."""

    code = "auth_required"
    resumable = True


class ResourceError(TranscriptorError):
    """The machine could not spare what the job needed, right now."""

    code = "resource_error"
    resumable = True


class InfraError(TranscriptorError):
    """Something Transcriptor depends on failed, and may work on a retry."""

    code = "infra_error"
    resumable = True


class EngineError(TranscriptorError):
    """The ASR engine is unusable as configured."""

    code = "engine_error"
    resumable = False


# --------------------------------------------------------------------------
# Named errors from project-details §14.  Subclasses rather than bare `code=`
# arguments, so a caller can match on the type where that reads better than the
# string, and so the resumable decision is attached to the definition.


class SourceNotFound(SourceError):
    code = "source_not_found"


class RelativePathNotAllowed(SourceError):
    """A stdio server launched by a GUI client has no meaningful working
    directory, so guessing what a relative path means would be worse than
    refusing."""

    code = "relative_path_not_allowed"


class NoAudioStream(SourceError):
    code = "no_audio_stream"


class NoSpeechDetected(SourceError):
    """VAD found nothing.  Reported honestly rather than as a hallucinated
    transcript over silence."""

    code = "no_speech_detected"


class DurationExceedsLimit(SourceError):
    code = "duration_exceeds_limit"


class LiveStreamNotSupported(SourceError):
    code = "live_stream_not_supported"


class PlaylistNotSupported(SourceError):
    code = "playlist_not_supported"


class DrmProtected(SourceError):
    code = "drm_protected"


class UnsupportedSource(SourceError):
    """Phase 1 accepts local absolute paths only."""

    code = "unsupported_source"


class InsufficientMemory(ResourceError):
    code = "insufficient_memory"


class MemoryPressure(ResourceError):
    code = "memory_pressure"


class ScratchDiskFull(ResourceError):
    code = "scratch_disk_full"


class DatabaseUnreachable(InfraError):
    code = "database_unreachable"


class DatabaseLost(InfraError):
    code = "database_lost"


class MigrationsFailed(InfraError):
    """The server refuses to serve on a half-built schema."""

    code = "migrations_failed"


class ExtractorFailed(InfraError):
    code = "extractor_failed"


class ExtractionFailed(InfraError):
    """ffmpeg or ffprobe failed.  Its stderr is surfaced verbatim."""

    code = "extraction_failed"


class ModelNotDownloaded(EngineError):
    code = "model_not_downloaded"


class JobNotFound(SourceError):
    code = "job_not_found"


class IllegalTransition(TranscriptorError):
    """A status change the state machine does not permit.  A bug, not a
    condition — so it is not resumable."""

    code = "illegal_transition"
    resumable = False


class Cancelled(TranscriptorError):
    """The job was asked to stop, and it stopped at a chunk boundary.

    A honoured request rather than a failure, so the pipeline maps it to
    ``cancelled`` rather than ``failed``.  Everything committed before the
    boundary survives.
    """

    code = "cancelled"
    resumable = False
