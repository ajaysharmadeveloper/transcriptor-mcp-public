"""Cancel, delete, and export — the paths that change or leave the system.

Kept apart from :mod:`reads` because these are the only read-adjacent
operations that are not read-only, and one of them is the single destructive
tool in the set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from transcriptor.domain.errors import JobNotFound, TranscriptorError
from transcriptor.domain.job import JobStatus
from transcriptor.ports.files import FileWriter
from transcriptor.ports.stores import JobStore, TranscriptStore
from transcriptor.services.reads import parse_format, parse_job_id

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExportResult:
    path: str
    bytes: int
    format: str


@dataclass(frozen=True, slots=True)
class DeleteResult:
    deleted: bool
    freed_bytes: int


class LifecycleService:
    def __init__(
        self,
        jobs: JobStore,
        transcripts: TranscriptStore,
        *,
        files: FileWriter,
        scratch_dir: Path,
        runner=None,
        transcription=None,
    ) -> None:
        self._jobs = jobs
        self._transcripts = transcripts
        self._files = files
        self._scratch_dir = scratch_dir
        self._runner = runner
        self._transcription = transcription

    # ---------------------------------------------------------------- batch

    async def batch(
        self,
        folder: str,
        *,
        pattern: str = "*",
        language: str | None = None,
        limit: int = 100,
    ) -> dict:
        """Queue every media file in a folder.

        One job per file, created through exactly the same path as a single
        request — so dedupe, resume and the active-job index all behave
        identically. Files already transcribed under these settings come back as
        reused rather than being queued again.

        Not recursive: a folder is a folder. Recursing would silently pull in a
        backup directory nobody meant to transcribe.
        """
        directory = Path(folder).expanduser()
        if not directory.is_absolute():
            raise TranscriptorError(
                "folder must be an absolute path",
                code="relative_path_not_allowed",
                resumable=False,
                folder=folder,
            )
        if self._transcription is None:
            raise TranscriptorError(
                "batch mode is not wired up",
                code="internal_error",
                resumable=False,
            )

        found = await self._files.list_media(directory, pattern)
        queued: list[dict] = []
        skipped: list[dict] = []

        for path in found[:limit]:
            try:
                result = await self._transcription.create(
                    str(path), language=language
                )
            except TranscriptorError as exc:
                skipped.append(
                    {"source": str(path), "error_code": exc.code,
                     "error": exc.user_message}
                )
                continue
            queued.append(
                {
                    "source": str(path),
                    "job_id": result.job_id,
                    "status": result.status,
                    "reused": result.reused,
                    "reuse_reason": result.reuse_reason,
                }
            )

        return {
            "folder": str(directory),
            "matched": len(found),
            "queued": queued,
            "skipped": skipped,
            "truncated": len(found) > limit,
        }

    # --------------------------------------------------------------- cancel

    async def cancel(self, job_id: str) -> dict:
        """Ask a job to stop.

        A queued job stops at once.  A running one stops at the next chunk
        boundary — inference in a thread cannot be interrupted mid-call, so
        cancelling costs up to one chunk, about two minutes.  Everything already
        committed survives either way.
        """
        uid = parse_job_id(job_id)
        job = await self._jobs.get(uid)
        if job is None:
            raise JobNotFound(f"no such job: {job_id}", job_id=job_id)

        if job.status.is_terminal:
            return {
                "cancelled": False,
                "status": job.status.value,
                "detail": f"already {job.status.value}",
            }

        signalled = False
        if self._runner is not None:
            signalled = self._runner.request_cancel(uid)

        if not signalled:
            # Not the job in flight, so it is queued or paused: stop it here.
            await self._jobs.transition(
                uid, JobStatus.CANCELLED, note="cancelled by request"
            )
            return {"cancelled": True, "status": "cancelled", "detail": "stopped"}

        return {
            "cancelled": True,
            "status": job.status.value,
            "detail": "stopping at the next chunk boundary, up to ~2 minutes",
        }

    # --------------------------------------------------------------- delete

    async def delete(self, job_id: str, *, delete_media: bool = False) -> DeleteResult:
        """Remove a transcript.  The one destructive tool.

        Refuses while the job is in flight rather than deleting rows another
        task is still writing — cancel it first.
        """
        uid = parse_job_id(job_id)
        job = await self._jobs.get(uid)
        if job is None:
            raise JobNotFound(f"no such job: {job_id}", job_id=job_id)

        if job.status is JobStatus.RUNNING:
            raise TranscriptorError(
                "that job is still running — cancel it first, then delete",
                code="job_in_flight",
                resumable=False,
                job_id=job_id,
            )

        freed = 0
        if delete_media:
            freed = await self._files.remove_tree(self._scratch_dir / str(uid))

        # segments, chunks, job_events and renders all cascade from jobs.
        removed = await self._jobs.delete(uid)
        log.info(
            "job deleted",
            extra={"job_id": job_id, "freed_bytes": freed, "media": delete_media},
        )
        return DeleteResult(deleted=removed, freed_bytes=freed)

    # --------------------------------------------------------------- export

    async def export(self, job_id: str, *, fmt: str, dest_path: str) -> ExportResult:
        """Write a format to a path the caller names.

        The single place Transcriptor writes output to disk, and only because
        someone asked. Absolute paths only, for the same reason sources are:
        a stdio server launched by a GUI client has no meaningful working
        directory.
        """
        uid = parse_job_id(job_id)
        if await self._jobs.get(uid) is None:
            raise JobNotFound(f"no such job: {job_id}", job_id=job_id)

        destination = Path(dest_path).expanduser()
        if not destination.is_absolute():
            raise TranscriptorError(
                "dest_path must be absolute — a stdio server has no meaningful "
                "working directory to resolve a relative path against",
                code="relative_path_not_allowed",
                resumable=False,
                dest_path=dest_path,
            )

        chosen = parse_format(fmt)
        content = await self._transcripts.render(uid, chosen)
        written = await self._files.write_text(destination, content)
        return ExportResult(path=str(destination), bytes=written, format=chosen.value)
