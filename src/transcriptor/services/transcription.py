"""TranscriptionService — the create path.

This is the only thing ``transcript_create`` does, and it must return a job id
in under 250ms without waiting for any work (rule N5).  So: parse, upsert the
source, claim atomically, wake the runner, return.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from transcriptor.domain.job import (
    ClaimResult,
    Created,
    JobStatus,
    ReuseReason,
    claimed_job_id,
    reuse_reason,
)
from transcriptor.domain.source import (
    CaptionPolicy,
    SourceKind,
    file_fingerprint,
    parse_source,
    result_key,
)
from transcriptor.domain.errors import AuthError, SourceNotFound
from transcriptor.ports.stores import JobStore

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """What a cheap look at a source can tell you, with no download."""

    title: str | None
    uploader: str | None
    duration_s: float | None
    is_live: bool
    has_manual_captions: bool | None
    has_auto_captions: bool | None
    auth_required: bool
    platform: str
    source_key: str
    estimated_transcribe_s: int | None
    existing: dict | None


@dataclass(frozen=True, slots=True)
class CreateResult:
    job_id: str
    status: str
    reused: bool
    reuse_reason: str | None
    estimated_seconds: int | None = None
    queued_reason: str | None = None


class TranscriptionService:
    def __init__(
        self,
        jobs: JobStore,
        *,
        model: str,
        wake: asyncio.Event | None = None,
        gateway=None,
        extractor=None,
        # Whisper large-v3 runs at roughly 8-12 minutes per hour of audio.
        realtime_factor: float = 0.17,
    ) -> None:
        self._jobs = jobs
        self._model = model
        self._wake = wake
        self._gateway = gateway
        self._extractor = extractor
        self._realtime_factor = realtime_factor

    async def probe(
        self,
        source: str,
        *,
        language: str | None = None,
        captions: CaptionPolicy = CaptionPolicy.MANUAL_ONLY,
    ) -> ProbeResult:
        """Cheap metadata, and what would happen if you asked for it.

        Passing the same ``language`` and ``captions`` you would use for
        ``create`` makes ``existing`` an exact ``result_key`` lookup, so it
        reports what *would* happen rather than what might.
        """
        ref = parse_source(source)
        fingerprint: str | None = None
        auth_required = False
        meta = None

        if ref.kind is SourceKind.FILE:
            path = Path(ref.key)
            if not path.exists():
                # Reported from `raw`, not from `key`. The key is case-folded
                # on Windows so that two spellings of one path dedupe to one
                # identity — correct for storage, wrong to read back to
                # someone, who typed `C:\Users\...` and would be told that
                # `c:\users\...` is missing and wonder what changed it.
                shown = Path(ref.raw).expanduser()
                raise SourceNotFound(f"no such file: {shown}", path=str(shown))
            fingerprint = file_fingerprint(path)
            if self._extractor is not None:
                meta = await self._extractor.probe(path)
        elif self._gateway is not None:
            try:
                meta = await self._gateway.probe(ref)
            except AuthError:
                auth_required = True

        key = result_key(
            identity=self._identity(ref, fingerprint),
            model=self._model,
            language=language,
            captions=captions,
        )
        existing = await self._jobs.find_by_result_key(key)

        duration = meta.duration_s if meta else None
        return ProbeResult(
            title=meta.title if meta else None,
            uploader=meta.uploader if meta else None,
            duration_s=duration,
            is_live=bool(meta.is_live) if meta else False,
            has_manual_captions=meta.has_manual_captions if meta else None,
            has_auto_captions=meta.has_auto_captions if meta else None,
            auth_required=auth_required,
            platform=ref.platform,
            source_key=ref.key,
            estimated_transcribe_s=(
                int(duration * self._realtime_factor) if duration else None
            ),
            existing=existing,
        )

    @staticmethod
    def _identity(ref, fingerprint: str | None) -> str:
        """A local file's path is not its identity — the file there may have
        been re-encoded or replaced.  Folding the fingerprint in means an edited
        file is treated as new content rather than served from cache."""
        return f"{ref.key}#{fingerprint}" if fingerprint else ref.key

    async def create(
        self,
        source: str,
        *,
        language: str | None = None,
        captions: CaptionPolicy = CaptionPolicy.MANUAL_ONLY,
        force: bool = False,
    ) -> CreateResult:
        """Claim a job and return immediately.  No work starts on this path."""
        ref = parse_source(source)

        fingerprint: str | None = None
        duration_s: float | None = None
        if ref.kind is SourceKind.FILE:
            path = Path(ref.key)
            if not path.exists():
                # Reported from `raw`, not from `key`. The key is case-folded
                # on Windows so that two spellings of one path dedupe to one
                # identity — correct for storage, wrong to read back to
                # someone, who typed `C:\Users\...` and would be told that
                # `c:\users\...` is missing and wonder what changed it.
                shown = Path(ref.raw).expanduser()
                raise SourceNotFound(f"no such file: {shown}", path=str(shown))
            fingerprint = file_fingerprint(path)

        source_id = await self._jobs.upsert_source(ref, fingerprint=fingerprint)

        identity = self._identity(ref, fingerprint)
        key = result_key(
            identity=identity,
            model=self._model,
            language=language,
            captions=captions,
        )

        claim = await self._jobs.claim_or_attach(
            key,
            source_id,
            model=self._model,
            language=language,
            captions=captions,
            fingerprint=fingerprint,
            force=force,
        )

        if isinstance(claim, Created) and self._wake is not None:
            # The Event handles the common case; the runner's 2s poll handles
            # anything it missed.
            self._wake.set()

        job = await self._jobs.get(claimed_job_id(claim))
        assert job is not None

        source_row = await self._jobs.get_source(source_id) or {}
        duration_s = source_row.get("duration_s")

        reason = reuse_reason(claim)
        return CreateResult(
            job_id=str(job.id),
            status=job.status.value,
            reused=not isinstance(claim, Created),
            reuse_reason=reason.value if reason else None,
            estimated_seconds=(
                int(duration_s * self._realtime_factor) if duration_s else None
            ),
        )
