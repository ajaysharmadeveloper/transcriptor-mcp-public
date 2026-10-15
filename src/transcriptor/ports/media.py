"""MediaGateway and AudioExtractor.

Both are async because both drive child processes — ffmpeg, ffprobe, yt-dlp —
which are started with ``create_subprocess_exec`` and never ``subprocess.run``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from transcriptor.domain.source import CaptionPolicy, SourceMeta, SourceRef
from transcriptor.domain.transcript import Segment, Timeline


@runtime_checkable
class AudioExtractor(Protocol):
    """ffprobe and ffmpeg.  Phase 1 uses this for local files."""

    async def probe(self, path: Path) -> SourceMeta:
        """Duration, codec, whether there is an audio track at all."""
        ...

    async def to_wav16k(self, src: Path, dest: Path, *, cancel=None) -> Path:
        """Decode to 16kHz mono WAV — what the engine expects.

        ``cancel`` kills the child process immediately: unlike inference, a
        subprocess does not have to run to completion.
        """
        ...

    async def write_speech_only(
        self, wav: Path, timeline: Timeline, dest: Path
    ) -> Path:
        """Concatenate the speech regions into a new WAV, so the engine never
        sees the silence it would hallucinate over."""
        ...

    async def audio_hash(self, wav: Path) -> str:
        """blake2b over the decoded audio.

        Identical regardless of container, bitrate or where the file came from,
        which is what makes it the layer-2 dedupe key.  Hashing tens of
        megabytes is not "fast enough" for the event loop, so this runs in the
        cpu executor.
        """
        ...


@runtime_checkable
class MediaGateway(Protocol):
    """URL ingestion via yt-dlp.  Phase 2 — declared now so the pipeline shape
    does not have to change when it lands."""

    async def probe(self, ref: SourceRef) -> SourceMeta: ...

    async def fetch_audio(self, ref: SourceRef, dest: Path, *, cancel=None) -> Path:
        """``bestaudio`` only — a 40-minute 4K video is ~30MB of audio rather
        than 2GB of video."""
        ...

    async def fetch_captions(
        self, ref: SourceRef, policy: CaptionPolicy
    ) -> list[Segment] | None:
        """Human captions are usually better than Whisper; auto-generated ones
        are worse.  The policy decides which are acceptable."""
        ...
