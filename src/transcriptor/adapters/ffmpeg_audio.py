"""FfmpegAudio — ffprobe, ffmpeg, and the audio hash.

Every child process is started with ``create_subprocess_exec`` (rule R3), so
nothing here blocks the event loop.  ffmpeg's stderr is surfaced verbatim on
failure rather than being reduced to a generic message: for a codec problem the
real stderr is the useful part.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import wave
from concurrent.futures import Executor
from pathlib import Path

from transcriptor.adapters._proc import killer as _killer, subprocess_flags
from transcriptor.domain.errors import (
    Cancelled,
    ExtractionFailed,
    NoAudioStream,
    ScratchDiskFull,
    SourceNotFound,
)
from transcriptor.domain.source import SourceMeta
from transcriptor.domain.transcript import Timeline

_HASH_CHUNK = 1 << 20  # 1MiB


class FfmpegAudio:
    """Implements :class:`transcriptor.ports.media.AudioExtractor`."""

    def __init__(
        self,
        *,
        cpu_executor: Executor | None = None,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
    ) -> None:
        self._cpu = cpu_executor
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe

    # ---------------------------------------------------------------- probe

    async def probe(self, path: Path) -> SourceMeta:
        if not path.exists():
            raise SourceNotFound(f"no such file: {path}", path=str(path))
        if shutil.which(self._ffprobe) is None:
            raise ExtractionFailed(
                "ffprobe is not on PATH — install it with `brew install ffmpeg`"
            )

        out = await self._run(
            None,
            self._ffprobe,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        )
        data = json.loads(out)

        audio = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None
        )
        if audio is None:
            raise NoAudioStream(f"no audio track in {path}", path=str(path))

        fmt = data.get("format", {})
        duration = fmt.get("duration") or audio.get("duration")
        if duration is None:
            raise ExtractionFailed(
                f"ffprobe reported no duration for {path}", path=str(path)
            )

        return SourceMeta(
            duration_s=float(duration),
            has_audio=True,
            container=fmt.get("format_name"),
            audio_codec=audio.get("codec_name"),
            sample_rate=int(audio["sample_rate"]) if audio.get("sample_rate") else None,
            channels=audio.get("channels"),
            title=(fmt.get("tags") or {}).get("title"),
        )

    # -------------------------------------------------------------- extract

    async def to_wav16k(self, src: Path, dest: Path, *, cancel=None) -> Path:
        """Decode to 16kHz mono signed-16 PCM — what Whisper expects."""
        if not src.exists():
            raise SourceNotFound(f"no such file: {src}", path=str(src))

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            await self._run(
                cancel,
                self._ffmpeg,
                "-nostdin",
                "-loglevel", "error",
                "-y",
                "-i", str(src),
                "-vn",                    # audio only; never decode the video
                "-ac", "1",
                "-ar", "16000",
                "-acodec", "pcm_s16le",
                "-f", "wav",
                str(dest),
            )
        except ExtractionFailed as exc:
            stderr = str(exc)
            if "No space left on device" in stderr:
                raise ScratchDiskFull(
                    f"no space left while writing {dest.parent}", path=str(dest.parent)
                ) from exc
            raise

        if not dest.exists() or dest.stat().st_size == 0:
            raise ExtractionFailed(
                f"ffmpeg produced no output at {dest}", path=str(dest)
            )
        return dest

    async def write_speech_only(
        self, wav: Path, timeline: Timeline, dest: Path
    ) -> Path:
        """Concatenate the speech regions into a new 16kHz mono WAV.

        Whisper then never sees the silence it would otherwise hallucinate over,
        and :meth:`Timeline.to_original` maps the resulting timings back.

        Done with the stdlib ``wave`` module rather than an ffmpeg filtergraph:
        a hundred regions would make a filter string thousands of characters
        long, and this is a straight byte copy of PCM frames.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._cpu, _write_speech_only, wav, timeline, dest
        )

    # ----------------------------------------------------------------- hash

    async def audio_hash(self, wav: Path) -> str:
        """blake2b over the decoded audio, off the event loop.

        Hashing a 40MB file is not "fast enough" for the loop (rule R2), so it
        goes to the cpu executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._cpu, _blake2b_file, wav)

    # ------------------------------------------------------------- internal

    async def _run(self, cancel, program: str, *args: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            program,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Keeps a console window from flashing up per child on Windows.
            **subprocess_flags(),
        )

        # A child process stops the moment it is killed, so cancellation here is
        # immediate rather than waiting for a chunk boundary.
        killer = _killer(proc)
        if cancel is not None:
            cancel.on_cancel(killer)
        try:
            stdout, stderr = await proc.communicate()
        finally:
            if cancel is not None:
                cancel.forget(killer)

        if cancel is not None and cancel.cancelled:
            raise Cancelled(cancel.reason or "cancelled", program=program)

        if proc.returncode != 0:
            # Surfaced verbatim: for a codec problem the real stderr is the
            # useful part, and paraphrasing it loses that.
            raise ExtractionFailed(
                stderr.decode(errors="replace").strip()
                or f"{program} exited with {proc.returncode}",
                program=program,
                returncode=proc.returncode,
            )
        return stdout.decode(errors="replace")


def _blake2b_file(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    with path.open("rb") as fh:
        while block := fh.read(_HASH_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _write_speech_only(wav: Path, timeline: Timeline, dest: Path) -> Path:
    """Copy just the speech frames.  Runs on the cpu executor."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav), "rb") as source:
        rate = source.getframerate()
        width = source.getsampwidth()
        channels = source.getnchannels()
        total = source.getnframes()

        with wave.open(str(dest), "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(width)
            out.setframerate(rate)
            for start_s, end_s in timeline.regions:
                start = max(int(start_s * rate), 0)
                end = min(int(end_s * rate), total)
                if end <= start:
                    continue
                source.setpos(start)
                out.writeframes(source.readframes(end - start))
    return dest

