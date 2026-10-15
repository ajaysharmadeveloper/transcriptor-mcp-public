"""ClipService — turn a transcript into cuttable moments.

Phase 5.  Emits, per clip: the ffmpeg command to cut it, the command to crop it
to 9:16, and a burned-caption SRT rebased to the clip's own clock.

Commands are *returned, never run*.  Transcriptor's job is to know where the
good moments are; deciding to write a video file is the caller's, and running
ffmpeg over someone's media without being asked would be well outside what a
transcription server should do.
"""

from __future__ import annotations

import dataclasses
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

from transcriptor.domain.clips import Clip, find_clips
from transcriptor.domain.errors import JobNotFound, TranscriptorError
from transcriptor.ports.renderer import SubtitleRenderer
from transcriptor.ports.stores import JobStore, TranscriptStore
from transcriptor.services.reads import parse_job_id

log = logging.getLogger(__name__)

# 1080x1920 is the shape every vertical platform wants.
VERTICAL_WIDTH, VERTICAL_HEIGHT = 1080, 1920


@dataclass(frozen=True, slots=True)
class ClipPlan:
    index: int
    start: float
    end: float
    duration: float
    words: int
    density: float
    score: float
    text: str
    srt: str
    cut_command: str
    vertical_command: str
    burn_command: str


class ClipService:
    def __init__(
        self,
        jobs: JobStore,
        transcripts: TranscriptStore,
        *,
        renderer: SubtitleRenderer,
    ) -> None:
        self._jobs = jobs
        self._transcripts = transcripts
        self._renderer = renderer

    async def find(
        self,
        job_id: str,
        *,
        count: int = 5,
        min_s: float = 15.0,
        max_s: float = 60.0,
        dest_dir: str | None = None,
    ) -> dict:
        uid = parse_job_id(job_id)
        job = await self._jobs.get(uid)
        if job is None:
            raise JobNotFound(f"no such job: {job_id}", job_id=job_id)
        if max_s < min_s:
            raise TranscriptorError(
                f"max_s ({max_s}) is below min_s ({min_s})",
                code="invalid_parameter",
                resumable=False,
            )

        source = await self._jobs.get_source(job.source_id) or {}
        media = source.get("source_key", "")
        segments = await self._transcripts.segments(uid)
        if not segments:
            return {"job_id": job_id, "source": media, "clips": [], "detail":
                    "no transcript yet — clips are found in the words, not the audio"}

        found = find_clips(segments, count=count, min_s=min_s, max_s=max_s)
        out = Path(dest_dir).expanduser() if dest_dir else Path("./clips")

        plans = [
            self._plan(
                n, clip, media, out,
                self._renderer.clip_srt(segments, clip.start, clip.end),
            )
            for n, clip in enumerate(found, start=1)
        ]
        log.info(
            "clips found",
            extra={"job_id": job_id, "clips": len(plans), "segments": len(segments)},
        )
        return {
            "job_id": job_id,
            "source": media,
            "clips": [dataclasses.asdict(p) for p in plans],
            "detail": (
                "commands are returned, not run — nothing is written until you "
                "run them"
            ),
        }

    @staticmethod
    def _plan(index: int, clip: Clip, media: str, out: Path, srt: str) -> ClipPlan:
        stem = f"clip{index:02d}"
        cut = out / f"{stem}.mp4"
        vertical = out / f"{stem}-9x16.mp4"
        burned = out / f"{stem}-captioned.mp4"
        subs = out / f"{stem}.srt"

        q = shlex.quote
        # -ss before -i seeks fast; re-encoding rather than copying because a
        # stream copy can only cut on a keyframe, which moves the start by up to
        # a couple of seconds and would undo the sentence alignment.
        cut_command = (
            f"ffmpeg -y -ss {clip.start:.3f} -to {clip.end:.3f} -i {q(media)} "
            f"-c:v libx264 -crf 18 -preset veryfast -c:a aac {q(str(cut))}"
        )
        # Fill the frame from the centre, then pad if the source is narrower.
        vertical_command = (
            f"ffmpeg -y -i {q(str(cut))} -vf "
            f'"scale={VERTICAL_WIDTH}:{VERTICAL_HEIGHT}:force_original_aspect_ratio=increase,'
            f'crop={VERTICAL_WIDTH}:{VERTICAL_HEIGHT}" '
            f"-c:a copy {q(str(vertical))}"
        )
        burn_command = (
            f"ffmpeg -y -i {q(str(vertical))} -vf "
            f"\"subtitles={q(str(subs))}:force_style='Alignment=2,FontSize=18,"
            f"BorderStyle=3,Outline=1'\" -c:a copy {q(str(burned))}"
        )
        return ClipPlan(
            index=index,
            start=clip.start,
            end=clip.end,
            duration=round(clip.duration, 2),
            words=clip.words,
            density=clip.density,
            score=clip.score,
            text=clip.text,
            srt=srt,
            cut_command=cut_command,
            vertical_command=vertical_command,
            burn_command=burn_command,
        )
