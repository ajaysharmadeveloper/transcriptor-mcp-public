"""txt / srt / vtt / json rendered from segments, on demand.

Segments are the source of truth; every format is a pure function of them.
Nothing is written to disk — phase 1 has no ``renders`` cache and no
``transcript_export``, so a render exists only in the response.

Pure and synchronous: fast enough for the event loop at phase 1 sizes (a 2-hour
transcript is ~2,000 segments and renders in single-digit milliseconds), and
trivially testable.
"""

from __future__ import annotations

import json
import re

from transcriptor.domain.errors import TranscriptorError
from transcriptor.domain.transcript import Format, Segment


def _timestamp(seconds: float, *, sep: str) -> str:
    """``HH:MM:SS,mmm`` for SRT, ``HH:MM:SS.mmm`` for VTT.

    Rounds to milliseconds first, so 59.9996 becomes 00:01:00.000 rather than
    00:00:59.1000.
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def render_txt(segments: list[Segment]) -> str:
    """Plain text, one line per segment, blank lines collapsed."""
    lines = [s.text.strip() for s in segments if s.text.strip()]
    return "\n".join(lines) + ("\n" if lines else "")


def render_srt(segments: list[Segment]) -> str:
    """SubRip.  Cues are numbered from 1 and separated by a blank line.

    Numbering follows position in the rendered output, not ``segment.idx``, so
    a partial transcript still produces a valid file.
    """
    blocks: list[str] = []
    for n, s in enumerate((s for s in segments if s.text.strip()), start=1):
        start = _timestamp(s.start, sep=",")
        end = _timestamp(max(s.end, s.start), sep=",")
        blocks.append(f"{n}\n{start} --> {end}\n{s.text.strip()}\n")
    return "\n".join(blocks)


def render_vtt(segments: list[Segment]) -> str:
    """WebVTT.  Phase 2 surfaces this through ``transcript_get``."""
    blocks = ["WEBVTT\n"]
    for s in segments:
        if not s.text.strip():
            continue
        start = _timestamp(s.start, sep=".")
        end = _timestamp(max(s.end, s.start), sep=".")
        blocks.append(f"{start} --> {end}\n{s.text.strip()}\n")
    return "\n".join(blocks)


def render_json(segments: list[Segment]) -> str:
    payload = [
        {
            "idx": s.idx,
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": s.text,
            **(
                {"words": [
                    {"start": round(w.start, 3), "end": round(w.end, 3), "text": w.text}
                    for w in s.words
                ]}
                if s.words
                else {}
            ),
        }
        for s in segments
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


_RENDERERS = {
    Format.TXT: render_txt,
    Format.SRT: render_srt,
    Format.VTT: render_vtt,
    Format.JSON: render_json,
}

# Phase 1 ships txt and srt; vtt and json are wired but not exposed by the tools.
PHASE_1_FORMATS = (Format.TXT, Format.SRT)


def render(segments: list[Segment], fmt: Format) -> str:
    try:
        return _RENDERERS[fmt](segments)
    except KeyError:
        raise TranscriptorError(
            f"unknown format {fmt!r}", code="unknown_format", resumable=False
        ) from None


# ------------------------------------------------------------------ parsing

_VTT_TIME = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
)
# Inline cue markup: <c>, <i>, <00:00:01.000> karaoke timings, and so on.
_VTT_TAGS = re.compile(r"<[^>]*>")


def _vtt_seconds(hours: str | None, minutes: str, seconds: str, millis: str) -> float:
    return (
        int(hours or 0) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis.ljust(3, "0")) / 1000.0
    )


def parse_vtt(text: str) -> list[Segment]:
    """WebVTT (or SRT) cues into segments.

    Written here rather than pulled in as a dependency because the shape is
    trivial and the awkward parts are specific to caption files: YouTube's
    auto-captions repeat each line as a rolling two-line window, which would
    otherwise double every word in the transcript.
    """
    segments: list[Segment] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    idx = 0
    position = 0
    while position < len(lines):
        match = _VTT_TIME.search(lines[position])
        if match is None:
            position += 1
            continue

        start = _vtt_seconds(*match.group(1, 2, 3, 4))
        end = _vtt_seconds(*match.group(5, 6, 7, 8))

        position += 1
        body: list[str] = []
        while position < len(lines) and lines[position].strip():
            if _VTT_TIME.search(lines[position]):
                break
            body.append(_VTT_TAGS.sub("", lines[position]).strip())
            position += 1

        content = " ".join(part for part in body if part).strip()
        if not content:
            continue

        # Auto-captions roll: each cue repeats the previous line plus one new
        # one. Keep only what is genuinely new.
        if segments:
            previous = segments[-1].text
            if content == previous:
                continue
            if content.startswith(previous + " "):
                content = content[len(previous) :].strip()
            elif previous.endswith(content):
                continue

        segments.append(Segment(idx=idx, start=start, end=max(end, start), text=content))
        idx += 1

    return segments


def render_clip_srt(segments: list[Segment], start: float, end: float) -> str:
    """An SRT for one clip, rebased so the clip starts at zero.

    A burned-in caption track has to be relative to the cut file; leaving the
    original timestamps would put every caption minutes late.
    """
    within = [s for s in segments if s.end > start and s.start < end]
    rebased = [
        Segment(
            idx=n,
            start=max(s.start - start, 0.0),
            end=min(s.end, end) - start,
            text=s.text,
        )
        for n, s in enumerate(within)
    ]
    return render_srt(rebased)


class SrtRenderer:
    """Implements :class:`transcriptor.ports.renderer.SubtitleRenderer`."""

    def clip_srt(self, segments: list[Segment], start: float, end: float) -> str:
        return render_clip_srt(segments, start, end)
