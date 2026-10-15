"""Segment, Chunk, Timeline, AudioSlice.

Segments are the source of truth.  txt, srt, vtt and json are pure functions of
them, which is why nothing is stored twice.
"""

from __future__ import annotations

import difflib
import enum
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


class ChunkStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Format(str, enum.Enum):
    TXT = "txt"
    SRT = "srt"
    VTT = "vtt"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class Word:
    """A word-level timing.  Roughly triples segment row size, which is why
    ``include_words`` defaults to off."""

    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass(frozen=True, slots=True)
class Segment:
    """One timestamped piece of transcript."""

    idx: int
    start: float
    end: float
    text: str
    words: list[Word] | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Chunk:
    """One unit of work within a job, and the checkpoint unit.

    ``Transcribe`` is the only stage that can take hours, which is exactly why
    the chunk is what gets committed.
    """

    idx: int
    start_s: float
    end_s: float
    status: ChunkStatus = ChunkStatus.PENDING
    attempts: int = 0
    engine_ms: int | None = None

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class Timeline:
    """Speech regions found by VAD, as (start, end) pairs on the original clock.

    Whisper hallucinates confident invented sentences over silence and drifts on
    long-form audio, so the non-speech is removed before transcription and the
    resulting timings are mapped back through here.  That mapping is the whole
    reason this type exists: without it every timestamp would refer to a
    compressed timeline the caller never sees.
    """

    regions: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for start, end in self.regions:
            if end < start:
                raise ValueError(f"region ends before it starts: ({start}, {end})")

    @property
    def speech_seconds(self) -> float:
        return sum(end - start for start, end in self.regions)

    @property
    def is_empty(self) -> bool:
        return self.speech_seconds <= 0

    @property
    def span(self) -> tuple[float, float] | None:
        """First speech to last, on the original clock."""
        if not self.regions:
            return None
        return self.regions[0][0], self.regions[-1][1]

    def to_original(self, speech_s: float) -> float:
        """Map a time on the speech-only clock back to the original recording.

        Speech-only time runs continuously across the concatenated regions, so
        this walks them, subtracting each region's length until the offset falls
        inside one.
        """
        if not self.regions:
            return speech_s
        remaining = max(speech_s, 0.0)
        for start, end in self.regions:
            length = end - start
            if remaining <= length:
                return start + remaining
            remaining -= length
        # Past the end of all speech: clamp to the last region's end rather than
        # inventing a timestamp beyond the recording.
        return self.regions[-1][1]

    def to_speech(self, original_s: float) -> float:
        """The inverse.  A time inside a silence maps to the start of the next
        speech region, since no speech-only time corresponds to it."""
        elapsed = 0.0
        for start, end in self.regions:
            if original_s < start:
                return elapsed
            if original_s <= end:
                return elapsed + (original_s - start)
            elapsed += end - start
        return elapsed

    def remap(self, segments: list[Segment]) -> list[Segment]:
        """Move segments from the speech-only clock onto the original one."""
        return [
            Segment(
                idx=s.idx,
                start=self.to_original(s.start),
                end=self.to_original(s.end),
                text=s.text,
                words=(
                    [
                        Word(
                            start=self.to_original(w.start),
                            end=self.to_original(w.end),
                            text=w.text,
                            probability=w.probability,
                        )
                        for w in s.words
                    ]
                    if s.words
                    else None
                ),
            )
            for s in segments
        ]

    def merged(self, *, gap_s: float = 0.0) -> Timeline:
        """Regions sorted, overlaps collapsed, and gaps under ``gap_s`` closed.

        Cutting at every micro-pause would hand Whisper a stream of fragments
        with no context, which is how VAD makes a transcript worse rather than
        better.
        """
        if not self.regions:
            return Timeline([])
        ordered = sorted(self.regions)
        merged: list[tuple[float, float]] = [ordered[0]]
        for start, end in ordered[1:]:
            last_start, last_end = merged[-1]
            if start - last_end <= gap_s:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return Timeline(merged)

    def to_json(self) -> list[list[float]]:
        return [[start, end] for start, end in self.regions]

    @classmethod
    def from_json(cls, raw: object) -> Timeline | None:
        if not raw:
            return None
        return cls([(float(a), float(b)) for a, b in raw])  # type: ignore[misc]

    @classmethod
    def whole(cls, duration_s: float) -> Timeline:
        """Everything is speech.  What a detector-free pipeline uses."""
        return cls([(0.0, duration_s)])


@dataclass(frozen=True, slots=True)
class AudioSlice:
    """What the engine is handed: a path plus the window to transcribe.

    A path rather than bytes because both ffmpeg and Whisper read from files.
    """

    path: Path
    start_s: float = 0.0
    end_s: float | None = None


# ---------------------------------------------------------------- chunking


def plan_chunks(
    duration_s: float, *, chunk_s: float = 120.0, overlap_s: float = 15.0
) -> list[Chunk]:
    """Split a duration into overlapping windows.

    Chunks advance by ``chunk_s - overlap_s`` and each one re-covers the last
    ``overlap_s`` of its predecessor.  Without the overlap you lose words at
    every seam; :func:`stitch_overlap` removes the duplicates it creates.

    A trailing sliver shorter than the overlap is absorbed into the previous
    chunk instead of becoming a chunk of its own: a two-second window gives
    Whisper no context and reliably produces junk.
    """
    if duration_s <= 0:
        return []
    if overlap_s >= chunk_s:
        raise ValueError("overlap_s must be smaller than chunk_s")
    if duration_s <= chunk_s:
        return [Chunk(idx=0, start_s=0.0, end_s=duration_s)]

    stride = chunk_s - overlap_s
    chunks: list[Chunk] = []
    start = 0.0
    idx = 0
    while start < duration_s:
        end = min(start + chunk_s, duration_s)
        chunks.append(Chunk(idx=idx, start_s=start, end_s=end))
        if end >= duration_s:
            break
        start += stride
        idx += 1

    # Absorb a trailing sliver rather than transcribing it alone.
    if len(chunks) > 1 and chunks[-1].duration < overlap_s:
        tail = chunks.pop()
        previous = chunks[-1]
        chunks[-1] = Chunk(
            idx=previous.idx, start_s=previous.start_s, end_s=tail.end_s
        )
    return chunks


# --------------------------------------------------------------- stitching

# Unicode major categories to discard: Punctuation, Symbols, separators (Z) and
# Control.  Letters (L), Marks (M) and Numbers (N) survive.
#
# Deliberately not `[^\w\s]`: `\w` excludes combining marks, which splits
# "नमस्ते" into "नमस" + "त" and would make Hindi, Thai, Tamil and Arabic overlaps
# fail to match at every seam.  This system targets 99 languages, so the
# tokeniser cannot be English-shaped.
_DISCARD_CATEGORIES = frozenset("PSZC")


def normalise_tokens(text: str) -> list[str]:
    """Words, case-folded and stripped of punctuation.

    Compared this way so "Hello, world." and "hello world" match across a seam —
    Whisper punctuates the same audio differently depending on what context
    preceded it.  NFKC first, because the same word can arrive in two Unicode
    normal forms.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    cleaned = "".join(
        " " if unicodedata.category(ch)[0] in _DISCARD_CATEGORIES else ch
        for ch in folded
    )
    return cleaned.split()


@dataclass(frozen=True, slots=True)
class Stitched:
    """The result of removing an overlap's duplicates."""

    segments: list[Segment]
    dropped: int = 0
    trimmed: int = 0
    matched_tokens: int = 0


def stitch_overlap(
    previous: list[Segment],
    incoming: list[Segment],
    *,
    min_match_tokens: int = 3,
) -> Stitched:
    """Drop the part of ``incoming`` that ``previous`` already transcribed.

    Both lists must already be on the original timeline.

    The seam is found by the longest contiguous run of matching tokens between
    the tail of ``previous`` and the head of ``incoming`` — not by timestamps,
    because Whisper's segment boundaries move by a few hundred milliseconds
    depending on the context it saw, so the same words get different times in
    the two chunks.

    When no run of at least ``min_match_tokens`` is found the overlap carried no
    shared words — silence, music, or a genuine mishearing — and the fallback is
    time coverage: anything the previous chunk already spanned is dropped. That
    is a defined case, not an error being swallowed.
    """
    if not incoming:
        return Stitched(segments=[])
    if not previous:
        return Stitched(segments=list(incoming))

    previous_end = max(s.end for s in previous)
    overlap_start = incoming[0].start

    # Only the parts either side of the seam can be duplicates.
    prev_tail = [s for s in previous if s.end > overlap_start]
    head = [s for s in incoming if s.start < previous_end]
    if not prev_tail or not head:
        return Stitched(segments=list(incoming))

    prev_tokens = [tok for s in prev_tail for tok in normalise_tokens(s.text)]

    head_tokens: list[str] = []
    # token index -> (segment position in `incoming`, token position in segment)
    origin: list[tuple[int, int]] = []
    for position, segment in enumerate(head):
        for offset, token in enumerate(normalise_tokens(segment.text)):
            head_tokens.append(token)
            origin.append((position, offset))

    if not prev_tokens or not head_tokens:
        return Stitched(segments=list(incoming))

    match = difflib.SequenceMatcher(
        None, prev_tokens, head_tokens, autojunk=False
    ).find_longest_match(0, len(prev_tokens), 0, len(head_tokens))

    if match.size < min_match_tokens:
        kept = _monotonic([s for s in incoming if s.end > previous_end], previous_end)
        return Stitched(segments=kept, dropped=len(incoming) - len(kept))

    # Everything up to the end of the matched run is already committed.
    cut = match.b + match.size
    if cut >= len(head_tokens):
        kept = _monotonic(list(incoming[len(head) :]), previous_end)
        return Stitched(
            segments=kept,
            dropped=len(incoming) - len(kept),
            matched_tokens=match.size,
        )

    cut_segment, cut_offset = origin[cut]
    kept = list(incoming[len(head) :])
    trimmed = 0

    tail_of_head = head[cut_segment:]
    if tail_of_head:
        first = tail_of_head[0]
        words = first.text.split()
        if cut_offset and cut_offset < len(words):
            # The seam falls mid-segment: keep the words the previous chunk did
            # not reach, and move the start proportionally.
            remaining = " ".join(words[cut_offset:])
            span = first.end - first.start
            fraction = cut_offset / max(len(words), 1)
            first = Segment(
                idx=first.idx,
                start=first.start + span * fraction,
                end=first.end,
                text=remaining,
                words=first.words[cut_offset:] if first.words else None,
            )
            trimmed = 1
        kept = [first, *tail_of_head[1:], *kept]

    kept = _monotonic(kept, previous_end)

    return Stitched(
        segments=kept,
        dropped=len(incoming) - len(kept) - trimmed,
        trimmed=trimmed,
        matched_tokens=match.size,
    )


def _monotonic(segments: list[Segment], previous_end: float) -> list[Segment]:
    """Keep the timeline moving forwards across a seam.

    Whisper's segment boundaries drift between chunks, so a kept segment can
    legitimately start *before* the last committed one ended.  Left alone that
    renders a subtitle that appears while the previous one is still on screen,
    and an SRT whose cues run backwards.

    Segments already covered in full are dropped; one that straddles the seam
    has its start pulled forward to it.  Text is never altered — only the clock.
    """
    out: list[Segment] = []
    floor = previous_end
    for segment in segments:
        if segment.end <= floor:
            continue
        start = max(segment.start, floor)
        out.append(
            segment
            if start == segment.start
            else Segment(
                idx=segment.idx,
                start=start,
                end=segment.end,
                text=segment.text,
                words=segment.words,
            )
        )
        floor = max(floor, segment.end)
    return out
