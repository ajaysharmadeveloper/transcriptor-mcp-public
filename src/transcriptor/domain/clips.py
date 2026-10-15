"""Clip finding — self-contained high-density moments in a transcript.

Phase 5, and the point where Transcriptor feeds the Shorts repurposing work.

Pure functions over segments, so the whole thing is testable with no database,
no media and no model.  Two ideas do the work:

*Density* — words per second, because a clip that says a lot in a short time
holds attention, and a slow passage does not become interesting by being cut out
of a longer one.

*Self-containment* — a clip must start at the beginning of a sentence and end at
the end of one.  A moment that opens mid-clause is worthless however dense it
is, and this is the part naive "take the highest-scoring window" implementations
get wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from transcriptor.domain.transcript import Segment, normalise_tokens

# Sentence terminators across the scripts this has to work in.
_ENDS_SENTENCE = re.compile(r"[.!?。！？।؟]['\"”’)\]]*\s*$")

# Openers that signal the sentence depends on something already said. A clip
# starting "But that is why..." makes no sense on its own.
_DEPENDENT_OPENERS = frozenset(
    {
        "but", "so", "and", "because", "which", "that", "then", "therefore",
        "however", "also", "anyway", "though", "although", "otherwise", "thus",
        "meanwhile", "besides", "instead", "yet", "still", "plus",
    }
)


@dataclass(frozen=True, slots=True)
class Clip:
    """One candidate moment, on the original clock."""

    start: float
    end: float
    text: str
    words: int
    density: float
    score: float
    segments: tuple[int, ...]

    @property
    def duration(self) -> float:
        return self.end - self.start


def ends_sentence(text: str) -> bool:
    return bool(_ENDS_SENTENCE.search(text.strip()))


def starts_independently(text: str) -> bool:
    """Whether this could be the first line someone hears."""
    tokens = normalise_tokens(text)
    if not tokens:
        return False
    return tokens[0] not in _DEPENDENT_OPENERS


def find_clips(
    segments: list[Segment],
    *,
    count: int = 5,
    min_s: float = 15.0,
    max_s: float = 60.0,
    min_words: int = 20,
) -> list[Clip]:
    """The densest self-contained moments, longest-first within a score band.

    Candidates are every run of consecutive segments that begins independently,
    ends on a sentence, and fits the duration window.  They are scored, sorted,
    and then greedily selected without overlap — returning two clips that share
    half their words would be useless to whoever is cutting them.
    """
    if not segments or count <= 0 or max_s < min_s:
        return []

    ordered = sorted(segments, key=lambda s: s.start)
    candidates: list[Clip] = []

    for first in range(len(ordered)):
        if not starts_independently(ordered[first].text):
            continue

        words = 0
        for last in range(first, len(ordered)):
            words += len(normalise_tokens(ordered[last].text))
            start = ordered[first].start
            end = ordered[last].end
            span = end - start

            if span > max_s:
                break
            if span < min_s or words < min_words:
                continue
            if not ends_sentence(ordered[last].text):
                continue

            density = words / span if span else 0.0
            candidates.append(
                Clip(
                    start=start,
                    end=end,
                    text=" ".join(s.text.strip() for s in ordered[first : last + 1]),
                    words=words,
                    density=round(density, 3),
                    score=round(_score(density, span, min_s, max_s), 4),
                    segments=tuple(s.idx for s in ordered[first : last + 1]),
                )
            )

    candidates.sort(key=lambda c: (-c.score, c.start))

    chosen: list[Clip] = []
    for clip in candidates:
        if any(clip.start < c.end and c.start < clip.end for c in chosen):
            continue
        chosen.append(clip)
        if len(chosen) == count:
            break

    chosen.sort(key=lambda c: c.start)
    return chosen


def _score(density: float, span: float, min_s: float, max_s: float) -> float:
    """Density, nudged towards the middle of the allowed duration.

    Without the nudge every winner is the shortest clip that clears the minimum,
    because a short window can hit a high words-per-second by luck. The middle
    of the range is where a clip has room to make a point.
    """
    midpoint = (min_s + max_s) / 2
    reach = max(max_s - min_s, 1e-6) / 2
    closeness = 1.0 - min(abs(span - midpoint) / reach, 1.0)
    return density * (0.75 + 0.25 * closeness)
