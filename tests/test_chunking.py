"""Chunk planning and overlap stitching.

project-details §6: "Without overlap you lose words at every seam; without
dedupe you get them twice. This is where naive implementations quietly produce
garbage, so it gets dedicated unit tests against a synthetic multi-chunk
fixture."

All pure — no database, no engine, no audio.
"""

from __future__ import annotations

import pytest

from transcriptor.domain.transcript import (
    Segment,
    Word,
    normalise_tokens,
    plan_chunks,
    stitch_overlap,
)

# --------------------------------------------------------------- planning


def test_a_short_file_is_one_chunk() -> None:
    chunks = plan_chunks(77.2, chunk_s=120, overlap_s=15)
    assert len(chunks) == 1
    assert (chunks[0].start_s, chunks[0].end_s) == (0.0, 77.2)


def test_chunks_advance_by_the_stride_and_overlap_by_the_overlap() -> None:
    chunks = plan_chunks(300, chunk_s=120, overlap_s=15)
    assert [(c.idx, c.start_s, c.end_s) for c in chunks] == [
        (0, 0.0, 120.0),
        (1, 105.0, 225.0),
        (2, 210.0, 300.0),
    ]
    for previous, following in zip(chunks, chunks[1:]):
        assert previous.end_s - following.start_s == pytest.approx(15.0)


def test_the_whole_timeline_is_covered_with_no_gaps() -> None:
    chunks = plan_chunks(7200, chunk_s=120, overlap_s=15)
    assert chunks[0].start_s == 0.0
    assert chunks[-1].end_s == 7200
    for previous, following in zip(chunks, chunks[1:]):
        assert following.start_s < previous.end_s, "a gap would lose audio"


def test_a_trailing_sliver_is_absorbed_not_transcribed_alone() -> None:
    """A two-second window gives Whisper no context and reliably produces junk."""
    chunks = plan_chunks(211, chunk_s=120, overlap_s=15)
    assert all(c.duration >= 15 for c in chunks)
    assert chunks[-1].end_s == 211


def test_indices_are_contiguous_from_zero() -> None:
    chunks = plan_chunks(1000, chunk_s=120, overlap_s=15)
    assert [c.idx for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("duration", [0, -1])
def test_no_duration_means_no_chunks(duration: float) -> None:
    assert plan_chunks(duration) == []


def test_an_overlap_at_least_as_big_as_the_chunk_is_rejected() -> None:
    """It would never advance — an infinite loop rather than a bad transcript."""
    with pytest.raises(ValueError):
        plan_chunks(300, chunk_s=120, overlap_s=120)


# ------------------------------------------------------------- tokenisation


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("HELLO world", ["hello", "world"]),
        ("नमस्ते दुनिया", ["नमस्ते", "दुनिया"]),
        ("مرحبا بالعالم", ["مرحبا", "بالعالم"]),
        ("こんにちは、世界", ["こんにちは", "世界"]),
        ("สวัสดีครับ", ["สวัสดีครับ"]),
        ("¿Qué tal?", ["qué", "tal"]),
    ],
)
def test_tokenisation_survives_99_languages(text: str, expected: list[str]) -> None:
    """`\\w` drops combining marks, which would split Devanagari and Thai at
    every character and make those overlaps never match."""
    assert normalise_tokens(text) == expected


# ---------------------------------------------------------------- stitching


def seg(idx: int, start: float, end: float, text: str) -> Segment:
    return Segment(idx=idx, start=start, end=end, text=text)


def test_an_exact_duplicate_is_dropped() -> None:
    previous = [seg(0, 100, 105, "the quick brown fox"),
                seg(1, 105, 110, "jumps over the lazy dog")]
    incoming = [seg(0, 105, 110, "jumps over the lazy dog"),
                seg(1, 110, 115, "and then it kept running")]

    result = stitch_overlap(previous, incoming)
    assert [s.text for s in result.segments] == ["and then it kept running"]
    assert result.dropped == 1
    assert result.matched_tokens == 5


def test_punctuation_and_casing_differences_still_match() -> None:
    """Whisper punctuates the same audio differently depending on the context it
    saw, so a byte-comparison across the seam would never match."""
    previous = [seg(0, 100, 110, "jumps over the lazy dog")]
    incoming = [seg(0, 105, 110, "Jumps over, the lazy dog!"),
                seg(1, 110, 115, "and then it kept running")]

    result = stitch_overlap(previous, incoming)
    assert [s.text for s in result.segments] == ["and then it kept running"]


def test_timestamps_shifting_across_the_seam_do_not_break_the_match() -> None:
    """Segment boundaries move by a few hundred ms between chunks. The seam is
    found by tokens, not by time."""
    previous = [seg(0, 100, 110.0, "jumps over the lazy dog")]
    incoming = [seg(0, 104.4, 109.7, "jumps over the lazy dog"),
                seg(1, 109.7, 115, "and then it kept running")]

    result = stitch_overlap(previous, incoming)
    assert [s.text for s in result.segments] == ["and then it kept running"]


def test_a_seam_falling_mid_segment_trims_rather_than_dropping() -> None:
    """The incoming segment straddles the seam: its first words are already
    committed, its last words are new. Dropping it whole would lose them."""
    previous = [seg(0, 100, 110, "the quick brown fox jumps")]
    incoming = [seg(0, 105, 115, "the quick brown fox jumps over the lazy dog")]

    result = stitch_overlap(previous, incoming)
    assert result.trimmed == 1
    assert [s.text for s in result.segments] == ["over the lazy dog"]
    assert result.segments[0].start > 105, "the start moves with the trimmed words"
    assert result.segments[0].end == 115


def test_words_are_trimmed_alongside_the_text() -> None:
    previous = [seg(0, 0, 2, "one two")]
    incoming = [
        Segment(
            idx=0, start=0, end=4, text="one two three four",
            words=[Word(0, 1, "one"), Word(1, 2, "two"),
                   Word(2, 3, "three"), Word(3, 4, "four")],
        )
    ]
    result = stitch_overlap(previous, incoming, min_match_tokens=2)
    assert [w.text for w in result.segments[0].words] == ["three", "four"]


def test_no_shared_words_falls_back_to_time_coverage() -> None:
    """Silence or music in the overlap means nothing to match on. Anything the
    previous chunk already spanned is dropped — a defined case, not an error."""
    previous = [seg(0, 100, 110, "the quick brown fox")]
    incoming = [seg(0, 105, 109, "completely different words here"),
                seg(1, 110, 115, "and then it kept running")]

    result = stitch_overlap(previous, incoming)
    assert [s.text for s in result.segments] == ["and then it kept running"]
    assert result.matched_tokens == 0


def test_a_short_coincidental_match_is_not_trusted() -> None:
    """Two shared words across a seam is a coincidence, not an alignment."""
    previous = [seg(0, 100, 110, "the quick brown fox")]
    incoming = [seg(0, 105, 109, "the quick something else entirely")]

    result = stitch_overlap(previous, incoming, min_match_tokens=3)
    assert result.matched_tokens == 0


def test_the_first_chunk_keeps_everything() -> None:
    incoming = [seg(0, 0, 5, "the very first words")]
    result = stitch_overlap([], incoming)
    assert result.segments == incoming
    assert result.dropped == 0


def test_an_empty_chunk_stitches_to_nothing() -> None:
    result = stitch_overlap([seg(0, 0, 5, "something")], [])
    assert result.segments == []


def test_a_fully_duplicated_chunk_yields_nothing() -> None:
    """Every word already committed — the chunk contributes no new text."""
    previous = [seg(0, 100, 110, "jumps over the lazy dog")]
    incoming = [seg(0, 105, 110, "jumps over the lazy dog")]

    result = stitch_overlap(previous, incoming)
    assert result.segments == []
    assert result.dropped == 1


def test_stitching_is_multilingual() -> None:
    previous = [seg(0, 100, 110, "नमस्ते दुनिया यह एक परीक्षण है")]
    incoming = [seg(0, 105, 110, "नमस्ते दुनिया यह एक परीक्षण है"),
                seg(1, 110, 115, "और यह जारी है")]

    result = stitch_overlap(previous, incoming)
    assert [s.text for s in result.segments] == ["और यह जारी है"]


def test_no_word_is_ever_emitted_twice_across_a_synthetic_multi_chunk_run() -> None:
    """The property that actually matters, asserted end to end over the pure
    functions.

    A perfect engine transcribes each overlapping window; after stitching, the
    result must read as the original passage exactly once — no word lost at a
    seam, none repeated.
    """
    sentence = (
        "the quick brown fox jumps over the lazy dog while the cat sat on a mat "
        "and the rain fell softly on the roof of the old wooden house near the "
        "river where the heron waits each morning for the tide to turn again"
    ).split()
    # Derived from the word count, so no float rounding can drop the last word.
    per_word = 0.5
    total = len(sentence) * per_word
    words_per_segment = 4

    def transcribe(start: float, end: float) -> list[Segment]:
        """Returns the words inside the window, grouped into segments the way a
        real engine emits phrases rather than single words."""
        first = max(int(start / per_word), 0)
        last = min(round(end / per_word), len(sentence))
        out: list[Segment] = []
        for n, i in enumerate(range(first, last, words_per_segment)):
            group = sentence[i : min(i + words_per_segment, last)]
            out.append(
                Segment(
                    idx=n,
                    start=i * per_word,
                    end=(i + len(group)) * per_word,
                    text=" ".join(group),
                )
            )
        return out

    committed: list[Segment] = []
    matches = 0
    for chunk in plan_chunks(total, chunk_s=20, overlap_s=5):
        raw = transcribe(chunk.start_s, chunk.end_s)
        tail = [s for s in committed if s.end > chunk.start_s - 1.0]
        result = stitch_overlap(tail, raw, min_match_tokens=3)
        matches += result.matched_tokens
        committed.extend(result.segments)

    produced = " ".join(s.text for s in committed).split()
    assert produced == sentence, "every word exactly once, in order"
    assert matches > 0, "the token alignment must be doing the work, not the fallback"


def test_segment_times_stay_ordered_across_seams() -> None:
    """A stitched transcript that jumps backwards in time renders a broken SRT."""
    sentence = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
                "kilo lima mike november oscar papa quebec romeo sierra tango").split()
    per_word = 0.5
    total = len(sentence) * per_word

    def transcribe(start: float, end: float) -> list[Segment]:
        first, last = int(start / per_word), min(round(end / per_word), len(sentence))
        return [
            Segment(n, i * per_word, (i + 2) * per_word, " ".join(sentence[i : i + 2]))
            for n, i in enumerate(range(first, last, 2))
        ]

    committed: list[Segment] = []
    for chunk in plan_chunks(total, chunk_s=4, overlap_s=1):
        raw = transcribe(chunk.start_s, chunk.end_s)
        tail = [s for s in committed if s.end > chunk.start_s - 1.0]
        committed.extend(stitch_overlap(tail, raw, min_match_tokens=2).segments)

    starts = [s.start for s in committed]
    assert starts == sorted(starts), "segments must never go backwards in time"


# ------------------------------------------------------- monotonic timeline


def test_a_segment_starting_before_the_seam_is_pulled_forward() -> None:
    """Whisper's boundaries drift between chunks, so an incoming segment can
    legitimately start before the last committed one ended.  Left alone that
    renders a subtitle appearing while the previous one is still on screen."""
    previous = [seg(0, 50, 55.14, "still make it I want")]
    incoming = [seg(0, 54.22, 57.04, "walk it line by line in a real")]

    result = stitch_overlap(previous, incoming)
    assert result.segments[0].start >= 55.14
    assert result.segments[0].end == 57.04
    assert result.segments[0].text == "walk it line by line in a real", (
        "only the clock moves, never the words"
    )


def test_a_segment_entirely_behind_the_seam_is_dropped() -> None:
    previous = [seg(0, 50, 60, "the quick brown fox jumps")]
    incoming = [seg(0, 55, 58, "wholly different words"),
                seg(1, 60, 65, "and then it kept running")]

    result = stitch_overlap(previous, incoming)
    assert [s.text for s in result.segments] == ["and then it kept running"]


def test_the_stitched_result_is_always_ordered() -> None:
    previous = [seg(0, 0, 10, "alpha bravo charlie delta")]
    incoming = [seg(0, 9.5, 12, "echo foxtrot"), seg(1, 8.0, 14, "golf hotel")]

    result = stitch_overlap(previous, incoming)
    starts = [s.start for s in result.segments]
    ends = [s.end for s in result.segments]
    assert starts == sorted(starts)
    assert all(s >= 10.0 for s in starts), "nothing may precede the committed end"
    assert ends == sorted(ends)


# ------------------------------------------------------------ seam prompting


@pytest.mark.parametrize(
    ("committed", "expected"),
    [
        ("That fixes this one. It doesn't fix it. I still make it.", "I still make it."),
        ("First. Second! Third?", "Third?"),
        ("no terminal punctuation at all", "no terminal punctuation at all"),
        ("यह पहला वाक्य है। यह दूसरा वाक्य है।", "यह दूसरा वाक्य है।"),
        ("", None),
    ],
)
def test_the_seam_prompt_is_the_last_sentence(committed: str, expected) -> None:
    """Measured at the 50s seam of the 77s fixture: a 220-character prompt made
    Whisper echo it back into the transcript (9 repeated 5-grams), and the echo
    lands past the seam where stitching never looks. One sentence gives the same
    casing and punctuation with none of that."""
    from transcriptor.services.stages.transcribe import _prompt_from

    previous = [Segment(0, 0, 1, committed)] if committed else []
    assert _prompt_from(previous) == expected


def test_the_seam_prompt_is_bounded() -> None:
    """It shares Whisper's context window with the transcript being produced."""
    from transcriptor.services.stages.transcribe import _prompt_from

    prompt = _prompt_from([Segment(0, 0, 1, "word " * 500)])
    assert prompt is not None and len(prompt) <= 120
