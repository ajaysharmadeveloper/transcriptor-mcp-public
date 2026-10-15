"""Renderers: pure functions of segments, no files written."""

from __future__ import annotations

import json

import pytest

from transcriptor.adapters.renderers import _timestamp, render
from transcriptor.domain.transcript import Format, Segment, Word

SEGMENTS = [
    Segment(0, 0.0, 2.5, "Hello world"),
    Segment(1, 2.5, 5.25, "Second line"),
]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (5.25, "00:00:05,250"),
        (59.9996, "00:01:00,000"),   # rounds up into the next minute
        (3661.5, "01:01:01,500"),
        (-1.0, "00:00:00,000"),      # clamped, never negative
    ],
)
def test_srt_timestamps(seconds: float, expected: str) -> None:
    assert _timestamp(seconds, sep=",") == expected


def test_srt_is_well_formed() -> None:
    assert render(SEGMENTS, Format.SRT) == (
        "1\n00:00:00,000 --> 00:00:02,500\nHello world\n\n"
        "2\n00:00:02,500 --> 00:00:05,250\nSecond line\n"
    )


def test_srt_numbers_by_position_so_a_partial_transcript_is_valid() -> None:
    """A resumed or aborted job starts at segment idx 7; the cues must still
    number from 1."""
    partial = [Segment(7, 10.0, 12.0, "later"), Segment(8, 12.0, 14.0, "still later")]
    assert render(partial, Format.SRT).startswith("1\n")
    assert "\n2\n" in render(partial, Format.SRT)


def test_blank_segments_are_dropped_not_emitted_as_empty_cues() -> None:
    segs = [Segment(0, 0.0, 1.0, "  "), Segment(1, 1.0, 2.0, "real")]
    assert render(segs, Format.SRT) == "1\n00:00:01,000 --> 00:00:02,000\nreal\n"
    assert render(segs, Format.TXT) == "real\n"


def test_empty_transcript_renders_empty_not_broken() -> None:
    assert render([], Format.SRT) == ""
    assert render([], Format.TXT) == ""


def test_end_before_start_never_produces_a_reversed_cue() -> None:
    segs = [Segment(0, 5.0, 4.0, "clock skew")]
    line = render(segs, Format.SRT).splitlines()[1]
    start, end = line.split(" --> ")
    assert end >= start


def test_txt_is_one_line_per_segment() -> None:
    assert render(SEGMENTS, Format.TXT) == "Hello world\nSecond line\n"


def test_vtt_has_a_header() -> None:
    out = render(SEGMENTS, Format.VTT)
    assert out.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in out


def test_json_carries_words_when_present() -> None:
    segs = [Segment(0, 0.0, 1.0, "hi", words=[Word(0.0, 1.0, "hi")])]
    payload = json.loads(render(segs, Format.JSON))
    assert payload[0]["words"] == [{"start": 0.0, "end": 1.0, "text": "hi"}]
    assert "words" not in json.loads(render(SEGMENTS, Format.JSON))[0]


def test_unicode_survives_rendering() -> None:
    """99 languages means never assuming ASCII."""
    segs = [Segment(0, 0.0, 1.0, "नमस्ते दुनिया")]
    assert "नमस्ते" in render(segs, Format.SRT)
    assert "नमस्ते" in json.loads(render(segs, Format.JSON))[0]["text"]
