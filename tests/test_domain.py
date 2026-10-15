"""Domain rules, tested with no database, no engine and no filesystem."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from transcriptor.domain.errors import (
    IllegalTransition,
    MemoryPressure,
    PlaylistNotSupported,
    RelativePathNotAllowed,
    TranscriptorError,
    UnsupportedSource,
)
from transcriptor.domain.job import (
    JobStatus,
    ReuseReason,
    TRANSITIONS,
    assert_transition,
    is_resume,
    may_transition,
)
from transcriptor.domain.source import CaptionPolicy, SourceKind, parse_source, result_key


# ------------------------------------------------------------- state machine


def test_every_status_has_a_transition_entry() -> None:
    assert set(TRANSITIONS) == set(JobStatus)


@pytest.mark.parametrize(
    ("frm", "to"),
    [
        (JobStatus.QUEUED, JobStatus.RUNNING),
        (JobStatus.QUEUED, JobStatus.CANCELLED),
        (JobStatus.RUNNING, JobStatus.PAUSED),
        (JobStatus.RUNNING, JobStatus.DONE),
        (JobStatus.RUNNING, JobStatus.INTERRUPTED),
        (JobStatus.PAUSED, JobStatus.RUNNING),
        (JobStatus.FAILED, JobStatus.QUEUED),
        (JobStatus.INTERRUPTED, JobStatus.QUEUED),
    ],
)
def test_legal_transitions(frm: JobStatus, to: JobStatus) -> None:
    assert may_transition(frm, to)


@pytest.mark.parametrize(
    ("frm", "to"),
    [
        (JobStatus.DONE, JobStatus.RUNNING),
        (JobStatus.DONE, JobStatus.QUEUED),
        (JobStatus.CANCELLED, JobStatus.QUEUED),
        (JobStatus.QUEUED, JobStatus.DONE),
        (JobStatus.QUEUED, JobStatus.PAUSED),
        (JobStatus.RUNNING, JobStatus.QUEUED),
    ],
)
def test_illegal_transitions(frm: JobStatus, to: JobStatus) -> None:
    assert not may_transition(frm, to)
    with pytest.raises(IllegalTransition):
        assert_transition(frm, to)


def test_done_and_cancelled_are_final_even_to_resume() -> None:
    """Only failed and interrupted are resumable.  A completed or deliberately
    cancelled job stays put."""
    assert TRANSITIONS[JobStatus.DONE] == frozenset()
    assert TRANSITIONS[JobStatus.CANCELLED] == frozenset()
    assert not is_resume(JobStatus.DONE, JobStatus.QUEUED)
    assert is_resume(JobStatus.FAILED, JobStatus.QUEUED)


def test_active_statuses_match_the_partial_unique_index() -> None:
    """`is_active` and `jobs_one_active_idx` must never drift apart."""
    active = {s for s in JobStatus if s.is_active}
    assert active == {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSED}
    assert {s for s in JobStatus if s.is_terminal} == set(JobStatus) - active


# -------------------------------------------------------------------- errors


def test_resumable_is_decided_where_the_error_is_raised() -> None:
    assert MemoryPressure("tight").resumable is True
    assert MemoryPressure("tight").code == "memory_pressure"
    assert UnsupportedSource("nope").resumable is False


def test_error_context_is_structured_not_stringly_typed() -> None:
    err = MemoryPressure("tight", available_gb=1.2, floor_gb=3.0)
    assert err.context == {"available_gb": 1.2, "floor_gb": 3.0}


def test_every_named_error_declares_a_code_and_resumable() -> None:
    """Invariant I10, at the definition site."""
    from transcriptor.domain import errors

    named = [
        v
        for v in vars(errors).values()
        if isinstance(v, type)
        and issubclass(v, TranscriptorError)
        and v is not TranscriptorError
    ]
    assert len(named) > 15
    for cls in named:
        assert isinstance(cls.code, str) and cls.code
        assert isinstance(cls.resumable, bool)


# -------------------------------------------------------------------- source


def test_absolute_path_is_accepted_and_resolved(tmp_path) -> None:
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    ref = parse_source(str(f))
    assert ref.kind is SourceKind.FILE
    assert ref.platform == "local"
    # normcase on both sides: the key is the dedupe identity, and on a
    # case-insensitive filesystem two spellings of one path must collapse to
    # one. It is identity on POSIX. What the user typed is kept verbatim in
    # `raw`, which is what any response shows them.
    assert ref.key == os.path.normcase(str(f.resolve()))


def test_home_relative_path_is_expanded() -> None:
    ref = parse_source("~/Movies/ep12.mp4")
    # Absolute rather than "starts with /" — on Windows an expanded home is
    # C:\Users\..., and the old assertion made this test POSIX-only while
    # looking general.
    assert Path(ref.key).is_absolute()
    assert "~" not in ref.key


def test_relative_path_is_refused_rather_than_guessed() -> None:
    """A stdio server launched by a GUI client has no meaningful working
    directory, so guessing would be worse than refusing."""
    with pytest.raises(RelativePathNotAllowed) as excinfo:
        parse_source("videos/ep12.mp4")
    assert excinfo.value.code == "relative_path_not_allowed"


def test_a_url_is_normalised_to_its_identity() -> None:
    """Tracking parameters stripped, video id kept — so a shared link and a
    clean one are the same source_key, and layer-1 dedupe answers with no
    network at all."""
    ref = parse_source("https://youtu.be/7ksp9N7vguE?si=jDRyqzNlBauV804x")
    assert ref.kind is SourceKind.URL
    assert ref.platform == "youtube"
    assert ref.key == "https://www.youtube.com/watch?v=7ksp9N7vguE"


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/7ksp9N7vguE?si=x",
        "https://www.youtube.com/watch?v=7ksp9N7vguE&t=42&feature=share",
        "https://m.youtube.com/watch?v=7ksp9N7vguE",
        "https://www.youtube-nocookie.com/embed/7ksp9N7vguE",
    ],
)
def test_every_form_of_a_youtube_link_dedupes_to_one_key(url: str) -> None:
    assert parse_source(url).key == "https://www.youtube.com/watch?v=7ksp9N7vguE"


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/playlist?list=PL123",
        "https://www.youtube.com/@someone",
        "https://www.youtube.com/channel/UC123",
        "https://www.tiktok.com/@handle",
    ],
)
def test_a_playlist_or_channel_is_refused_as_ambiguous(url: str) -> None:
    """Not unsupported — ambiguous. Transcriptor cannot tell which video was
    meant, and guessing would be worse than saying so."""
    with pytest.raises(PlaylistNotSupported):
        parse_source(url)


@pytest.mark.parametrize(
    ("url", "platform"),
    [
        ("https://www.tiktok.com/@a/video/7", "tiktok"),
        ("https://x.com/u/status/1", "x"),
        ("https://vimeo.com/12345", "vimeo"),
        ("https://www.instagram.com/reel/XYZ/", "instagram"),
        ("https://cdn.example.com/raw/ep12.mp4", "generic"),
    ],
)
def test_single_item_urls_are_accepted(url: str, platform: str) -> None:
    assert parse_source(url).platform == platform


def test_a_non_http_scheme_is_refused() -> None:
    with pytest.raises(UnsupportedSource):
        parse_source("ftp://example.com/a.mp4")


# ---------------------------------------------------------------- result_key

BASE = dict(
    identity="audio-hash", model="whisper-8bit", language=None,
    captions=CaptionPolicy.MANUAL_ONLY,
)


def test_result_key_is_stable() -> None:
    assert result_key(**BASE) == result_key(**BASE)


@pytest.mark.parametrize(
    "changed",
    [
        {"identity": "other-hash"},
        {"model": "whisper-fp16"},
        {"language": "hi"},
        {"captions": CaptionPolicy.NEVER},
    ],
    ids=["identity", "model", "language", "captions"],
)
def test_every_component_changes_the_key(changed: dict) -> None:
    assert result_key(**{**BASE, **changed}) != result_key(**BASE)


def test_language_none_and_auto_are_the_same_request() -> None:
    assert result_key(**{**BASE, "language": None}) == result_key(
        **{**BASE, "language": "auto"}
    )


def test_components_cannot_collide_by_running_together() -> None:
    """Length-prefixed, so ('ab','c') and ('a','bc') stay distinct."""
    a = result_key(identity="ab", model="c", language="x", captions=CaptionPolicy.NEVER)
    b = result_key(identity="a", model="bc", language="x", captions=CaptionPolicy.NEVER)
    assert a != b


def test_reuse_reasons_cover_the_documented_set() -> None:
    assert {r.value for r in ReuseReason} == {
        "exact_source",
        "same_audio",
        "already_running",
        "resumed",
    }
