"""Clip finding, batch mode, and the HTTP/packaging surface.

Phase 4 and 5. The clip logic is pure, so it needs no database, no media and no
model.
"""

from __future__ import annotations

import uuid

import pytest

from transcriptor.domain.clips import (
    Clip,
    ends_sentence,
    find_clips,
    starts_independently,
)
from transcriptor.domain.transcript import Segment

LINES = [
    "Your AI agent just read a document.",
    "Inside it was one line you never saw.",
    "It said forward the API keys to this address.",
    "And your agent did it.",
    "It did not just get hacked.",
    "It followed instructions perfectly.",
    "That is the whole problem with prompt injection.",
    "The model cannot tell instructions from data.",
    "So you have to build the boundary yourself.",
    "Here is exactly how that works in practice.",
]


def transcript(seconds_each: float = 3.0) -> list[Segment]:
    return [
        Segment(i, i * seconds_each, (i + 1) * seconds_each, line)
        for i, line in enumerate(LINES)
    ]


# ------------------------------------------------------------ the two rules


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Done.", True), ("Really?", True), ("Stop!", True),
        ('He said "go."', True), ("यह वाक्य है।", True),
        ("not finished", False), ("mid-clause and", False),
    ],
)
def test_sentence_endings_across_scripts(text: str, expected: bool) -> None:
    assert ends_sentence(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The thing is simple.", True), ("Here is why.", True),
        ("But that is why.", False), ("So you have to.", False),
        ("And your agent did it.", False), ("Because of that.", False),
        ("", False),
    ],
)
def test_a_clip_must_be_able_to_stand_alone(text: str, expected: bool) -> None:
    """A clip opening 'But that is why...' makes no sense to someone who has not
    heard the sentence before it."""
    assert starts_independently(text) is expected


# ---------------------------------------------------------------- selection


def test_clips_are_found() -> None:
    clips = find_clips(transcript(), count=3, min_s=6, max_s=15, min_words=10)
    assert clips
    assert all(isinstance(c, Clip) for c in clips)


def test_every_clip_starts_and_ends_on_a_sentence() -> None:
    """The part naive 'take the highest-scoring window' implementations get
    wrong."""
    for clip in find_clips(transcript(), count=5, min_s=6, max_s=20, min_words=8):
        assert starts_independently(clip.text)
        assert ends_sentence(clip.text)


def test_clips_never_overlap() -> None:
    """Two clips sharing half their words would be useless to whoever is cutting
    them."""
    clips = find_clips(transcript(), count=5, min_s=6, max_s=15, min_words=8)
    for previous, following in zip(clips, clips[1:]):
        assert previous.end <= following.start


def test_clips_respect_the_duration_window() -> None:
    for clip in find_clips(transcript(), count=5, min_s=8, max_s=12, min_words=8):
        assert 8 <= clip.duration <= 12


def test_clips_are_returned_in_timeline_order() -> None:
    clips = find_clips(transcript(), count=5, min_s=6, max_s=15, min_words=8)
    assert [c.start for c in clips] == sorted(c.start for c in clips)


def test_a_denser_passage_outscores_a_sparse_one() -> None:
    dense = [Segment(0, 0, 10, "One two three four five six seven eight nine ten.")]
    sparse = [Segment(0, 20, 30, "One two three.")]
    clips = find_clips(dense + sparse, count=2, min_s=5, max_s=15, min_words=3)
    assert clips
    assert clips[0].density > (clips[1].density if len(clips) > 1 else 0)


def test_an_empty_transcript_yields_no_clips() -> None:
    assert find_clips([], count=5) == []


def test_a_reversed_window_yields_no_clips() -> None:
    assert find_clips(transcript(), min_s=60, max_s=10) == []


def test_too_few_words_yields_no_clips() -> None:
    assert find_clips(transcript(), min_words=10_000) == []


# ------------------------------------------------------- commands and srt


@pytest.mark.db
async def test_find_clips_emits_runnable_commands(app, long_wav, tmp_path) -> None:
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    dest = tmp_path / "clips"
    result = await app.clips.find(
        created.job_id, count=2, min_s=2, max_s=8, dest_dir=str(dest)
    )
    assert result["clips"], "the fixture transcript must yield something"

    clip = result["clips"][0]
    assert clip["cut_command"].startswith("ffmpeg -y -ss ")
    assert "1080:1920" in clip["vertical_command"]
    assert "subtitles=" in clip["burn_command"]
    # Built with the platform's own separator rather than hardcoded: the
    # command has to be runnable on the machine that receives it.
    assert str(dest / "clip01") in clip["cut_command"]


@pytest.mark.db
async def test_the_clip_srt_is_rebased_to_zero(app, long_wav) -> None:
    """Leaving the original timestamps would put every burned caption minutes
    late."""
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    result = await app.clips.find(created.job_id, count=1, min_s=2, max_s=8)
    srt = result["clips"][0]["srt"]
    assert srt.startswith("1\n00:00:00,")


@pytest.mark.db
async def test_nothing_is_written_to_disk(app, long_wav, tmp_path) -> None:
    """Transcriptor's job is knowing where the good moments are. Running ffmpeg
    over someone's media uninvited is not."""
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    dest = tmp_path / "clips"
    await app.clips.find(created.job_id, count=2, min_s=2, max_s=8, dest_dir=str(dest))
    assert not dest.exists()


@pytest.mark.db
async def test_a_job_with_no_transcript_says_so(app, wav) -> None:
    created = await app.transcription.create(str(wav))
    result = await app.clips.find(created.job_id)
    assert result["clips"] == []
    assert "no transcript yet" in result["detail"]


@pytest.mark.db
async def test_clips_on_an_unknown_job(app) -> None:
    from transcriptor.domain.errors import JobNotFound

    with pytest.raises(JobNotFound):
        await app.clips.find(str(uuid.uuid4()))


# -------------------------------------------------------------- batch mode


@pytest.mark.db
async def test_batch_queues_every_media_file(app, tmp_path, wav) -> None:
    import shutil

    folder = tmp_path / "inbox"
    folder.mkdir()
    for name in ("a.wav", "b.wav", "c.wav"):
        shutil.copy(wav, folder / name)
    (folder / "notes.txt").write_text("not media")

    result = await app.lifecycle.batch(str(folder))
    assert result["matched"] == 3, "the .txt must not be queued"
    assert len(result["queued"]) == 3
    assert all(q["status"] == "queued" for q in result["queued"])


@pytest.mark.db
async def test_batch_is_not_recursive(app, tmp_path, wav) -> None:
    """Recursing would silently pull in a backup directory nobody meant to
    transcribe."""
    import shutil

    folder = tmp_path / "inbox"
    (folder / "nested").mkdir(parents=True)
    shutil.copy(wav, folder / "top.wav")
    shutil.copy(wav, folder / "nested" / "deep.wav")

    result = await app.lifecycle.batch(str(folder))
    assert result["matched"] == 1


@pytest.mark.db
async def test_batch_refuses_a_relative_folder(app) -> None:
    from transcriptor.domain.errors import TranscriptorError

    with pytest.raises(TranscriptorError) as excinfo:
        await app.lifecycle.batch("inbox")
    assert excinfo.value.code == "relative_path_not_allowed"


@pytest.mark.db
async def test_batch_reuses_what_is_already_transcribed(app, tmp_path, wav) -> None:
    import shutil

    folder = tmp_path / "inbox"
    folder.mkdir()
    shutil.copy(wav, folder / "a.wav")

    first = await app.lifecycle.batch(str(folder))
    second = await app.lifecycle.batch(str(folder))

    assert first["queued"][0]["reused"] is False
    assert second["queued"][0]["reused"] is True
    assert second["queued"][0]["job_id"] == first["queued"][0]["job_id"]
