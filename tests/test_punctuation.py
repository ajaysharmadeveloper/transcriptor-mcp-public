"""Punctuation restoration.

Disabling `condition_on_previous_text` bought ~4x throughput and all but
eliminated repetition, at the cost of sentence boundaries: only the first 30s
window of each Whisper call is well segmented. This stage puts them back.
"""

from __future__ import annotations

import uuid

import pytest

from transcriptor.adapters.fake_punctuator import FakePunctuator
from transcriptor.adapters.hf_punctuator import _capitalise, _insert
from transcriptor.domain.job import JobStatus
from transcriptor.domain.transcript import Format


# ------------------------------------------------------------- pure helpers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello world. this works.", "Hello world. This works."),
        ("first? second! third.", "First? Second! Third."),
        # Any non-space closes the opportunity, or "89% can" becomes "89% Can".
        ("89% can be taken over.", "89% can be taken over."),
        ("already Capitalised.", "Already Capitalised."),
        ("", ""),
        ("   ", "   "),
    ],
)
def test_sentence_casing(text: str, expected: str) -> None:
    assert _capitalise(text) == expected


def test_marks_are_inserted_at_the_reported_offsets() -> None:
    """Offsets rather than re-tokenising: rejoining subwords turned "easiest"
    into "e, asi, est" on the first attempt."""
    text = "all of it stay with me"
    groups = [{"entity_group": ".", "end": 9}, {"entity_group": "0", "end": 22}]
    assert _insert(text, groups) == "all of it. stay with me"


def test_an_already_punctuated_position_is_left_alone() -> None:
    """What makes the whole pass idempotent."""
    text = "all of it. stay with me"
    groups = [{"entity_group": ".", "end": 10}]
    assert _insert(text, groups) == text


def test_out_of_range_offsets_are_ignored() -> None:
    assert _insert("short", [{"entity_group": ".", "end": 999}]) == "short"


# --------------------------------------------------------- language support


@pytest.mark.parametrize("language", ["en", "de", "fr", "es", "pt", "en-US"])
def test_supported_languages(language: str) -> None:
    assert FakePunctuator().supports(language) is True


@pytest.mark.parametrize("language", ["hi", "ja", "ar", "th", "zh", None, ""])
def test_unsupported_languages_are_left_alone(language) -> None:
    """Transcriptor targets 99 languages and no punctuation model comes close.
    Leaving the transcript alone beats corrupting it."""
    assert FakePunctuator().supports(language) is False


# ---------------------------------------------------------------- the stage

pytestmark_db = pytest.mark.db


@pytest.mark.db
async def test_the_transcript_is_punctuated(app, long_wav) -> None:
    created = await app.transcription.create(str(long_wav))
    assert await app.runner.tick() is JobStatus.DONE

    view = await app.reads.get(created.job_id, fmt="txt", mode="full")
    lines = [ln for ln in view.content.splitlines() if ln.strip()]
    assert lines
    assert all(ln.rstrip()[-1] in ".?!" for ln in lines), lines[:3]
    assert all(ln[0].isupper() for ln in lines if ln[0].isalpha())


@pytest.mark.db
async def test_an_unsupported_language_is_untouched(app, long_wav) -> None:
    app.engine._language = "hi"
    created = await app.transcription.create(str(long_wav), language="hi")
    assert await app.runner.tick() is JobStatus.DONE

    assert app.punctuator.calls == [], "the model must not have been asked"
    view = await app.reads.get(created.job_id, fmt="txt", mode="full")
    assert view.content.splitlines()[0].startswith("hello world")


@pytest.mark.db
async def test_the_asr_model_is_evicted_before_punctuating(app, long_wav) -> None:
    """Whisper peaks at 3.04GB and the punctuator needs ~1.1GB; together they
    exceed the 4GB self cap."""
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    assert app.engine.unload_calls > 0
    assert app.engine.loaded is False


@pytest.mark.db
async def test_the_render_cache_is_dropped_after_a_correction(
    app, long_wav
) -> None:
    """Segments are the source of truth; correcting them invalidates every
    cached render."""
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()
    uid = uuid.UUID(created.job_id)

    srt = await app.transcripts.render(uid, Format.SRT)
    assert "." in srt
    # The cache was populated after the correction, so it must agree with the
    # committed segments.
    assert await app.transcripts.cached_render(uid, Format.SRT) == srt


@pytest.mark.db
async def test_running_it_twice_changes_nothing(app, long_wav) -> None:
    """Idempotent, which is what lets the stage share Finalize's checkpoint and
    survive a crash between the two."""
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()
    first = await app.reads.get(created.job_id, fmt="txt", mode="full")

    job = await app.jobs.get(uuid.UUID(created.job_id))
    ctx = await app.context_factory(job)
    from transcriptor.services.stages.punctuate import PunctuateStage

    await PunctuateStage(punctuator=app.punctuator).run(ctx)
    second = await app.reads.get(created.job_id, fmt="txt", mode="full")
    assert second.content == first.content


@pytest.mark.db
async def test_timings_are_never_altered(app, long_wav) -> None:
    """Only the words change — a correction must not move a subtitle."""
    created = await app.transcription.create(str(long_wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    from transcriptor.services.stages.detect import DetectStage
    from transcriptor.services.stages.extract import ExtractStage
    from transcriptor.services.stages.punctuate import PunctuateStage
    from transcriptor.services.stages.resolve import ResolveStage
    from transcriptor.services.stages.transcribe import TranscribeStage

    await ResolveStage().run(ctx)
    await ctx.refresh()
    await ExtractStage().run(ctx)
    await DetectStage(detector=app.detector).run(ctx)
    await TranscribeStage().run(ctx)

    before = [(s.idx, s.start, s.end) for s in await app.transcripts.segments(ctx.job_id)]
    await PunctuateStage(punctuator=app.punctuator).run(ctx)
    after = [(s.idx, s.start, s.end) for s in await app.transcripts.segments(ctx.job_id)]
    assert after == before


@pytest.mark.db
async def test_no_punctuator_leaves_the_transcript_as_transcribed(
    app, long_wav
) -> None:
    """A missing dependency must degrade, not fail: the transcript is correct
    without punctuation, only less readable."""
    from transcriptor.services.stages.punctuate import PunctuateStage

    stage = PunctuateStage(punctuator=None)
    job = await app.jobs.get(
        uuid.UUID((await app.transcription.create(str(long_wav))).job_id)
    )
    ctx = await app.context_factory(job)
    assert await stage.satisfied(ctx) is True


# ------------------------------------------------------------- real model


@pytest.mark.slow
def test_the_real_model_restores_the_degraded_text() -> None:
    """The exact text this feature exists for, taken from a real transcript."""
    pytest.importorskip("transformers", reason="run: uv sync --extra punctuate")
    from transcriptor.adapters.hf_punctuator import HuggingFacePunctuator

    punctuator = HuggingFacePunctuator()
    degraded = (
        "this is the crack underneath all of it stay with me and you will know "
        "how to close it so how does"
    )
    restored = punctuator.restore(degraded)

    assert restored != degraded
    assert restored.count(".") >= 2, restored
    assert restored[0].isupper()
    assert set(restored.lower().replace(".", "").replace(",", "").replace("?", "").split()) == set(
        degraded.split()
    ), "no word may be added, lost or altered"


@pytest.mark.slow
def test_the_real_model_leaves_good_text_alone() -> None:
    pytest.importorskip("transformers", reason="run: uv sync --extra punctuate")
    from transcriptor.adapters.hf_punctuator import HuggingFacePunctuator

    good = "It didn't just get hacked."
    assert HuggingFacePunctuator().restore(good) == good
