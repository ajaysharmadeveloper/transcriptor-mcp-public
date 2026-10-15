"""The Detect stage in the pipeline: checkpointing, resume, and remapping.

Uses FakeDetector so the timeline is chosen rather than measured — these are
about what the pipeline does with a timeline, not about whether Silero is any
good. That is ``test_vad.py``.
"""

from __future__ import annotations

import uuid

import pytest

from transcriptor.domain.job import JobStatus
from transcriptor.domain.transcript import Timeline
from transcriptor.services.stages.detect import SPEECH_WAV_NAME

pytestmark = pytest.mark.db


async def test_the_timeline_is_stored_on_the_job(app, wav) -> None:
    created = await app.transcription.create(str(wav))
    await app.runner.tick()

    job = await app.jobs.get(uuid.UUID(created.job_id))
    assert job.speech_timeline is not None
    assert job.speech_timeline.regions, "the checkpoint must be a real timeline"


async def test_a_silent_recording_is_reported_not_hallucinated(app, wav) -> None:
    """The failure VAD exists to prevent: confident invented sentences over
    silence."""
    app.detector._timeline = Timeline([])
    created = await app.transcription.create(str(wav))

    assert await app.runner.tick() is JobStatus.FAILED
    status = await app.reads.status(created.job_id)
    assert status.error_code == "no_speech_detected"

    view = await app.reads.get(created.job_id, fmt="txt", mode="full")
    assert view.content == "", "not one invented word"


async def test_the_engine_never_sees_the_silence(app, long_wav) -> None:
    """Only the speech regions reach Whisper.  The 12s fixture is cut to 4s of
    'speech', so the chunk plan must cover 4s, not 12s."""
    app.detector._timeline = Timeline([(2.0, 4.0), (8.0, 10.0)])
    created = await app.transcription.create(str(long_wav))
    assert await app.runner.tick() is JobStatus.DONE

    furthest = max(s.end_s or 0 for s in app.engine.transcribe_calls)
    assert furthest == pytest.approx(4.0, abs=0.01), (
        "the engine's clock must end at the speech duration, not the recording's"
    )

    # Chunks overlap, so the total asked for exceeds the speech slightly — but
    # it must be nowhere near the 12s of recording.
    transcribed = sum((s.end_s or 0) - s.start_s for s in app.engine.transcribe_calls)
    assert 4.0 <= transcribed < 7.0, (
        f"the engine was given {transcribed:.1f}s for 4s of speech in a 12s file"
    )


async def test_timestamps_are_remapped_onto_the_original_clock(app, long_wav) -> None:
    """Without the remap every timestamp would refer to a compressed timeline
    the caller never sees."""
    app.detector._timeline = Timeline([(2.0, 4.0), (8.0, 10.0)])
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    segments = await app.transcripts.segments(uuid.UUID(created.job_id))
    assert segments
    for s in segments:
        inside = any(
            start - 0.01 <= s.start <= end + 0.01 for start, end in [(2.0, 4.0), (8.0, 10.0)]
        )
        assert inside, f"segment at {s.start:.2f}s falls in a silence"
    assert segments[0].start >= 2.0, "nothing before the first speech region"
    assert segments[-1].end <= 10.05, "nothing after the last"


async def test_a_segment_never_lands_in_a_gap(app, long_wav) -> None:
    app.detector._timeline = Timeline([(0.0, 2.0), (9.0, 11.0)])
    created = await app.transcription.create(str(long_wav))
    await app.runner.tick()

    segments = await app.transcripts.segments(uuid.UUID(created.job_id))
    for s in segments:
        assert not (2.05 < s.start < 8.95), f"segment starts inside the gap at {s.start}"


# --------------------------------------------------------------------- resume


async def test_detect_is_satisfied_once_the_timeline_and_audio_exist(
    app, wav
) -> None:
    created = await app.transcription.create(str(wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    from transcriptor.services.stages.detect import DetectStage
    from transcriptor.services.stages.extract import ExtractStage
    from transcriptor.services.stages.resolve import ResolveStage

    stage = DetectStage(detector=app.detector)
    await ResolveStage().run(ctx)
    await ctx.refresh()
    await ExtractStage().run(ctx)

    assert await stage.satisfied(ctx) is False
    await stage.run(ctx)
    await ctx.refresh()
    assert await stage.satisfied(ctx) is True


async def test_a_resumed_job_rebuilds_the_audio_without_rerunning_silero(
    app, wav
) -> None:
    """The stored timeline is the point of the column: after a crash the speech
    audio is rebuilt from it rather than paying for detection again."""
    created = await app.transcription.create(str(wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    from transcriptor.services.stages.detect import DetectStage
    from transcriptor.services.stages.extract import ExtractStage
    from transcriptor.services.stages.resolve import ResolveStage

    stage = DetectStage(detector=app.detector)
    await ResolveStage().run(ctx)
    await ctx.refresh()
    await ExtractStage().run(ctx)
    await stage.run(ctx)
    await ctx.refresh()

    detections = len(app.detector.calls)

    # A crash wipes scratch but leaves Postgres intact.
    ctx.speech_wav.unlink()
    assert await stage.satisfied(ctx) is False
    await stage.run(ctx)

    assert len(app.detector.calls) == detections, "Silero must not run twice"
    assert ctx.speech_wav.exists(), "but the audio must be back"


async def test_the_speech_wav_lands_in_the_job_scratch(app, wav) -> None:
    created = await app.transcription.create(str(wav))
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    from transcriptor.services.stages.detect import DetectStage
    from transcriptor.services.stages.extract import ExtractStage
    from transcriptor.services.stages.resolve import ResolveStage

    await ResolveStage().run(ctx)
    await ctx.refresh()
    await ExtractStage().run(ctx)
    await DetectStage(detector=app.detector).run(ctx)

    assert ctx.speech_wav == ctx.job_scratch / SPEECH_WAV_NAME
    assert ctx.speech_wav.exists()
