"""URL ingestion, the captions shortcut, and layer-2 dedupe.

Driven through a fake gateway: the suite must not reach the network, and yt-dlp
against a live site is a test of YouTube, not of Transcriptor.  The real URL is
exercised by hand — see FIXTURES.md.
"""

from __future__ import annotations

import uuid

import pytest

from transcriptor.domain.job import JobStatus, Origin
from transcriptor.domain.source import CaptionPolicy, SourceMeta
from transcriptor.domain.transcript import Segment

pytestmark = pytest.mark.db

URL = "https://youtu.be/7ksp9N7vguE?si=jDRyqzNlBauV804x"
CLEAN = "https://www.youtube.com/watch?v=7ksp9N7vguE"


class FakeGateway:
    """A MediaGateway that serves a local file instead of the network."""

    def __init__(self, *, media, meta=None, captions=None) -> None:
        self._media = media
        self._meta = meta or SourceMeta(
            duration_s=3.0, has_audio=True, title="A Test Video",
            uploader="Someone", has_manual_captions=False, has_auto_captions=False,
        )
        self._captions = captions
        self.probes = 0
        self.downloads = 0
        self.caption_calls: list[CaptionPolicy] = []

    async def probe(self, ref):
        self.probes += 1
        return self._meta

    async def fetch_audio(self, ref, dest, *, cancel=None):
        import shutil

        self.downloads += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        target = dest.with_suffix(".wav")
        shutil.copy(self._media, target)
        return target

    async def fetch_captions(self, ref, policy):
        self.caption_calls.append(policy)
        return self._captions


@pytest.fixture
def gateway(wav):
    return FakeGateway(media=wav)


# ------------------------------------------------------------------- probe


async def test_probe_reports_metadata_without_downloading(app, gateway) -> None:
    result = await app.transcription.probe(URL)

    assert result.title == "A Test Video"
    assert result.uploader == "Someone"
    assert result.platform == "youtube"
    assert result.source_key == CLEAN, "the tracking parameter is stripped"
    assert result.estimated_transcribe_s is not None
    assert result.existing is None
    assert gateway.downloads == 0, "probe must not download"


async def test_probe_reports_an_existing_job_for_the_same_settings(
    app, gateway
) -> None:
    created = await app.transcription.create(URL, captions=CaptionPolicy.NEVER)
    result = await app.transcription.probe(URL, captions=CaptionPolicy.NEVER)

    assert result.existing is not None
    assert result.existing["job_id"] == created.job_id
    assert result.existing["status"] == "queued"


async def test_probe_reports_nothing_for_different_settings(app, gateway) -> None:
    """Passing the settings you would use for create is what makes `existing`
    report what *would* happen rather than what might."""
    await app.transcription.create(URL, captions=CaptionPolicy.NEVER)
    other = await app.transcription.probe(URL, language="hi", captions=CaptionPolicy.NEVER)
    assert other.existing is None


async def test_probe_of_a_local_file_uses_ffprobe(app, wav) -> None:
    result = await app.transcription.probe(str(wav))
    assert result.platform == "local"
    assert result.duration_s == pytest.approx(3.0, abs=0.2)


# ---------------------------------------------------------------- download


async def test_a_url_job_downloads_and_transcribes(app, gateway) -> None:
    created = await app.transcription.create(URL, captions=CaptionPolicy.NEVER)
    assert await app.runner.tick() is JobStatus.DONE

    assert gateway.downloads == 1
    view = await app.reads.get(created.job_id, fmt="txt", mode="full")
    assert view.content.strip()


async def test_the_download_lands_in_the_job_scratch(app, gateway) -> None:
    created = await app.transcription.create(URL, captions=CaptionPolicy.NEVER)
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)

    from transcriptor.services.stages.fetch import FetchStage
    from transcriptor.services.stages.resolve import ResolveStage

    await ResolveStage(gateway=gateway).run(ctx)
    await ctx.refresh()
    await FetchStage(gateway=gateway).run(ctx)

    assert ctx.media_path is not None
    assert ctx.media_path.parent == ctx.job_scratch


async def test_two_forms_of_the_same_link_are_one_job(app, gateway) -> None:
    """Layer 1 — the shared URL and the clean URL must not start two
    transcriptions."""
    first = await app.transcription.create(URL, captions=CaptionPolicy.NEVER)
    second = await app.transcription.create(CLEAN, captions=CaptionPolicy.NEVER)

    assert second.job_id == first.job_id
    assert second.reused is True
    assert second.reuse_reason == "already_running"


# ---------------------------------------------------------------- captions


CAPTION_SEGMENTS = [
    Segment(0, 0.0, 2.0, "Human written captions."),
    Segment(1, 2.0, 4.0, "With correct punctuation."),
]


async def test_manual_captions_skip_inference_entirely(app, wav) -> None:
    """Human captions are usually better than Whisper — correct punctuation,
    correct names, correct jargon. Worth taking."""
    app.gateway = FakeGateway(
        media=wav, captions=(CAPTION_SEGMENTS, "captions_manual")
    )
    app.runner._context_factory = app.context_factory
    pipeline = _pipeline_with(app, app.gateway)

    created = await app.transcription.create(URL)
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)
    assert await pipeline.execute(ctx) is JobStatus.DONE

    assert app.engine.transcribe_calls == [], "not one call to the engine"
    job = await app.jobs.get(uuid.UUID(created.job_id))
    assert job.origin is Origin.CAPTIONS_MANUAL

    view = await app.reads.get(created.job_id, fmt="txt", mode="full")
    assert "Human written captions." in view.content


async def test_captions_never_forces_transcription(app, wav) -> None:
    gateway = FakeGateway(media=wav, captions=(CAPTION_SEGMENTS, "captions_manual"))
    pipeline = _pipeline_with(app, gateway, captions=CaptionPolicy.NEVER)

    created = await app.transcription.create(URL, captions=CaptionPolicy.NEVER)
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)
    ctx.captions = CaptionPolicy.NEVER
    assert await pipeline.execute(ctx) is JobStatus.DONE

    assert gateway.caption_calls == [], "the policy must be honoured before fetching"
    assert app.engine.transcribe_calls, "Whisper must actually have run"
    job = await app.jobs.get(uuid.UUID(created.job_id))
    assert job.origin is Origin.ASR


async def test_the_origin_always_says_where_the_transcript_came_from(
    app, wav
) -> None:
    """Every result reports its origin, so you always know what you actually got
    rather than guessing from the quality."""
    gateway = FakeGateway(media=wav, captions=(CAPTION_SEGMENTS, "captions_auto"))
    pipeline = _pipeline_with(app, gateway, captions=CaptionPolicy.PREFER)

    created = await app.transcription.create(URL, captions=CaptionPolicy.PREFER)
    job = await app.jobs.next_queued()
    ctx = await app.context_factory(job)
    ctx.captions = CaptionPolicy.PREFER
    await pipeline.execute(ctx)

    job = await app.jobs.get(uuid.UUID(created.job_id))
    assert job.origin is Origin.CAPTIONS_AUTO


def _pipeline_with(app, gateway, *, captions=CaptionPolicy.MANUAL_ONLY):
    from transcriptor.services.pipeline import Pipeline
    from transcriptor.services.stages.detect import DetectStage
    from transcriptor.services.stages.extract import ExtractStage
    from transcriptor.services.stages.fetch import FetchStage
    from transcriptor.services.stages.finalize import FinalizeStage
    from transcriptor.services.stages.resolve import ResolveStage
    from transcriptor.services.stages.transcribe import TranscribeStage

    return Pipeline(
        [
            ResolveStage(gateway=gateway),
            FetchStage(gateway=gateway, captions=captions),
            ExtractStage(),
            DetectStage(detector=app.detector),
            TranscribeStage(),
            FinalizeStage(),
        ]
    )


# ------------------------------------------------------------ layer 2


async def test_identical_audio_is_not_transcribed_twice(app, wav, tmp_path) -> None:
    """Layer 2 — the same audio arriving from a different source must not cost a
    second hour of GPU."""
    import shutil

    created = await app.transcription.create(str(wav))
    assert await app.runner.tick() is JobStatus.DONE
    first_calls = len(app.engine.transcribe_calls)

    # A byte-identical copy at a different path: layer 1 cannot see it.
    twin = tmp_path / "a-different-name.wav"
    shutil.copy(wav, twin)

    second = await app.transcription.create(str(twin))
    assert second.job_id != created.job_id, "different path, so a different job"
    assert await app.runner.tick() is JobStatus.DONE

    assert len(app.engine.transcribe_calls) == first_calls, (
        "identical audio must not be transcribed again"
    )
    original = await app.reads.get(created.job_id, fmt="txt", mode="full")
    copy = await app.reads.get(second.job_id, fmt="txt", mode="full")
    assert copy.content == original.content


async def test_reused_audio_carries_the_detected_language(app, wav, tmp_path) -> None:
    """Reporting `language: null` for a byte-identical transcript would be a
    worse answer than the one the original job already produced."""
    import shutil

    created = await app.transcription.create(str(wav))
    await app.runner.tick()
    original = await app.reads.status(created.job_id)
    assert original.language == "en"

    twin = tmp_path / "same-audio-different-name.wav"
    shutil.copy(wav, twin)
    second = await app.transcription.create(str(twin))
    await app.runner.tick()

    assert (await app.reads.status(second.job_id)).language == original.language


async def test_force_skips_layer_2_as_well_as_layer_1(app, wav, tmp_path) -> None:
    """§13: "force=true skips layers 1 and 2". Layer 1 could honour it at create
    time; layer 2 runs a stage later, so the flag has to survive on the job — it
    did not, and a forced job silently had its transcript copied from a twin."""
    import shutil

    created = await app.transcription.create(str(wav))
    await app.runner.tick()
    calls = len(app.engine.transcribe_calls)

    twin = tmp_path / "identical-audio.wav"
    shutil.copy(wav, twin)

    forced = await app.transcription.create(str(twin), force=True)
    assert await app.runner.tick() is JobStatus.DONE

    assert len(app.engine.transcribe_calls) > calls, (
        "force must transcribe again rather than copying the identical twin"
    )


async def test_a_forced_job_stays_forced_across_a_restart(app, wav) -> None:
    """Read off the job row, not held in memory, so a resumed forced job does
    not quietly start reusing."""
    created = await app.transcription.create(str(wav), force=True)
    job = await app.jobs.get(uuid.UUID(created.job_id))
    assert job.forced is True

    unforced = await app.transcription.create(str(wav), force=True)
    assert (await app.jobs.get(uuid.UUID(unforced.job_id))).forced is True


# ------------------------------------------------------- the launch environment


def test_a_js_runtime_is_offered_to_ytdlp() -> None:
    """YouTube serves a JavaScript challenge and yt-dlp enables only `deno` by
    default. Extraction without a runtime is deprecated and silently drops
    formats, so an installed node or bun must be used rather than wasted."""
    import shutil

    from transcriptor.adapters.ytdlp_gateway import YtDlpMediaGateway

    flags = YtDlpMediaGateway()._runtime_flags()
    if any(shutil.which(b) for b in ("deno", "node", "bun")):
        assert flags[0] == "--js-runtimes"
        name, path = flags[1].split(":", 1)
        assert shutil.which(name) == path
    else:
        assert flags == [], "nothing installed means no flag, not a broken one"


async def test_a_download_honours_its_cancel_token(wav, tmp_path) -> None:
    """The token was accepted and ignored, so cancelling a job left yt-dlp
    downloading to completion."""
    from transcriptor.adapters.ytdlp_gateway import YtDlpMediaGateway
    from transcriptor.domain.errors import Cancelled
    from transcriptor.domain.job import CancelToken
    from transcriptor.domain.source import parse_source

    gateway = YtDlpMediaGateway()
    if not gateway.available:
        pytest.skip("yt-dlp is not installed")

    token = CancelToken()
    token.cancel("stopped before it began")
    with pytest.raises(Cancelled):
        await gateway.fetch_audio(
            parse_source("https://example.com/nothing.mp4"),
            tmp_path / "out",
            cancel=token,
        )
