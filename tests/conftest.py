"""Shared fixtures.

The default suite runs with no inference library, no model weights and no GPU —
that is what the ``ASREngine`` port buys.  Tests needing real inference are
marked ``slow``.

**Database tests need nothing installed.** They used to be marked ``db`` and
skipped for anyone without a running Postgres, which meant 181 of them — most
of the store, dedupe, resume and reaper coverage — ran on the author's machine
and nowhere else. Each test now gets its own SQLite file in a temporary
directory, so they run everywhere, on every CI runner, for every contributor.
The marker stays so `-m "not db"` still means something, but it no longer
gates anything.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from transcriptor.config import REPO_ROOT, Settings, get_settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings with the ambient env files ignored, so tests are hermetic."""
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest_asyncio.fixture
async def db(tmp_path):
    """A :class:`Database` on its own file, migrations applied.

    A file rather than ``:memory:``, and that is deliberate: an in-memory
    SQLite database is per-connection, so a pooled engine would hand different
    connections different empty databases and the failures would look like
    phantom data loss. A temp file behaves like the real thing, including the
    WAL and the pragmas.
    """
    from transcriptor.adapters.db import Database, apply_migrations, create_engine_for

    settings = Settings(_env_file=None, data_dir=tmp_path)

    engine = create_engine_for(settings)
    async with engine.connect() as conn:
        await apply_migrations(conn)

    database = Database(engine)
    try:
        yield database
    finally:
        await database.dispose()


@pytest_asyncio.fixture
async def jobs(db):
    from transcriptor.adapters.sqlite_job_store import SqliteJobStore

    return SqliteJobStore(db)


@pytest_asyncio.fixture
async def transcripts(db):
    """Wrapped exactly as server.py wraps it.

    Composing differently here would mean the decorator is never exercised, and
    a method missing from it would only ever fail in production.
    """
    from transcriptor.adapters.buffered_store import BufferedTranscriptStore
    from transcriptor.adapters.sqlite_transcript_store import SqliteTranscriptStore

    return BufferedTranscriptStore(SqliteTranscriptStore(db), max_segments=500)


@pytest_asyncio.fixture
async def source_id(jobs) -> uuid.UUID:
    from transcriptor.domain.source import SourceKind, SourceRef

    ref = SourceRef(
        kind=SourceKind.FILE,
        raw="/tmp/example.mp4",
        key="/tmp/example.mp4",
        platform="local",
    )
    return await jobs.upsert_source(ref, fingerprint="123:456")


# --------------------------------------------------------------- application


def _make_wav(tmp_path, seconds: float, name: str = "tone.wav"):
    """A real 16kHz mono WAV, generated with ffmpeg.

    Real audio rather than a stub, so the extract stage and the audio hash are
    exercised for what they actually do.
    """
    import shutil as _shutil
    import subprocess

    if _shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not on PATH")

    dest = tmp_path / name
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(dest),
        ],
        check=True,
    )
    return dest


@pytest.fixture
def wav(tmp_path):
    """3 seconds — two chunks at the test chunk size, so stitching runs."""
    return _make_wav(tmp_path, 3)


@pytest.fixture
def long_wav(tmp_path):
    """12 seconds — many chunks, so resume and per-chunk commits are real."""
    return _make_wav(tmp_path, 12, name="long.wav")


@pytest.fixture
def engine():
    from transcriptor.adapters.fake_engine import FakeEngine

    return FakeEngine(segment_s=1.0)


@pytest.fixture
def executors():
    from concurrent.futures import ThreadPoolExecutor

    inference = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
    cpu = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu")
    try:
        yield inference, cpu
    finally:
        inference.shutdown(wait=False, cancel_futures=True)
        cpu.shutdown(wait=False, cancel_futures=True)


@pytest.fixture
def governor():
    """A governor over a monitor that always reports a comfortable machine, so
    admission is not at the mercy of whatever else is running."""
    import datetime as dt

    from transcriptor.domain.pressure import ResourceSample
    from transcriptor.services.governor import ResourceGovernor

    class QuietMonitor:
        gb = 12.0

        def sample(self) -> ResourceSample:
            return ResourceSample(
                rss_gb=0.2,
                available_gb=self.gb,
                swap_used_gb=0.0,
                swap_delta_gb=0.0,
                total_gb=16.0,
                at=dt.datetime.now(dt.UTC),
            )

    return ResourceGovernor(QuietMonitor())


@pytest.fixture
def detector():
    """Most of the suite uses synthetic tones, which real VAD correctly calls
    silence.  Running Silero there would fail every pipeline test with
    `no_speech_detected` for reasons unrelated to what they check."""
    from transcriptor.adapters.fake_detector import FakeDetector

    return FakeDetector()


@pytest.fixture
def gateway():
    """No yt-dlp by default: the suite must not reach the network."""
    return None


@pytest.fixture
def punctuator():
    """Deterministic, so the stage is tested without a 1GB download."""
    from transcriptor.adapters.fake_punctuator import FakePunctuator

    return FakePunctuator()


@pytest.fixture
def app(
    jobs, transcripts, engine, executors, governor, detector, gateway,
    punctuator, tmp_path,
):
    """The whole application layer, wired exactly as server.py wires it, but
    with FakeEngine — no MLX, no model weights, no Apple Silicon."""
    import asyncio
    from types import SimpleNamespace

    from transcriptor.adapters.ffmpeg_audio import FfmpegAudio
    from transcriptor.services.admission import AdmissionController
    from transcriptor.services.health import HealthService, TaskHealth
    from transcriptor.services.pipeline import JobContext, Pipeline
    from transcriptor.services.reads import ReadService
    from transcriptor.services.reaper import Reaper
    from transcriptor.services.runner import Heartbeat, JobRunner
    from transcriptor.services.stages.detect import DetectStage
    from transcriptor.services.stages.extract import ExtractStage
    from transcriptor.services.stages.fetch import FetchStage
    from transcriptor.services.stages.finalize import FinalizeStage
    from transcriptor.services.stages.punctuate import PunctuateStage
    from transcriptor.services.stages.resolve import ResolveStage
    from transcriptor.services.stages.transcribe import TranscribeStage
    from transcriptor.services.transcription import TranscriptionService

    inference, cpu = executors
    extractor = FfmpegAudio(cpu_executor=cpu)
    scratch = tmp_path / "scratch"
    wake = asyncio.Event()
    # Small windows so a few seconds of fixture audio still spans several
    # chunks, and the stitching path is exercised for real.
    chunk_s, overlap_s = 2.0, 0.5
    # Short enough that the pause-then-abort rung runs in test time.
    pause_max_s = 0.2

    async def context_factory(job):
        return JobContext(
            job=job, jobs=jobs, transcripts=transcripts, engine=engine,
            extractor=extractor, inference_executor=inference, cpu_executor=cpu,
            scratch_dir=scratch, max_duration_s=14400, language=job.language,
            chunk_s=chunk_s, overlap_s=overlap_s,
            pause_max_s=pause_max_s, pressure_poll_s=0.05,
        )

    context_factory.engine_memory_gb = engine.memory_estimate_gb

    pipeline = Pipeline(
        [
            ResolveStage(gateway=gateway),
            FetchStage(gateway=gateway),
            ExtractStage(),
            DetectStage(detector=detector),
            TranscribeStage(governor=governor),
            PunctuateStage(punctuator=punctuator),
            FinalizeStage(),
        ]
    )
    runner = JobRunner(
        jobs=jobs, pipeline=pipeline, admission=AdmissionController(),
        governor=governor, context_factory=context_factory, wake=wake,
    )
    reaper = Reaper(jobs, stale_after_s=90)
    tasks = TaskHealth()

    from transcriptor.adapters.local_files import LocalFiles
    from transcriptor.adapters.renderers import SrtRenderer
    from transcriptor.services.eviction import ModelEvictor
    from transcriptor.services.lifecycle import LifecycleService
    from transcriptor.services.clips import ClipService
    from transcriptor.services.prune import ScratchPruner

    transcription = TranscriptionService(
        jobs, model="fake-engine", wake=wake, gateway=gateway, extractor=extractor,
    )
    return SimpleNamespace(
        jobs=jobs, transcripts=transcripts, engine=engine, detector=detector,
        punctuator=punctuator, scratch=scratch,
        evictor=ModelEvictor(
            engine=engine, governor=governor, runner=runner, base_idle_s=600,
        ),
        wake=wake, pipeline=pipeline, runner=runner, reaper=reaper, tasks=tasks,
        heartbeat=Heartbeat(jobs=jobs, runner=runner),
        transcription=transcription,
        reads=ReadService(jobs, transcripts, runner=runner),
        health=HealthService(
            jobs=jobs, engine=engine, governor=governor, tasks=tasks,
            reaper=reaper, runner=runner, model="fake-engine",
        ),
        context_factory=context_factory,
        files=LocalFiles(cpu_executor=cpu),
        pruner=ScratchPruner(
            jobs=jobs, files=LocalFiles(cpu_executor=cpu), scratch_dir=scratch,
            max_age_days=0.0,
        ),
        clips=ClipService(jobs, transcripts, renderer=SrtRenderer()),
        lifecycle=LifecycleService(
            jobs, transcripts, files=LocalFiles(cpu_executor=cpu),
            scratch_dir=scratch, runner=runner,
            transcription=transcription,
        ),
    )


# ------------------------------------------------------------------ vad audio


def _ffmpeg_wav(tmp_path, name: str, *lavfi: str):
    import shutil as _shutil
    import subprocess

    if _shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not on PATH")
    dest = tmp_path / name
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *lavfi,
         "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(dest)],
        check=True,
    )
    return dest


@pytest.fixture
def silence_wav(tmp_path):
    """Digital silence.  Must yield zero segments, never a hallucination."""
    return _ffmpeg_wav(tmp_path, "silence.wav",
                       "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "8")


@pytest.fixture
def music_bed_wav(tmp_path):
    """A tone bed — non-speech audio that is not silent, which is the case
    Whisper most reliably invents sentences over."""
    return _ffmpeg_wav(
        tmp_path, "music.wav",
        "-f", "lavfi", "-i",
        "sine=frequency=220:duration=8,tremolo=f=3:d=0.7",
    )


@pytest.fixture
def speech_wav(tmp_path):
    """Real recorded speech, taken from the acceptance video.

    Synthetic audio cannot exercise a speech detector; this is the only fixture
    that can prove Silero finds speech rather than merely rejecting tones.
    """
    import shutil as _shutil
    import subprocess

    # Synthetic tones cannot exercise a speech detector, so these tests need
    # audio a model will actually recognise as speech. Twenty seconds is enough.
    # Not committed — it is someone's media — so the tests skip without it.
    candidates = (
        REPO_ROOT / "tests" / "fixtures" / "speech-sample.wav",
        REPO_ROOT / "local_video_test" / "speech-sample.wav",
        REPO_ROOT / "local_video_test" / "test-video.mp4",
    )
    for candidate in candidates:
        if candidate.exists():
            source = candidate
            break
    else:
        pytest.skip(
            "no speech fixture — tests/fixtures/FIXTURES.md has the one-line "
            "ffmpeg command to make one from any video with speech in it"
        )

    if source.suffix == ".wav":
        return source

    if _shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not on PATH")
    dest = tmp_path / "speech.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
         "-vn", "-t", "20", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
         str(dest)],
        check=True,
    )
    return dest
