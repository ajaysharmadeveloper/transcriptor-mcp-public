"""Composition root.

The only module that may import ``adapters``.  Owns process lifespan, task
supervision, and the ``transcriptor-mcp`` CLI.

Subcommands
    (none)     run the FastMCP stdio server        [checkpoint 3]
    migrate    apply numbered .sql migrations      [checkpoint 2]
    setup      warm up the model, pin HF_HOME      [checkpoint 4]
    doctor     environment preflight               [checkpoint 1]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import itertools
import logging
import os
import platform
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from transcriptor import tools
from transcriptor.config import Settings, get_settings, masked_dsn
from transcriptor.domain.source import CaptionPolicy

log = logging.getLogger(__name__)

# Where the tools this server shells out to actually live. A GUI-launched
# process — which is how Claude Desktop starts this — inherits a minimal PATH of
# roughly /usr/bin:/bin:/usr/sbin:/sbin. Homebrew's ffmpeg and the venv's yt-dlp
# are both invisible to it, so every subprocess fails with "not on PATH" even
# though the binaries are installed and `make doctor` passes in a terminal.
#
# The venv's own bin comes from sys.executable, so this follows the install
# rather than assuming a location.
#
# The rest is per-platform, because "where does a package manager put things"
# has a different answer on each. Windows needs none: there is no equivalent
# convention, and shutil.which already applies PATHEXT to find ffmpeg.exe.
_EXTRA_PATHS = {
    "darwin": ("/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin"),
    "linux": ("/usr/local/bin", "/usr/bin", "/bin", "/snap/bin"),
    "win32": (),
}


def _tool_path_candidates() -> list[str]:
    """Directories to prepend, most specific first.

    The venv's own script directory is ``bin`` on POSIX and ``Scripts`` on
    Windows, and it is derived from ``sys.executable`` either way so it follows
    the install rather than assuming a layout.
    """
    venv_bin = Path(sys.executable).parent
    return [str(venv_bin), *_EXTRA_PATHS.get(sys.platform, ())]


def ensure_tool_path() -> str:
    """Put the venv and the platform's usual binary directories on PATH.

    Called before anything resolves a binary, so `doctor` diagnoses the same
    environment `serve` will run in.
    """
    existing = os.environ.get("PATH", "").split(os.pathsep)
    additions = [
        c for c in _tool_path_candidates() if c not in existing and Path(c).is_dir()
    ]
    if additions:
        os.environ["PATH"] = os.pathsep.join([*additions, *existing])
    return os.environ["PATH"]


# What to tell someone whose ffmpeg is missing. Naming the wrong package
# manager is worse than naming none: it sends them to install Homebrew on a
# Debian box.
_FFMPEG_HINT = {
    "darwin": "brew install ffmpeg",
    "linux": "sudo apt install ffmpeg   (or dnf/pacman/zypper equivalent)",
    "win32": "winget install Gyan.FFmpeg   (or: choco install ffmpeg)",
}


# --------------------------------------------------------------------------- doctor

OK, WARN, FAIL = "ok", "warn", "fail"

_MARK = {OK: "\033[32m✓\033[0m", WARN: "\033[33m!\033[0m", FAIL: "\033[31m✗\033[0m"}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    fix: str | None = None


def _check_platform(s: Settings) -> Check:
    """Can the *selected* engine run on this machine?

    This used to hard-fail anything that was not arm64, which was right when
    MLX was the only engine and wrong the moment there were two.  What matters
    is not the architecture but whether the engine chosen for this machine can
    actually run here, so that is what it reports.
    """
    system = platform.system()
    machine = platform.machine()
    py = platform.python_version()
    engine = s.resolved_engine
    where = f"{system} {machine}, python {py}"

    if engine == "fake":
        return Check("platform", WARN, f"{where} — TRANSCRIPTOR_FAKE_ENGINE=1 is set")

    if engine == "mlx":
        if machine != "arm64" or system != "Darwin":
            return Check(
                "platform",
                FAIL,
                f"{where} — the mlx engine needs Apple Silicon",
                "unset TRANSCRIPTOR_ENGINE to fall back to faster-whisper, or "
                "run on an arm64 interpreter (a Rosetta python cannot load MLX)",
            )
        if importlib.util.find_spec("mlx_whisper") is None:
            return Check(
                "platform", FAIL, f"{where} — mlx-whisper is not installed",
                "uv sync --extra mlx",
            )
        return Check("platform", OK, f"{where} — engine mlx (metal)")

    if importlib.util.find_spec("faster_whisper") is None:
        return Check(
            "platform",
            FAIL,
            f"{where} — faster-whisper is not installed",
            "uv sync --extra cuda   (or --extra cpu on a machine with no NVIDIA GPU)",
        )

    from transcriptor.adapters.faster_whisper_engine import cuda_device_count

    if cuda_device_count() > 0:
        return Check("platform", OK, f"{where} — engine faster-whisper (cuda)")

    # Not a failure. It works, and on a short file it is perfectly usable —
    # but someone who expected GPU speed should be told before they queue four
    # hours of audio, not after.
    return Check(
        "platform",
        WARN,
        f"{where} — engine faster-whisper (cpu): no CUDA device, expect "
        f"roughly real-time rather than faster",
        "install the CUDA build of CTranslate2 if this machine has an NVIDIA GPU",
    )


def _check_ffmpeg() -> Check:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        return Check(
            "ffmpeg",
            FAIL,
            f"not found: {', '.join(missing)} (searched {os.environ.get('PATH', '')})",
            _FFMPEG_HINT.get(sys.platform, "install ffmpeg and put it on PATH"),
        )
    return Check("ffmpeg", OK, f"{shutil.which('ffmpeg')}")


def _check_ytdlp() -> Check:
    """Needed for URL sources only, so its absence is a warning, not a failure."""
    found = shutil.which("yt-dlp")
    if found is None:
        return Check(
            "yt-dlp",
            WARN,
            "not found — URL sources will be refused, local files are unaffected",
            "uv sync --extra media",
        )
    return Check("yt-dlp", OK, found)


def _check_memory(s: Settings) -> Check:
    """The same monitor and the same pure thresholds the running server uses,
    so doctor cannot disagree with ``transcript_health``."""
    try:
        from transcriptor.adapters.psutil_monitor import PsutilResourceMonitor
        from transcriptor.domain.pressure import Pressure, Thresholds, classify
    except ImportError as exc:
        return Check("memory", FAIL, f"{exc}", "uv sync")

    thresholds = Thresholds(
        self_cap_gb=s.mem_self_cap_gb, system_floor_gb=s.mem_system_floor_gb
    )
    sample = PsutilResourceMonitor().sample()
    pressure = classify(sample, thresholds)
    detail = (
        f"{pressure.value} — available {sample.available_gb:.1f}GB of "
        f"{sample.total_gb:.0f}GB, floor {thresholds.system_floor_gb:.1f}GB, "
        f"swap {sample.swap_used_gb:.2f}GB (+{sample.swap_delta_gb:.2f}), "
        f"self cap {thresholds.self_cap_gb:.1f}GB"
    )
    if pressure is Pressure.CRITICAL:
        return Check("memory", FAIL, detail)
    if pressure is Pressure.ELEVATED:
        return Check("memory", WARN, detail)
    return Check("memory", OK, detail)


def _check_model(s: Settings) -> Check:
    # HF cache layout under HF_HOME: hub/models--<org>--<name>/
    model = s.resolved_model
    slug = "models--" + model.replace("/", "--")
    path = s.models_dir / "hub" / slug
    if not path.is_dir():
        return Check(
            "model",
            WARN,
            f"{model} not downloaded ({path})",
            f"transcriptor-mcp setup --model {model}",
        )
    size = (
        sum(
            f.stat().st_size
            for f in path.rglob("*")
            if f.is_file() and not f.is_symlink()
        )
        / 1024**3
    )
    return Check("model", OK, f"{model} present, {size:.1f}GB")


async def _check_database(s: Settings) -> tuple[Check, Check]:
    """Reachability and migration state.  Reported as two separate checks.

    "Unreachable" now means the file cannot be opened, which in practice means
    a permissions problem or a disk that is full — not a service that has not
    started yet. The whole class of "Docker was not up when the client launched
    me" failure went away with Postgres.
    """
    from transcriptor.adapters.db import database_path, url_for

    pending = Check("migrations", FAIL, "not checked — database unreachable")
    path = database_path(s)
    where = s.database_url or str(path)

    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        return Check("database", FAIL, "aiosqlite is not installed", "uv sync"), pending

    from sqlalchemy import text as sa_text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(url_for(s))
    try:
        async with engine.connect() as conn:
            version = (await conn.execute(sa_text("SELECT sqlite_version()"))).scalar()
            size = path.stat().st_size / 1024**2 if path.exists() else 0.0
            reach = Check(
                "database", OK, f"{where} — sqlite {version}, {size:.1f}MB"
            )

            on_disk = sorted(m.name for m in _migrations_dir().glob("*.sql"))
            has_table = (
                await conn.execute(
                    sa_text(
                        "SELECT count(*) FROM sqlite_master "
                        "WHERE type='table' AND name='schema_migrations'"
                    )
                )
            ).scalar()

            if not on_disk:
                mig = Check("migrations", WARN, "no .sql migrations on disk yet")
            elif not has_table:
                mig = Check(
                    "migrations",
                    FAIL,
                    f"schema_migrations missing, {len(on_disk)} on disk",
                    "transcriptor-mcp migrate",
                )
            else:
                rows = (
                    await conn.execute(sa_text("SELECT version FROM schema_migrations"))
                ).all()
                have = {r.version for r in rows}
                missing = [n for n in on_disk if n not in have]
                mig = (
                    Check("migrations", OK, f"{len(on_disk)} applied")
                    if not missing
                    else Check(
                        "migrations",
                        FAIL,
                        f"{len(missing)} pending: {', '.join(missing)}",
                        "transcriptor-mcp migrate",
                    )
                )
        return reach, mig
    except Exception as exc:  # surfaced verbatim
        return Check("database", FAIL, f"database_unreachable: {where} — {exc}"), pending
    finally:
        await engine.dispose()


def _db_path_for_log(s: Settings) -> str:
    from transcriptor.adapters.db import database_path

    return s.database_url or str(database_path(s))


def _migrations_dir() -> Path:
    return Path(__file__).resolve().parent / "migrations"


async def _doctor() -> int:
    s = get_settings()
    checks = [
        _check_platform(s),
        _check_ffmpeg(),
        _check_ytdlp(),
        _check_memory(s),
        _check_model(s),
    ]
    checks.extend(await _check_database(s))

    width = max(len(c.name) for c in checks)
    print(f"\ntranscriptor doctor  ·  data dir {s.data_dir}\n")
    for c in checks:
        print(f"  {_MARK[c.status]} {c.name.ljust(width)}  {c.detail}")
        if c.fix:
            print(f"    {' ' * width}   → {c.fix}")

    failed = [c.name for c in checks if c.status == FAIL]
    warned = [c.name for c in checks if c.status == WARN]
    print()
    if failed:
        print(f"  {len(failed)} failed: {', '.join(failed)}\n")
        return 1
    if warned:
        print(f"  all checks passed, {len(warned)} warning(s): {', '.join(warned)}\n")
        return 0
    print("  all checks passed\n")
    return 0


# --------------------------------------------------------------------------- serve


async def supervise(name: str, factory, tasks, *, backoff=(1, 2, 5, 15, 30)) -> None:
    """Restart a periodic task forever, with backoff, and record the crash.

    Every periodic task is supervised.  An unsupervised task that dies takes a
    guarantee with it — a dead heartbeat task means the reaper starts killing
    healthy jobs — so a crash is logged, surfaced through ``transcript_health``,
    and retried rather than silently ending the guarantee.
    """
    for delay in itertools.chain(backoff, itertools.repeat(60)):
        try:
            tasks.mark_healthy(name)
            await factory()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("task crashed", extra={"task": name})
            tasks.mark_degraded(name, f"{type(exc).__name__}: {exc}")
            await asyncio.sleep(delay)


def _wipe_scratch(scratch_dir: Path) -> None:
    """Scratch is wiped at startup as well as on job completion, so a crash
    cannot leave orphans behind."""
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir, ignore_errors=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)


def _build_engine(s: Settings, executor):
    """The engine is chosen here and nowhere else — that is what makes swapping
    it a config change (driver D7).

    ``executor`` is the single-worker inference pool, and passing it is not
    optional: MLX streams are per-thread, so loading the model on one thread and
    transcribing on another fails with "no Stream(gpu, N) in current thread".
    One executor with ``max_workers=1`` is what gives model residency exactly one
    owner (invariant I5).

    ``TRANSCRIPTOR_FAKE_ENGINE=1`` selects the deterministic fake, so the whole
    server can be started and driven with no MLX and no model weights.

    Which of the two real engines is built depends on the machine: MLX where
    Metal exists, CTranslate2 everywhere else.  The branch is here and nowhere
    else, so no layer above ``adapters`` has any idea which one it is talking
    to — that is the entire return on having a port (N1, driver D7).
    """
    engine = s.resolved_engine

    if engine == "fake":
        from transcriptor.adapters.fake_engine import FakeEngine

        return FakeEngine()

    if engine == "mlx":
        from transcriptor.adapters.whisper_mlx import WhisperMLXEngine

        return WhisperMLXEngine(
            model=s.resolved_model,
            models_dir=s.models_dir,
            executor=executor,
            self_cap_gb=s.mem_self_cap_gb,
        )

    if engine != "faster-whisper":
        raise ValueError(
            f"unknown TRANSCRIPTOR_ENGINE={engine!r} — "
            f"expected one of: auto, mlx, faster-whisper, fake"
        )

    from transcriptor.adapters.faster_whisper_engine import FasterWhisperEngine

    return FasterWhisperEngine(
        model=s.resolved_model,
        models_dir=s.models_dir,
        executor=executor,
        device=s.device,
        compute_type=s.compute_type,
    )


def _with_model(s: Settings, model: str) -> Settings:
    """A copy of the settings pinned to one model id.

    ``setup --model X`` has to warm exactly X, not whatever the environment
    would otherwise resolve to.
    """
    return s.model_copy(update={"model": model})


def _build_gateway(s: Settings):
    """yt-dlp, if it is installed.

    Optional rather than required: local files are the common case and must keep
    working on a machine with no yt-dlp, where a URL then fails with a named
    error instead of an import crash at startup.
    """
    from transcriptor.adapters.ytdlp_gateway import YtDlpMediaGateway

    gateway = YtDlpMediaGateway(
        cookies_from_browser=s.cookies_from_browser,
        cookies_file=s.cookies_file,
        max_download_mb=s.max_download_mb,
        remote_components=s.ytdlp_remote_components,
    )
    if not gateway.available:
        log.warning("yt-dlp is not on PATH — URL sources will be refused")
    return gateway


def _build_punctuator(s: Settings, executor):
    """The punctuation restorer, unless it is disabled or unavailable.

    Optional on purpose: the transcript is correct without it, only less
    readable, so a missing dependency must degrade rather than fail.
    """
    if not s.punctuate:
        log.info("punctuation restoration disabled")
        return None

    from transcriptor.adapters.hf_punctuator import HuggingFacePunctuator

    return HuggingFacePunctuator(model=s.punctuation_model, executor=executor)


def _build_detector():
    """Silero, unless it is unavailable or explicitly disabled.

    ``TRANSCRIPTOR_VAD=off`` treats the whole recording as speech, which is what
    the phase 1 pipeline did.  Useful for isolating a transcription problem from
    a detection one.
    """
    if os.environ.get("TRANSCRIPTOR_VAD", "").lower() in {"off", "0", "false"}:
        log.warning("VAD disabled — Whisper will see silence and may hallucinate")
        return None

    from transcriptor.adapters.silero_detector import SileroDetector

    return SileroDetector()


@asynccontextmanager
async def lifespan(_server):
    """Wire the application together and own every background task.

    Order matters: migrate before serving (never run on a half-built schema),
    wipe scratch, then reap once — after a crash the first thing to do is
    release orphaned jobs so the next request resumes rather than attaching to
    a dead one.
    """
    from transcriptor.adapters.buffered_store import BufferedTranscriptStore
    from transcriptor.adapters.db import apply_migrations, connect
    from transcriptor.adapters.ffmpeg_audio import FfmpegAudio
    from transcriptor.adapters.sqlite_job_store import SqliteJobStore, SqliteRequestLog
    from transcriptor.adapters.sqlite_transcript_store import SqliteTranscriptStore
    from transcriptor.adapters.psutil_monitor import PsutilResourceMonitor
    from transcriptor.domain.pressure import Thresholds
    from transcriptor.services.admission import AdmissionController
    from transcriptor.services.governor import ResourceGovernor
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
    from transcriptor.adapters.http_viewer import (
        TranscriptViewer,
        derive_token,
        safe_filename,
    )
    from transcriptor.adapters.local_files import LocalFiles
    from transcriptor.adapters.renderers import SrtRenderer
    from transcriptor.services.eviction import ModelEvictor
    from transcriptor.services.lifecycle import LifecycleService
    from transcriptor.services.clips import ClipService
    from transcriptor.services.prune import ScratchPruner
    from transcriptor.services.transcription import TranscriptionService

    s = get_settings()
    s.models_dir.mkdir(parents=True, exist_ok=True)
    _wipe_scratch(s.scratch_dir)

    # Weights land in the project, not $HOME.
    os.environ.setdefault("HF_HOME", str(s.models_dir))

    # max_workers=1 is not a tuning choice. It is the Semaphore(1): MLX is not
    # safe to drive from several threads for one model, and model residency gets
    # exactly one owner, so load/unload cannot race a transcribe (I5).
    inference = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
    cpu = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu")

    database = await connect(s)
    async with database.engine.connect() as conn:
        await apply_migrations(conn)

    jobs = SqliteJobStore(database)
    # A database outage mid-job must not lose a 40-minute transcription: failed
    # commits buffer and flush on reconnect.
    transcripts = BufferedTranscriptStore(
        SqliteTranscriptStore(database), max_segments=s.segment_buffer_max
    )
    requests = SqliteRequestLog(database)
    engine = _build_engine(s, inference)
    extractor = FfmpegAudio(cpu_executor=cpu)

    thresholds = Thresholds(
        self_cap_gb=s.mem_self_cap_gb, system_floor_gb=s.mem_system_floor_gb
    )
    governor = ResourceGovernor(
        PsutilResourceMonitor(
            allocator_gb=getattr(engine, "allocator_memory_gb", None),
            # Only the CTranslate2 engine on a CUDA box supplies this. On
            # unified memory it stays None and every VRAM rule stays inert,
            # because there the GPU already spends the memory psutil reports.
            vram_gb=getattr(engine, "vram_gb", None),
        ),
        thresholds=thresholds,
        poll_s=s.mem_poll_s,
    )
    admission = AdmissionController(
        thresholds=thresholds, max_concurrent=s.max_concurrent
    )
    reaper = Reaper(jobs, stale_after_s=s.stale_job_s, interval_s=s.reaper_s)
    tasks = TaskHealth()

    wake = asyncio.Event()
    detector = _build_detector()
    gateway = _build_gateway(s)
    default_captions = CaptionPolicy(s.captions_default)
    pipeline = Pipeline(
        [
            ResolveStage(gateway=gateway),
            FetchStage(gateway=gateway, captions=default_captions),
            ExtractStage(),
            DetectStage(detector=detector),
            TranscribeStage(governor=governor),
            PunctuateStage(punctuator=_build_punctuator(s, inference)),
            FinalizeStage(),
        ]
    )

    async def context_factory(job):
        return JobContext(
            job=job,
            jobs=jobs,
            transcripts=transcripts,
            engine=engine,
            extractor=extractor,
            inference_executor=inference,
            cpu_executor=cpu,
            scratch_dir=s.scratch_dir,
            max_duration_s=s.max_duration_s,
            language=job.language,
            captions=default_captions,
            chunk_s=s.chunk_s,
            overlap_s=s.overlap_s,
            pause_max_s=s.mem_pause_max_s,
            pressure_poll_s=min(s.mem_poll_s, 2.0),
        )

    context_factory.engine_memory_gb = engine.memory_estimate_gb
    # Zero for MLX, the fake, and any CPU build — see AdmissionController.
    context_factory.engine_vram_gb = getattr(engine, "vram_estimate_gb", 0.0)

    runner = JobRunner(
        jobs=jobs,
        pipeline=pipeline,
        admission=admission,
        governor=governor,
        context_factory=context_factory,
        wake=wake,
        keep_scratch=s.keep_scratch,
    )
    heartbeat = Heartbeat(jobs=jobs, runner=runner, interval_s=s.heartbeat_s)
    evictor = ModelEvictor(
        engine=engine, governor=governor, runner=runner, base_idle_s=s.idle_evict_s
    )
    files = LocalFiles(cpu_executor=cpu)
    pruner = ScratchPruner(
        jobs=jobs,
        files=files,
        scratch_dir=s.scratch_dir,
        max_age_days=s.scratch_ttl_days,
        interval_s=s.prune_interval_s,
    )

    # Release anything a previous process left behind, before serving.
    await reaper.run_once()

    background = [
        asyncio.create_task(supervise("runner", runner.run, tasks), name="runner"),
        asyncio.create_task(supervise("heartbeat", heartbeat.run, tasks), name="heartbeat"),
        asyncio.create_task(supervise("reaper", reaper.run, tasks), name="reaper"),
        asyncio.create_task(supervise("monitor", governor.run, tasks), name="monitor"),
        asyncio.create_task(supervise("evictor", evictor.run, tasks), name="evictor"),
        asyncio.create_task(supervise("pruner", pruner.run, tasks), name="pruner"),
    ]

    # engine.name, not settings.model: the result_key must describe what
    # actually produced the transcript. Otherwise a FakeEngine run would be
    # reused as though large-v3 had made it.
    transcription = TranscriptionService(
        jobs,
        model=engine.name,
        wake=wake,
        gateway=gateway,
        extractor=extractor,
    )

    reads = ReadService(jobs, transcripts, runner=runner)

    # The viewer takes a callable, not the service: adapters may not depend on
    # anything above them (N1). This closure is the seam.
    async def _render_for_viewer(job_id: str, fmt: str) -> tuple[str, str]:
        view = await reads.get(job_id, fmt=fmt, mode="full", max_chars=0)
        title = None
        job = await jobs.get(uuid.UUID(job_id))
        if job is not None:
            source = await jobs.get_source(job.source_id) or {}
            title = source.get("title")
        return view.content, safe_filename(title, job_id, fmt)

    viewer = TranscriptViewer(
        fetch=_render_for_viewer,
        host=s.viewer_host,
        port=s.viewer_port,
        # Derived, not random: Desktop runs two instances and only one gets the
        # port. Both must agree on the token or half the links 403.
        token=derive_token(str(s.database_url)),
    )
    if s.viewer_enabled:
        background.append(
            asyncio.create_task(
                supervise("viewer", viewer.run, tasks), name="viewer"
            )
        )

    deps = tools.Deps(
        transcription=transcription,
        reads=reads,
        health=HealthService(
            jobs=jobs,
            engine=engine,
            governor=governor,
            tasks=tasks,
            reaper=reaper,
            runner=runner,
            model=engine.name,
        ),
        lifecycle=LifecycleService(
            jobs,
            transcripts,
            files=files,
            scratch_dir=s.scratch_dir,
            runner=runner,
            transcription=transcription,
        ),
        clips=ClipService(jobs, transcripts, renderer=SrtRenderer()),
        requests=requests,
        viewer=viewer,
    )
    tools.register(_server, deps)

    log.info(
        "transcriptor ready",
        extra={
            "engine": s.resolved_engine,
            "model": s.resolved_model,
            "database": str(_db_path_for_log(s)),
        },
    )
    try:
        yield deps
    finally:
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)
        await engine.unload()
        inference.shutdown(wait=False, cancel_futures=True)
        cpu.shutdown(wait=False, cancel_futures=True)
        await database.dispose()
        _wipe_scratch(s.scratch_dir)


def build_server():
    from fastmcp import FastMCP

    return FastMCP(
        name="transcriptor",
        instructions=(
            "Local, on-device transcription. transcript_create takes a "
            "local absolute file path or a URL and returns a job_id "
            "immediately; poll transcript_status, then read transcript_get.\n\n"
            "Prefer mode='preview' — a full transcript can be tens of "
            "thousands of tokens. But that default exists to stop unprompted "
            "flooding, not to refuse a direct request: when the user asks for "
            "the whole transcript, mode='full' is the correct call.\n\n"
            "Responses carry download_urls and view_url once a job has "
            "produced anything. These are real, working links served by this "
            "process on loopback. When the user asks for a URL or a file, give "
            "them these — do not tell them no URL exists.\n\n"
            "Present a link as a markdown link inside a sentence, never inside "
            "a code block: fenced URLs are not clickable and have to be copied "
            "by hand. Prefer offering view_url on its own — that page links to "
            "all four formats, so one link beats five."
        ),
        lifespan=lifespan,
    )


def _serve(*, http: bool = False, host: str = "127.0.0.1", port: int = 8765) -> int:
    """stdio by default; HTTP on request.

    stdio is the right default: the client launches it, so there is no port, no
    token, no process to keep alive and no attack surface.  HTTP exists for the
    one case stdio cannot serve — an agent running on another machine — and it
    binds to loopback unless told otherwise, because this server has no
    authentication of its own.
    """
    logging.basicConfig(
        level=logging.INFO,
        # stdout is the MCP transport in stdio mode; never log to it.
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    server = build_server()
    if not http:
        server.run(transport="stdio", show_banner=False)
        return 0

    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "binding beyond loopback: Transcriptor has no authentication of its "
            "own, so put it behind something that does",
            extra={"host": host, "port": port},
        )
    server.run(transport="http", host=host, port=port, show_banner=False)
    return 0


# --------------------------------------------------------------------------- migrate


async def _migrate() -> int:
    """Apply pending migrations, or refuse to serve on a half-built schema.

    Uses the same engine the server does.  It previously opened a raw driver
    connection, which stopped working the moment ``apply_migrations`` moved to a
    SQLAlchemy connection — and only this CLI path was affected, so the server
    kept running while ``transcriptor-mcp migrate`` was broken.

    There is no "DSN is not set" branch any more: an unset URL is the normal
    case and means the project's own database file, which this creates.
    """
    from transcriptor.adapters.db import apply_migrations, create_engine_for

    from transcriptor.domain.errors import TranscriptorError

    s = get_settings()
    dsn = _db_path_for_log(s)
    engine = create_engine_for(s)
    try:
        async with engine.connect() as conn:
            applied = await apply_migrations(conn)
    except TranscriptorError as exc:
        print(f"{exc.code}: {exc.user_message}", file=sys.stderr)
        if statement := exc.context.get("statement"):
            print(f"\n  failing statement:\n    {statement}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surfaced, not masked
        print(f"database_unreachable: {dsn} — {exc}", file=sys.stderr)
        return 1
    finally:
        await engine.dispose()

    if applied:
        print(f"applied {len(applied)} migration(s):")
        for version in applied:
            print(f"  + {version}")
    else:
        print("schema is up to date")
    return 0


# ------------------------------------------------------------------ pg import


async def _import_postgres(dsn: str | None, *, dry_run: bool = False) -> int:
    """Copy an old Postgres database into the project's SQLite file.

    A one-off, for anyone who used Transcriptor before it stopped needing a
    database server. It reads Postgres and never writes to it, so a failed run
    costs nothing and can simply be repeated — every insert is
    ``INSERT OR IGNORE``.
    """
    from transcriptor.adapters.db import create_engine_for, database_path
    from transcriptor.adapters.pg_import import import_from_postgres

    s = get_settings()

    if not dsn:
        print(
            "give the old database with --from, e.g.\n"
            "  transcriptor-mcp import-postgres "
            "--from postgresql://postgres:PASSWORD@localhost:5432/transcriptor_db",
            file=sys.stderr,
        )
        return 1

    target = database_path(s)
    print(f"reading {masked_dsn(dsn)}")
    print(f"writing {target}" + ("  (dry run — nothing will be written)" if dry_run else ""))

    engine = create_engine_for(s)
    try:
        # The schema has to exist first. Saying so beats a stack trace naming
        # a missing table.
        from transcriptor.adapters.db import applied_versions

        async with engine.connect() as conn:
            if not await applied_versions(conn):
                print(
                    "\nthe target database has no schema yet — "
                    "run: transcriptor-mcp migrate",
                    file=sys.stderr,
                )
                return 1

        report = await import_from_postgres(dsn, engine, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001 - surfaced, not masked
        print(f"\nimport failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await engine.dispose()

    width = max(len(t) for t in report.source_rows)
    print()
    for table, found in report.source_rows.items():
        if dry_run:
            print(f"  {table.ljust(width)}  {found:>6} to copy")
            continue
        imported = report.imported.get(table, 0)
        skipped = report.skipped.get(table, 0)
        note = f"  ({skipped} already here)" if skipped else ""
        print(f"  {table.ljust(width)}  {imported:>6} imported{note}")

    if dry_run:
        print(f"\n  {sum(report.source_rows.values())} rows would be copied\n")
        return 0

    print(f"\n  {report.total_imported} rows imported, {report.total_skipped} skipped\n")
    if report.total_imported:
        print("  Postgres was not modified. Once you have checked the transcripts")
        print("  are readable, that database can be dropped.\n")
    return 0


# --------------------------------------------------------------------------- setup


async def _setup(model: str | None = None) -> int:
    """Download the model deliberately, rather than making the first user
    request wait ten minutes for it.

    ``HF_HOME`` is pinned to the project's models directory so the weights land
    where doctor, the prune scripts and the backups can see them — not in
    ``~/.cache/huggingface``.
    """
    s = get_settings()
    model = model or s.resolved_model

    s.models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(s.models_dir)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed — run: uv sync",
            file=sys.stderr,
        )
        return 1

    print(f"downloading {model}\n  into {s.models_dir}")
    loop = asyncio.get_running_loop()
    try:
        path = await loop.run_in_executor(
            None,
            lambda: snapshot_download(
                repo_id=model, cache_dir=str(s.models_dir / "hub")
            ),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced, not masked
        print(f"download failed: {exc}", file=sys.stderr)
        return 1

    # Measure the repo root, not the snapshot: HF puts the real bytes in
    # blobs/ and fills snapshots/ with symlinks to them.
    repo_root = Path(path).parent.parent
    size = sum(
        f.stat().st_size
        for f in repo_root.rglob("*")
        if f.is_file() and not f.is_symlink()
    ) / 1024**3
    print(f"  {size:.2f}GB at {path}")

    # Warm it once, so the first real request does not also pay the cold load.
    #
    # One worker, and the engine holds it: MLX streams are per-thread, so load
    # and transcribe have to happen on the same one. The same pool is handed to
    # whichever engine is built, so setup exercises the wiring serve will use.
    warm = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
    try:
        engine = _build_engine(_with_model(s, model), warm)
    except ImportError as exc:
        print(f"  weights downloaded; engine not installed ({exc}), skipping warm-up")
        return 0

    started = time.monotonic()
    try:
        await engine.load()
    except Exception as exc:  # noqa: BLE001
        print(f"  warm-up failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"  warm-up ok in {time.monotonic() - started:.1f}s "
        f"({engine.name} on {getattr(engine, 'device', 'metal')})"
    )
    await engine.unload()

    if s.punctuate:
        print(f"downloading {s.punctuation_model} (punctuation restoration)")
        try:
            from transcriptor.adapters.hf_punctuator import HuggingFacePunctuator

            punctuator = HuggingFacePunctuator(
                model=s.punctuation_model, executor=warm
            )
            started = time.monotonic()
            await punctuator.load()
            print(f"  ready in {time.monotonic() - started:.1f}s")
            await punctuator.unload()
        except Exception as exc:  # noqa: BLE001
            print(
                f"  punctuation model unavailable: {exc}\n"
                f"  transcripts will be correct but less readable — "
                f"run: uv sync --extra punctuate",
                file=sys.stderr,
            )

    warm.shutdown(wait=True)
    return 0


# --------------------------------------------------------------------------- ytdlp


def _update_ytdlp() -> int:
    """Bump yt-dlp in place.

    Sites break its extractors constantly — this is the one dependency that
    genuinely rots, and it wants bumping roughly monthly.
    """
    import subprocess

    print("updating yt-dlp in the project venv...")
    result = subprocess.run(
        ["uv", "pip", "install", "--upgrade", "yt-dlp"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        return 1

    version = subprocess.run(
        ["yt-dlp", "--version"], capture_output=True, text=True
    )
    print(f"yt-dlp is now {version.stdout.strip() or 'installed'}")
    return 0


# --------------------------------------------------------------------------- cli


def _not_yet(command: str, checkpoint: str) -> int:
    print(f"transcriptor-mcp {command}: not implemented yet ({checkpoint})", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="transcriptor-mcp",
        description="Local transcription MCP server. macOS, Linux and Windows.",
    )
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="run the MCP server (stdio by default)")
    serve.add_argument(
        "--http", action="store_true", help="serve over HTTP instead of stdio"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    sub.add_parser("doctor", help="environment preflight")
    sub.add_parser("migrate", help="apply pending migrations")
    setup = sub.add_parser("setup", help="download model weights")
    setup.add_argument("--model", default=None)
    sub.add_parser(
        "update-ytdlp",
        help="bump yt-dlp in place — sites break its extractors constantly",
    )
    pgimport = sub.add_parser(
        "import-postgres",
        help="one-off: copy transcripts out of the Postgres this used to use",
    )
    pgimport.add_argument(
        "--from",
        dest="from_dsn",
        default=None,
        metavar="DSN",
        help="postgresql://user:pass@host:port/dbname",
    )
    pgimport.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be copied and write nothing",
    )

    args = parser.parse_args(argv)

    # Before anything looks for ffmpeg or yt-dlp: a GUI-launched process has
    # neither Homebrew nor the venv on its PATH.
    ensure_tool_path()

    if args.command == "doctor":
        return asyncio.run(_doctor())
    if args.command == "migrate":
        return asyncio.run(_migrate())
    if args.command == "setup":
        return asyncio.run(_setup(args.model))
    if args.command == "update-ytdlp":
        return _update_ytdlp()
    if args.command == "import-postgres":
        return asyncio.run(_import_postgres(args.from_dsn, dry_run=args.dry_run))
    return _serve(
        http=getattr(args, "http", False),
        host=getattr(args, "host", "127.0.0.1"),
        port=getattr(args, "port", 8765),
    )


if __name__ == "__main__":
    raise SystemExit(main())
