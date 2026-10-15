"""Settings → typed config object.

Reads ``TRANSCRIPTOR_*`` from the environment, layered over two optional env
files.  Deliberately imports nothing else from the project: `config` sits
outside the layering and every layer above `domain` may read it.

``masked_dsn`` survives for the rare case of a URL that carries credentials.
The default database is a file inside the project, so ordinarily there is no
password here to leak.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/transcriptor/config.py → repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Later files win.  The project-local .env is what this machine actually uses;
# ~/.transcriptor/env stays supported for an installed (uv tool) deployment.
ENV_FILES = (Path.home() / ".transcriptor" / "env", REPO_ROOT / ".env")

_DSN_PASSWORD = re.compile(r"(?P<head>://[^:/@]+:)[^@]*(?P<tail>@)")

# Each engine takes its own weights, and neither id is loadable by the other:
# MLX reads an mlx-community conversion, CTranslate2 reads a CT2 conversion.
# Both produce the same transcript from the same large-v3 model — and because
# `engine.name` goes into `result_key` (§7), a transcript made by one is never
# handed back as though the other had produced it.
DEFAULT_MODELS = {
    "mlx": "mlx-community/whisper-large-v3-mlx-8bit",
    "faster-whisper": "Systran/faster-whisper-large-v3",
    "fake": "fake-engine",
}


def default_engine() -> str:
    """The engine for this machine when none was named.

    MLX is Metal-only, so Apple Silicon gets it and everything else — Linux,
    Windows, Intel Macs — gets the CTranslate2 build.  The check is for an
    importable ``mlx_whisper`` rather than for the architecture alone, so a
    Mac with the ``mlx`` extra left uninstalled falls back to something that
    works instead of failing at the first job.
    """
    if sys.platform == "darwin" and platform.machine() == "arm64":
        if importlib.util.find_spec("mlx_whisper") is not None:
            return "mlx"
    return "faster-whisper"


def masked_dsn(dsn: str) -> str:
    """The DSN with its password replaced, safe to print or log."""
    return _DSN_PASSWORD.sub(r"\g<head>********\g<tail>", dsn)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIPTOR_",
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- engine ------------------------------------------------------------
    # "auto" picks MLX on Apple Silicon and faster-whisper everywhere else.
    # Naming one explicitly is for testing a specific backend, or for forcing
    # the portable one on a Mac to reproduce what a Linux user sees.
    engine: str = "auto"
    # Empty means "whatever the chosen engine's default model is" — the two
    # engines take different weights and neither id is valid for the other, so
    # a single hardcoded default would be wrong on one platform or the other.
    model: str = ""
    # faster-whisper only. "auto" uses CUDA when CTranslate2 reports a device.
    device: str = "auto"
    # Empty means float16 on CUDA, int8 on CPU.
    compute_type: str | None = None

    # -- storage -----------------------------------------------------------
    # Model weights and scratch audio only.  Nothing durable lives here.
    data_dir: Path = REPO_ROOT / ".transcriptor"

    # -- database ----------------------------------------------------------
    # Empty means the project's own database file, <data_dir>/transcriptor.db.
    # That default is the whole point: there is no server to install, no
    # password to protect and no DSN to paste before the first transcript.
    # Set it only to put the file somewhere else.
    database_url: str = ""
    segment_buffer_max: int = 500

    # -- memory governance -------------------------------------------------
    # Maximum occupancy for Transcriptor itself. Measured peak with large-v3
    # 8-bit is 2.2GB, so this has headroom; fp16 at ~6.5GB is refused by it.
    mem_self_cap_gb: float = 4.0
    # Free memory to leave for everything else. Lowered from 3.0 on 2026-09-09
    # after measurement: see domain/pressure.Thresholds.
    mem_system_floor_gb: float = 2.5
    mem_poll_s: int = 5
    mem_pause_max_s: int = 60
    idle_evict_s: int = 600
    max_concurrent: int = 1

    # -- pipeline ----------------------------------------------------------
    chunk_s: int = 120
    overlap_s: int = 15
    keep_scratch: bool = False
    # Restore sentence boundaries after transcription. On by default: it is
    # what makes the fast decode path readable.
    punctuate: bool = True
    punctuation_model: str = "kredor/punctuate-all"
    # Abandoned scratch older than this is pruned. Active jobs are never
    # touched, whatever their age.
    scratch_ttl_days: float = 30.0
    prune_interval_s: float = 3600.0

    # -- transcript viewer -------------------------------------------------
    # A loopback HTTP server that hands a finished transcript to a browser, so
    # a completed job can report a download link instead of pushing thousands
    # of words through the chat window. Loopback only; see adapters/http_viewer.
    viewer_enabled: bool = True
    viewer_host: str = "127.0.0.1"
    viewer_port: int = 8766

    # -- job lifecycle -----------------------------------------------------
    heartbeat_s: int = 10
    stale_job_s: int = 90
    reaper_s: int = 60

    # -- input handling ----------------------------------------------------
    captions_default: str = "manual_only"
    max_duration_s: int = 14400
    max_download_mb: int = 2048

    # yt-dlp's challenge-solver components, e.g. "ejs:github". Off by default:
    # it downloads and runs a script from GitHub, which is remote code in a
    # project whose premise is that nothing leaves the machine.
    ytdlp_remote_components: str | None = None

    cookies_from_browser: str | None = None
    cookies_file: Path | None = None

    # -- derived paths -----------------------------------------------------
    @property
    def models_dir(self) -> Path:
        """HF_HOME points here.  Model weights only."""
        return self.data_dir / "models"

    @property
    def scratch_dir(self) -> Path:
        """In-flight audio.  Wiped on job completion and again at startup."""
        return self.data_dir / "tmp"

    @field_validator("data_dir", mode="after")
    @classmethod
    def _resolve_data_dir(cls, v: Path) -> Path:
        return v if v.is_absolute() else (REPO_ROOT / v).resolve()

    # -- engine resolution -------------------------------------------------
    @property
    def resolved_engine(self) -> str:
        """Which engine will actually be built.

        ``TRANSCRIPTOR_FAKE_ENGINE=1`` still wins, because it predates this
        setting and every debugging note in the README uses it.
        """
        if os.environ.get("TRANSCRIPTOR_FAKE_ENGINE") == "1":
            return "fake"
        if self.engine != "auto":
            return self.engine
        return default_engine()

    @property
    def resolved_model(self) -> str:
        """The model id, filled in from the engine where none was given.

        Read this rather than ``model`` anywhere the answer has to be a real
        repo id.  ``model`` stays as the user set it — empty means "you
        choose" — so that changing engines does not silently keep weights the
        new engine cannot read.
        """
        if self.model:
            return self.model
        return DEFAULT_MODELS.get(self.resolved_engine, DEFAULT_MODELS["mlx"])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
