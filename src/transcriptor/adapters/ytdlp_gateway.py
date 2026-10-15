"""YtDlpMediaGateway — URL ingestion.

Driven as a child process with ``create_subprocess_exec`` (rule R3), never
in-process: yt-dlp's download loop is blocking, and running it on the event loop
would freeze every tool call for the length of a download.

There is no per-platform code here.  Coverage is whatever yt-dlp supports, which
is over a thousand sites — which is also why this is the one dependency that
genuinely rots, and why ``update-ytdlp`` exists.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

from transcriptor.adapters._proc import killer as _killer, subprocess_flags
from transcriptor.domain.errors import (
    AuthError,
    Cancelled,
    DrmProtected,
    ExtractorFailed,
    LiveStreamNotSupported,
    PlaylistNotSupported,
    SourceError,
    SourceNotFound,
)
from transcriptor.domain.source import CaptionPolicy, SourceMeta, SourceRef
from transcriptor.domain.transcript import Segment

log = logging.getLogger(__name__)

# yt-dlp says a lot; these are the phrases worth turning into a named error
# rather than surfacing as a traceback.
_AUTH_MARKERS = (
    "sign in to confirm",
    "private video",
    "members-only",
    "login required",
    "requested format is not available",
    "this video is available to this channel's members",
    "age-restricted",
    "confirm your age",
    "use --cookies",
    "login to view",
)
_DRM_MARKERS = ("drm", "protected by digital rights")
# YouTube's JavaScript anti-bot challenge. Distinct from an auth failure, and
# the fix is completely different, so it gets its own message.
_CHALLENGE_MARKERS = (
    "the page needs to be reloaded",
    "n challenge solving failed",
    "challenge solver",
    "player response",
)
_NOT_FOUND_MARKERS = ("video unavailable", "does not exist", "410: gone", "404")


class YtDlpMediaGateway:
    """Implements :class:`transcriptor.ports.media.MediaGateway`."""

    def __init__(
        self,
        *,
        binary: str = "yt-dlp",
        cookies_from_browser: str | None = None,
        cookies_file: Path | None = None,
        max_download_mb: int = 2048,
        remote_components: str | None = None,
    ) -> None:
        self._binary = binary
        # Off by default. Enabling it makes yt-dlp download a challenge-solver
        # script from GitHub and run it — which is remote code, in a project
        # whose premise is that nothing leaves the machine. It is the only way
        # past YouTube's JS challenge on some videos, so it is offered, but the
        # user opts in knowing the trade.
        self._remote_components = remote_components
        self._cookies_from_browser = cookies_from_browser
        self._cookies_file = cookies_file
        self._max_download_mb = max_download_mb
        self._js_runtime = _find_js_runtime()
        if self._js_runtime is None:
            log.warning(
                "no JavaScript runtime found — YouTube extraction is deprecated "
                "without one and some formats will be missing. "
                "Install one: brew install deno",
            )

    @property
    def available(self) -> bool:
        return shutil.which(self._binary) is not None

    async def version(self) -> str:
        out, _ = await self._run("--version")
        return out.strip()

    # ---------------------------------------------------------------- probe

    async def probe(self, ref: SourceRef) -> SourceMeta:
        """Metadata only — no download.

        Lets a caller decide whether a three-hour podcast is worth twenty-five
        minutes of inference, and surfaces up front whether captions would make
        it instant.
        """
        out, _ = await self._run(
            "--dump-single-json",
            "--no-playlist",
            "--skip-download",
            *self._runtime_flags(),
            *self._auth_flags(),
            ref.key,
        )
        try:
            data = json.loads(out)
        except json.JSONDecodeError as exc:
            raise ExtractorFailed(
                f"yt-dlp returned something that is not JSON for {ref.key}",
                source=ref.key,
            ) from exc

        if data.get("_type") == "playlist" or "entries" in data:
            raise PlaylistNotSupported(
                "that URL resolves to a playlist, not a single video",
                source=ref.key,
            )
        if data.get("is_live") or data.get("live_status") == "is_live":
            raise LiveStreamNotSupported(
                "a live stream has no fixed endpoint to hash or chunk",
                source=ref.key,
            )

        subtitles = data.get("subtitles") or {}
        automatic = data.get("automatic_captions") or {}
        duration = data.get("duration")

        return SourceMeta(
            duration_s=float(duration) if duration else 0.0,
            has_audio=True,
            container=data.get("ext"),
            title=data.get("title"),
            uploader=data.get("uploader") or data.get("channel"),
            is_live=bool(data.get("is_live")),
            has_manual_captions=bool(subtitles),
            has_auto_captions=bool(automatic),
        )

    # ---------------------------------------------------------------- fetch

    async def fetch_audio(self, ref: SourceRef, dest: Path, *, cancel=None) -> Path:
        """``bestaudio`` only.

        A 40-minute 4K video is roughly 30MB of audio instead of 2GB of video —
        faster to fetch, and it keeps the scratch directory small.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        template = str(dest.with_suffix(".%(ext)s"))

        await self._run(
            "--no-playlist",
            "--no-progress",
            "--no-part",
            "-f",
            "bestaudio/best",
            "--max-filesize",
            f"{self._max_download_mb}M",
            "-o",
            template,
            *self._runtime_flags(),
            *self._auth_flags(),
            ref.key,
            # Without this the token was accepted and ignored, so cancelling a
            # job left yt-dlp downloading to completion.
            cancel=cancel,
        )

        produced = sorted(dest.parent.glob(dest.stem + ".*"))
        if not produced:
            raise ExtractorFailed(
                "yt-dlp reported success but produced no file — the format may "
                "have been rejected by --max-filesize",
                source=ref.key,
                max_mb=self._max_download_mb,
            )
        return produced[0]

    async def fetch_captions(
        self, ref: SourceRef, policy: CaptionPolicy
    ) -> tuple[list[Segment], str] | None:
        """Human captions when the policy allows, else nothing.

        Human-uploaded captions are usually *better* than Whisper — correct
        punctuation, correct names, correct jargon.  Auto-generated ones are
        *worse* — no punctuation, no casing, weak on accents.  They are not
        equivalent, so the policy decides and the result reports which it got.
        """
        if policy is CaptionPolicy.NEVER:
            return None

        meta = await self.probe(ref)
        wants_auto = policy is CaptionPolicy.PREFER
        if not meta.has_manual_captions and not (wants_auto and meta.has_auto_captions):
            return None

        import tempfile

        from transcriptor.adapters.renderers import parse_vtt

        with tempfile.TemporaryDirectory() as tmp:
            template = str(Path(tmp) / "caps.%(ext)s")
            flags = ["--write-subs"] if meta.has_manual_captions else []
            if wants_auto and not meta.has_manual_captions:
                flags = ["--write-auto-subs"]
            await self._run(
                "--no-playlist",
                "--skip-download",
                *flags,
                "--sub-format",
                "vtt",
                "--sub-langs",
                "all",
                "-o",
                template,
                *self._runtime_flags(),
                *self._auth_flags(),
                ref.key,
            )
            files = sorted(Path(tmp).glob("*.vtt"))
            if not files:
                return None
            segments = parse_vtt(files[0].read_text(encoding="utf-8", errors="replace"))

        if not segments:
            return None
        origin = (
            "captions_manual" if meta.has_manual_captions else "captions_auto"
        )
        log.info(
            "using captions instead of inference",
            extra={"source": ref.key, "origin": origin, "segments": len(segments)},
        )
        return segments, origin

    # ------------------------------------------------------------- internal

    def _runtime_flags(self) -> list[str]:
        """Point yt-dlp at whatever JS runtime this machine has.

        YouTube serves a JavaScript challenge, and yt-dlp only enables `deno` by
        default. Extraction without a runtime is deprecated and silently drops
        formats, so an installed node or bun is used rather than wasted.
        """
        flags: list[str] = []
        if self._js_runtime is not None:
            name, path = self._js_runtime
            flags += ["--js-runtimes", f"{name}:{path}"]
        if self._remote_components:
            flags += ["--remote-components", self._remote_components]
        return flags

    def _auth_flags(self) -> list[str]:
        """Cookies are read on the host and never logged, never transmitted
        anywhere except to the site being fetched."""
        if self._cookies_from_browser:
            return ["--cookies-from-browser", self._cookies_from_browser]
        if self._cookies_file:
            return ["--cookies", str(self._cookies_file)]
        return []

    async def _run(self, *args: str, cancel=None) -> tuple[str, str]:
        if not self.available:
            raise ExtractorFailed(
                f"{self._binary} is not on PATH — run: uv sync --extra media",
                binary=self._binary,
            )
        proc = await asyncio.create_subprocess_exec(
            self._binary,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Keeps a console window from flashing up per child on Windows.
            **subprocess_flags(),
        )
        killer = _killer(proc)
        if cancel is not None:
            cancel.on_cancel(killer)
        try:
            stdout, stderr = await proc.communicate()
        finally:
            if cancel is not None:
                cancel.forget(killer)

        err = stderr.decode(errors="replace")
        if cancel is not None and cancel.cancelled:
            raise Cancelled(cancel.reason or "cancelled", binary=self._binary)
        if proc.returncode != 0:
            raise self._classify(err)
        return stdout.decode(errors="replace"), err

    @staticmethod
    def _classify(stderr: str) -> SourceError:
        """One mapping point, so no caller ever parses yt-dlp's output."""
        lowered = stderr.lower()
        if any(m in lowered for m in _AUTH_MARKERS):
            return AuthError(
                "this source needs your session cookies — set "
                "TRANSCRIPTOR_COOKIES_FROM_BROWSER (chrome, safari, firefox, "
                "brave) or TRANSCRIPTOR_COOKIES_FILE",
                stderr=stderr.strip()[-500:],
            )
        if any(m in lowered for m in _CHALLENGE_MARKERS):
            return ExtractorFailed(
                "YouTube's JavaScript challenge could not be solved. This is not "
                "an authentication problem — cookies will not help. Either "
                "install Deno (brew install deno), or set "
                "TRANSCRIPTOR_YTDLP_REMOTE_COMPONENTS=ejs:github in .env, which "
                "lets yt-dlp download and run a solver script from GitHub. "
                "Downloading the audio yourself and passing the local path "
                "always works.",
                stderr=stderr.strip()[-500:],
            )
        if any(m in lowered for m in _DRM_MARKERS):
            return DrmProtected("this source is DRM-protected and not extractable")
        if any(m in lowered for m in _NOT_FOUND_MARKERS):
            return SourceNotFound(
                "yt-dlp could not find that video", stderr=stderr.strip()[-500:]
            )
        if "is not a valid url" in lowered or "unsupported url" in lowered:
            return ExtractorFailed(
                "yt-dlp has no extractor for that URL",
                stderr=stderr.strip()[-500:],
            )
        return ExtractorFailed(
            "yt-dlp failed — it may need updating for this site: "
            "transcriptor-mcp update-ytdlp",
            stderr=stderr.strip()[-500:],
        )



# yt-dlp enables only `deno` by default; node and bun work if pointed at.
_JS_RUNTIMES = ("deno", "node", "bun")


def _find_js_runtime() -> tuple[str, str] | None:
    for name in _JS_RUNTIMES:
        found = shutil.which(name)
        if found:
            return name, found
    return None
