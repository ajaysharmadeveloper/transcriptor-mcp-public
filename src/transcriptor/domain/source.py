"""SourceRef, ResultKey, and the normalisation rules.

Identity is the whole dedupe design, so it lives here as pure functions that
can be tested without a database, a network, or a filesystem.
"""

from __future__ import annotations

import enum
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from transcriptor.domain.errors import (
    PlaylistNotSupported,
    RelativePathNotAllowed,
    UnsupportedSource,
)


class SourceKind(str, enum.Enum):
    FILE = "file"
    URL = "url"


class CaptionPolicy(str, enum.Enum):
    """Human captions are usually better than Whisper; auto-generated ones are
    worse.  The two are not equivalent, so the choice is explicit."""

    MANUAL_ONLY = "manual_only"
    NEVER = "never"
    PREFER = "prefer"


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A source, normalised to its identity.

    ``key`` is what layer-1 dedupe matches on: the absolute resolved path for a
    file, or the normalised URL for a URL.
    """

    kind: SourceKind
    raw: str
    key: str
    platform: str

    @property
    def path(self) -> Path:
        if self.kind is not SourceKind.FILE:
            raise ValueError("not a file source")
        return Path(self.key)


def file_fingerprint(path: Path) -> str:
    """Size + mtime.

    For a local file the path alone is not identity — the file there may have
    been re-encoded or replaced.  A changed fingerprint means new content, so
    the source is re-transcribed rather than served from cache.
    """
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


# Query parameters that identify a *session or referral*, not a video.  Kept out
# of the source_key so a shared link and a clean one dedupe to the same job.
_TRACKING_PARAMS = frozenset(
    {
        "si", "feature", "app", "pp", "ab_channel", "gclid", "fbclid", "igshid",
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "ref", "ref_src", "ref_url", "s", "t", "start_radio", "rv", "context",
        "_nc_cat", "img_index", "share_id", "is_from_webapp", "sender_device",
        "web_id", "checksum", "sender_web_id",
    }
)

# Paths that mean "many videos", which is ambiguous rather than unsupported.
_PLAYLIST_MARKERS = ("/playlist", "/channel/", "/c/", "/user/", "/@")

# ...but these say "one specific item", and they win. A TikTok video lives at
# /@handle/video/<id>, so "/@" alone would reject every TikTok URL there is.
_SINGLE_ITEM_MARKERS = (
    "/video/", "/watch", "/reel/", "/reels/", "/p/", "/status/", "/shorts/",
    "/embed/", "/live/", "/tv/", "/clip/", "/posts/",
)

_PLATFORM_HOSTS = {
    "youtube": ("youtube.com", "youtu.be", "youtube-nocookie.com", "m.youtube.com"),
    "instagram": ("instagram.com", "instagr.am"),
    "tiktok": ("tiktok.com", "vm.tiktok.com"),
    "x": ("twitter.com", "x.com", "t.co"),
    "linkedin": ("linkedin.com", "lnkd.in"),
    "facebook": ("facebook.com", "fb.watch", "fb.com"),
    "vimeo": ("vimeo.com",),
    "reddit": ("reddit.com", "redd.it"),
}


def platform_of(host: str) -> str:
    host = host.lower().removeprefix("www.")
    for platform, hosts in _PLATFORM_HOSTS.items():
        if any(host == h or host.endswith("." + h) for h in hosts):
            return platform
    return "generic"


def normalise_url(raw: str) -> tuple[str, str]:
    """A URL reduced to its identity, plus the platform it belongs to.

    Tracking parameters are stripped and the video id kept, so the shared form
    of a link and the clean form produce one ``source_key`` — which is what
    makes layer-1 dedupe answer in milliseconds with no network at all.
    """
    parsed = urlsplit(raw.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    platform = platform_of(host)
    path = parsed.path.rstrip("/") or "/"

    query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k not in _TRACKING_PARAMS
    ]

    # youtu.be/<id> and youtube.com/watch?v=<id> are the same video.
    if platform == "youtube":
        video_id = dict(query).get("v")
        if not video_id and host == "youtu.be":
            video_id = path.lstrip("/").split("/")[0]
        if not video_id and path.startswith(("/shorts/", "/embed/", "/live/")):
            video_id = path.split("/")[2] if len(path.split("/")) > 2 else None
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}", platform

    if query:
        query.sort()
    normalised = urlunsplit(
        (parsed.scheme.lower() or "https", host, path, urlencode(query), "")
    )
    return normalised, platform


def _reject_unusable_url(raw: str, normalised: str, platform: str) -> None:
    """Named errors rather than a silent failure or a partial download.

    A playlist or channel URL is *ambiguous*, not unsupported — Transcriptor
    cannot tell which video was meant, and guessing would be worse than saying
    so.  Batch mode over a folder is a later phase.
    """
    lowered = normalised.lower()

    if any(marker in lowered for marker in _SINGLE_ITEM_MARKERS):
        # Explicitly one item, whatever else the path contains.
        if "list=" in lowered and "watch?v=" not in lowered:
            raise PlaylistNotSupported("that is a playlist URL", source=raw)
        return

    if any(marker in lowered for marker in _PLAYLIST_MARKERS) or "list=" in lowered:
        raise PlaylistNotSupported(
            "that looks like a playlist or channel rather than a single video — "
            "pass one video URL",
            source=raw,
        )


def parse_source(raw: str) -> SourceRef:
    """Work out what a source string is.  The caller never declares a type."""
    text = raw.strip()
    if not text:
        raise UnsupportedSource("source is empty")

    lowered = text.lower()
    if lowered.startswith(("http://", "https://")):
        key, platform = normalise_url(text)
        _reject_unusable_url(text, key, platform)
        return SourceRef(kind=SourceKind.URL, raw=raw, key=key, platform=platform)

    if "://" in text:
        raise UnsupportedSource(
            f"only http and https URLs are supported, got {text.split('://')[0]}://",
            source=text,
        )

    expanded = os.path.expanduser(text)

    # Reject before resolving: a stdio server launched by a GUI client has no
    # meaningful working directory, so resolving a relative path against it
    # would silently pick the wrong file.
    #
    # `Path` rather than `PurePosixPath`, and this is not a tidy-up. Under
    # PurePosixPath, `C:\Users\me\talk.mp4` has no leading slash and so reports
    # itself relative — every absolute Windows path was refused with
    # `relative_path`, which is to say local files did not work on Windows at
    # all. `Path` applies the running platform's own rules.
    if not Path(expanded).is_absolute():
        raise RelativePathNotAllowed(
            "give an absolute path — a stdio server has no meaningful working "
            "directory, so a relative path cannot be resolved reliably",
            source=text,
        )

    resolved = Path(expanded).resolve()
    return SourceRef(
        kind=SourceKind.FILE,
        raw=raw,
        key=_path_key(resolved),
        platform="local",
    )


def _path_key(resolved: Path) -> str:
    """The dedupe identity of a local path.

    ``os.path.normcase`` is the whole point, and it is a no-op on POSIX — so
    keys already stored on macOS and Linux are untouched. On Windows it folds
    case and normalises separators, without which ``C:\\Talk.mp4`` and
    ``c:/talk.mp4`` are two identities for one file: layer-1 dedupe would miss,
    the file would be transcribed twice, and nothing would report an error.
    Silent, and exactly the class of bug §19 collects.

    It deliberately does *not* try to resolve two paths that reach one file by
    different routes — a junction, a bind mount, a symlink farm. That is what
    ``file_fingerprint`` is for at the next layer.
    """
    return os.path.normcase(str(resolved))


def result_key(
    *,
    identity: str,
    model: str,
    language: str | None,
    captions: CaptionPolicy,
) -> str:
    """What makes two results equivalent.

    ``identity`` is the ``audio_hash`` once known, and the ``source_key`` before
    that.  Model, forced language and caption policy are folded in because a
    transcript produced under different settings is a *different artefact* —
    handing it back would be a correctness bug dressed as a cache hit.
    """
    parts = (identity, model, language or "auto", captions.value)
    digest = hashlib.blake2b(digest_size=32)
    for part in parts:
        # Length-prefixed, so ("ab", "c") and ("a", "bc") cannot collide.
        digest.update(str(len(part)).encode())
        digest.update(b":")
        digest.update(part.encode())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SourceMeta:
    """What a probe found out about a source, without transcribing it."""

    duration_s: float
    has_audio: bool
    container: str | None = None
    audio_codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    title: str | None = None
    uploader: str | None = None
    is_live: bool = False
    has_manual_captions: bool | None = None
    has_auto_captions: bool | None = None
