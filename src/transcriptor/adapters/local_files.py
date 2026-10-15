"""LocalFiles — the filesystem, off the event loop."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from concurrent.futures import Executor
from pathlib import Path

from transcriptor.domain.errors import SourceNotFound, TranscriptorError


class ExportPathUnwritable(TranscriptorError):
    code = "export_path_unwritable"
    resumable = False


class LocalFiles:
    """Implements :class:`transcriptor.ports.files.FileWriter`."""

    def __init__(self, *, cpu_executor: Executor | None = None) -> None:
        self._cpu = cpu_executor

    async def write_text(self, path: Path, content: str) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._cpu, _write_atomically, path, content)

    async def remove_tree(self, path: Path) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._cpu, _remove_tree, path)

    async def list_media(self, folder: Path, pattern: str = "*") -> list[Path]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._cpu, _list_media, folder, pattern)


def _write_atomically(path: Path, content: str) -> int:
    """Write via a temporary file in the same directory, then rename.

    So a failure part-way leaves the destination untouched rather than
    half-written — "nothing partial is left behind" is part of the contract.
    """
    data = content.encode()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp = tempfile.mkstemp(
            dir=str(path.parent), prefix=path.name + ".", suffix=".part"
        )
        try:
            with os.fdopen(handle, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise ExportPathUnwritable(
            f"cannot write to {path}: {exc.strerror or exc}", path=str(path)
        ) from exc
    return len(data)


def _remove_tree(path: Path) -> int:
    if not path.exists():
        return 0
    freed = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    shutil.rmtree(path, ignore_errors=True)
    return freed


# What ffmpeg can open, per project-details §7. Deliberately not "every file":
# pointing batch mode at a folder should not try to transcribe its README.
MEDIA_SUFFIXES = frozenset(
    {
        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv",
        ".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".aiff", ".wma",
    }
)


def _list_media(folder: Path, pattern: str) -> list[Path]:
    if not folder.is_dir():
        raise SourceNotFound(f"not a directory: {folder}", path=str(folder))
    return sorted(
        p
        for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES
    )
