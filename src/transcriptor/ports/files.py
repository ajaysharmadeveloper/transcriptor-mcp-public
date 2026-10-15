"""FileWriter — the only sanctioned way out to the filesystem.

``transcript_export`` is the single place Transcriptor writes output to disk,
and only because someone asked for it.  Behind a port so the write stays out of
the services layer and a test never touches a real path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class FileWriter(Protocol):
    async def write_text(self, path: Path, content: str) -> int:
        """Write ``content``, returning bytes written.

        Must be atomic from the caller's point of view: on failure nothing
        partial is left behind.
        """
        ...

    async def remove_tree(self, path: Path) -> int:
        """Delete a directory, returning the bytes freed."""
        ...

    async def list_media(self, folder: Path, pattern: str = "*") -> list[Path]:
        """Media files directly inside ``folder``, sorted.

        Behind the port because a tool handler must not touch the filesystem,
        and because a test should not need a real directory tree.
        """
        ...
