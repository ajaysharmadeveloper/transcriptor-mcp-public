"""Stopping a child process, on every operating system it might run on.

Cancellation is a guarantee this project makes (§6: download and extract cancel
immediately by killing the subprocess), and on Windows the obvious code does not
keep it.  ``Popen.kill`` there is ``TerminateProcess``, which stops the process
named and nothing it started.  yt-dlp starts ffmpeg to merge formats, so a
cancel would return promptly, report success, and leave a download running with
nobody left to stop it.

Shared by both subprocess adapters rather than written twice, because the two
copies of this that already existed were identical — and identical code that
has to change on one platform is how one copy quietly gets fixed and the other
does not.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import sys

log = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"


def kill_tree(proc) -> None:
    """Stop ``proc`` and everything it started.  Safe to call twice.

    A process that has already exited is the normal case, not an error: the
    canceller and the process finishing race by nature, and the loser must not
    raise.
    """
    if proc.returncode is not None:
        return

    if not is_windows():
        # POSIX: the child is in our process group and dies with the signal.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return

    # Windows has no process groups to signal, so the tree is walked by pid.
    # `taskkill /T` is the documented way to do that without pulling in psutil
    # here — this module stays dependency-free so both adapters can use it.
    with contextlib.suppress(Exception):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            check=False,
            capture_output=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    # taskkill can miss — the pid may have been reused, or the process may be
    # exiting already. Falling back to kill() at least stops the parent, which
    # is what the old code did on every platform.
    if proc.returncode is None:
        with contextlib.suppress(Exception):
            proc.kill()


def killer(proc):
    """A zero-argument callback that stops ``proc`` and its children.

    The pipeline holds one of these for the running child so a cancel can reach
    it without knowing what kind of process it is.
    """

    def kill() -> None:
        kill_tree(proc)

    return kill


def subprocess_flags() -> dict:
    """Keyword arguments that keep a child process from stealing the terminal.

    On Windows every ``create_subprocess_exec`` would otherwise flash a console
    window — harmless when run from a terminal, but this server is normally
    started by a GUI client, where a window appearing for every chunk of every
    job is a visible defect.  Empty on POSIX.
    """
    if not is_windows():
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


def executable_name(stem: str) -> str:
    """``ffmpeg`` on POSIX, ``ffmpeg.exe`` on Windows.

    ``shutil.which`` already applies PATHEXT, so this is only for the places
    that report a name to a human or build one by hand.
    """
    if is_windows() and not stem.lower().endswith(".exe"):
        return stem + ".exe"
    return stem


def is_windows() -> bool:
    """One place answers this, so a test can monkeypatch one thing."""
    return _IS_WINDOWS


__all__ = [
    "executable_name",
    "is_windows",
    "kill_tree",
    "killer",
    "subprocess_flags",
]
