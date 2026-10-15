"""What has to hold on Windows and Linux, not only on the machine this was
written on.

Several of these are marked to run only on the OS they describe. That is not a
way of skipping them: CI runs the suite on ubuntu, macos and windows, so each
one runs somewhere on every push. A test that quietly passes everywhere by
asserting nothing would be worse than no test at all (§19).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from transcriptor.domain.errors import RelativePathNotAllowed
from transcriptor.domain.source import parse_source

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows behaviour")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX behaviour")


# ------------------------------------------------------------ absolute paths


@posix_only
def test_a_posix_absolute_path_is_accepted() -> None:
    assert parse_source("/tmp/talk.mp4").key.endswith("talk.mp4")


@windows_only
def test_a_windows_absolute_path_is_accepted() -> None:
    """The bug this replaced: the check used PurePosixPath, under which
    ``C:\\Users\\me\\talk.mp4`` has no leading slash and so reported itself
    relative. Every absolute Windows path was refused with `relative_path`,
    which is to say local files did not work on Windows at all."""
    ref = parse_source(r"C:\Users\me\talk.mp4")
    assert ref.key.endswith("talk.mp4")


@windows_only
def test_a_windows_unc_path_is_accepted() -> None:
    ref = parse_source(r"\\server\share\talk.mp4")
    assert ref.key.endswith("talk.mp4")


def test_a_relative_path_is_still_refused() -> None:
    """A stdio server launched by a GUI client has no meaningful working
    directory, so resolving a relative path would silently pick the wrong
    file. That reasoning is platform-independent and must survive the fix."""
    with pytest.raises(RelativePathNotAllowed):
        parse_source("talk.mp4")


def test_a_bare_filename_with_a_drive_letter_shape_is_not_special_cased() -> None:
    """Guarding against an over-eager fix: "notes:2026" is not a drive letter,
    and on POSIX it is simply a relative path."""
    if sys.platform == "win32":
        pytest.skip("colon is not legal in a Windows filename")
    with pytest.raises(RelativePathNotAllowed):
        parse_source("notes:2026.mp4")


# --------------------------------------------------------------- dedupe keys


def test_the_path_key_is_normalised_for_the_platform() -> None:
    from transcriptor.domain.source import _path_key

    key = _path_key(Path(os.path.join(os.sep, "Some", "Path", "File.MP4")))
    assert key == os.path.normcase(key), "the key must already be normalised"


@windows_only
def test_case_differing_windows_paths_dedupe_to_one_key() -> None:
    """Windows filesystems are case-insensitive, so two spellings of one path
    are one file. Without normcase they are two source_keys, layer-1 dedupe
    misses, the file is transcribed twice, and nothing reports an error."""
    assert parse_source(r"C:\Media\Talk.mp4").key == parse_source(r"c:\media\talk.mp4").key


@posix_only
def test_case_differing_posix_paths_stay_distinct() -> None:
    """The mirror image, and the reason normcase rather than lower(): on Linux
    these are genuinely two different files."""
    assert parse_source("/media/Talk.mp4").key != parse_source("/media/talk.mp4").key


# ----------------------------------------------------------- killing children


def test_kill_tree_tolerates_a_process_that_already_exited() -> None:
    """The canceller and the process finishing race by nature. The loser must
    not raise."""
    from transcriptor.adapters._proc import kill_tree

    class Finished:
        returncode = 0
        pid = 1

        def kill(self):  # pragma: no cover - must never be reached
            raise AssertionError("a finished process must not be killed again")

    kill_tree(Finished())


@posix_only
def test_kill_tree_signals_the_process_on_posix() -> None:
    from transcriptor.adapters._proc import kill_tree

    killed = []

    class Running:
        returncode = None
        pid = 4321

        def kill(self):
            killed.append(True)

    kill_tree(Running())
    assert killed == [True]


def test_kill_tree_uses_taskkill_on_windows(monkeypatch) -> None:
    """Asserted by causing it rather than by checking a flag afterwards.

    ``proc.kill()`` on Windows is TerminateProcess, which stops the named
    process and nothing it started — and yt-dlp starts ffmpeg to merge
    formats. A cancel would return promptly, report success, and leave a
    download running with nobody left to stop it.
    """
    import transcriptor.adapters._proc as proc_mod

    commands: list[list[str]] = []
    monkeypatch.setattr(proc_mod, "is_windows", lambda: True)
    monkeypatch.setattr(
        proc_mod.subprocess,
        "run",
        lambda cmd, **kw: commands.append(cmd),
    )

    class Running:
        returncode = None
        pid = 4321

        def kill(self):
            self.returncode = -9

    proc_mod.kill_tree(Running())

    assert commands, "nothing was killed"
    assert commands[0][:4] == ["taskkill", "/F", "/T", "/PID"]
    assert "/T" in commands[0], "without /T the children survive the cancel"


def test_both_subprocess_adapters_use_the_shared_killer() -> None:
    """The two copies of this that existed were identical, which is how one
    gets fixed for Windows and the other does not (§19)."""
    from transcriptor.adapters import _proc, ffmpeg_audio, ytdlp_gateway

    assert ffmpeg_audio._killer is _proc.killer
    assert ytdlp_gateway._killer is _proc.killer


# ------------------------------------------------------------------- PATH


def test_the_venv_script_directory_follows_the_install() -> None:
    """`bin` on POSIX and `Scripts` on Windows — derived from sys.executable
    either way, so it follows the install rather than assuming a layout."""
    from transcriptor.server import _tool_path_candidates

    assert _tool_path_candidates()[0] == str(Path(sys.executable).parent)


def test_no_homebrew_paths_are_suggested_off_macos() -> None:
    from transcriptor.server import _EXTRA_PATHS

    for platform_key, paths in _EXTRA_PATHS.items():
        if platform_key != "darwin":
            assert not any("homebrew" in p for p in paths)


def test_every_supported_platform_has_an_ffmpeg_hint() -> None:
    """Naming the wrong package manager is worse than naming none: it sends a
    Debian user to install Homebrew."""
    from transcriptor.server import _FFMPEG_HINT

    assert set(_FFMPEG_HINT) == {"darwin", "linux", "win32"}
    assert "brew" in _FFMPEG_HINT["darwin"]
    assert "apt" in _FFMPEG_HINT["linux"]
    assert "winget" in _FFMPEG_HINT["win32"]


# --------------------------------------------------------- engine selection


def test_apple_silicon_prefers_mlx(monkeypatch) -> None:
    import transcriptor.config as config

    monkeypatch.setattr(config.sys, "platform", "darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(config.importlib.util, "find_spec", lambda name: object())
    assert config.default_engine() == "mlx"


@pytest.mark.parametrize(
    ("platform_name", "machine"),
    [("linux", "x86_64"), ("win32", "AMD64"), ("darwin", "x86_64")],
)
def test_everything_else_gets_faster_whisper(monkeypatch, platform_name, machine) -> None:
    import transcriptor.config as config

    monkeypatch.setattr(config.sys, "platform", platform_name)
    monkeypatch.setattr(config.platform, "machine", lambda: machine)
    assert config.default_engine() == "faster-whisper"


def test_apple_silicon_without_mlx_installed_falls_back(monkeypatch) -> None:
    """A Mac with the mlx extra left uninstalled should run something that
    works rather than fail at the first job."""
    import transcriptor.config as config

    monkeypatch.setattr(config.sys, "platform", "darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(config.importlib.util, "find_spec", lambda name: None)
    assert config.default_engine() == "faster-whisper"


def test_each_engine_has_its_own_default_model() -> None:
    """Neither id is loadable by the other, so one hardcoded default would be
    wrong on one platform or the other."""
    from transcriptor.config import DEFAULT_MODELS

    assert DEFAULT_MODELS["mlx"] != DEFAULT_MODELS["faster-whisper"]
    assert "mlx" in DEFAULT_MODELS["mlx"]
    assert "faster-whisper" in DEFAULT_MODELS["faster-whisper"]


def test_the_fake_engine_env_var_still_wins(monkeypatch) -> None:
    """It predates the setting, and every debugging note in the README uses
    it."""
    from transcriptor.config import Settings

    monkeypatch.setenv("TRANSCRIPTOR_FAKE_ENGINE", "1")
    assert Settings(_env_file=None, engine="mlx").resolved_engine == "fake"


def test_an_unknown_engine_name_fails_loudly(tmp_path, monkeypatch) -> None:
    """Rather than silently falling back to something the user did not ask
    for — a wrong engine is a wrong result_key and a wrong transcript."""
    from transcriptor.config import Settings
    from transcriptor.server import _build_engine

    monkeypatch.delenv("TRANSCRIPTOR_FAKE_ENGINE", raising=False)
    settings = Settings(_env_file=None, data_dir=tmp_path, engine="whisper.cpp")
    with pytest.raises(ValueError, match="unknown TRANSCRIPTOR_ENGINE"):
        _build_engine(settings, None)
