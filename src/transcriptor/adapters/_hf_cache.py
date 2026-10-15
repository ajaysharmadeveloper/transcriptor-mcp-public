"""Finding a model in the Hugging Face cache, without touching the network.

Both engine adapters need the same thing: the local snapshot directory for a
repo id, or a named error saying it was never downloaded.  Passing the repo id
straight to the library instead would let a run reach for the network in the
middle of a job — a surprise multi-gigabyte download at the worst moment, and a
silent breach of the promise that nothing leaves this machine.

It lives here rather than in one of the engines because the second copy of it
would be the bug: §19 records what happens when two identical implementations
drift, and this one is exactly the shape that drifts.
"""

from __future__ import annotations

from pathlib import Path

from transcriptor.domain.errors import ModelNotDownloaded


def repo_slug(repo_id: str) -> str:
    """``org/name`` → ``models--org--name``, the HF cache directory name."""
    return "models--" + repo_id.replace("/", "--")


def cache_root(models_dir: Path, repo_id: str) -> Path:
    """Where the weights for this repo would live if they were downloaded."""
    return models_dir / "hub" / repo_slug(repo_id)


def resolve_snapshot(
    models_dir: Path, repo_id: str, *, marker: str = "config.json"
) -> Path:
    """The local snapshot directory for ``repo_id``.

    ``marker`` is the file that proves a snapshot is complete rather than a
    half-finished download — every Whisper repo, MLX or CTranslate2, carries a
    ``config.json``.  Snapshots are tried newest first.
    """
    root = cache_root(models_dir, repo_id)
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        for candidate in sorted(snapshots.iterdir(), reverse=True):
            if (candidate / marker).exists():
                return candidate

    raise ModelNotDownloaded(
        f"{repo_id} is not downloaded. Run: transcriptor-mcp setup "
        f"--model {repo_id}",
        model=repo_id,
        expected=str(root),
    )
