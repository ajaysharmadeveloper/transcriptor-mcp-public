"""Config surface, and the one rule that must never regress: the DSN is masked."""

from __future__ import annotations

from transcriptor.config import Settings, masked_dsn


def test_masked_dsn_hides_the_password() -> None:
    dsn = "postgresql://postgres:hunter2@localhost:5432/transcriptor_db"
    masked = masked_dsn(dsn)
    assert "hunter2" not in masked
    assert masked == "postgresql://postgres:********@localhost:5432/transcriptor_db"


def test_masked_dsn_leaves_a_passwordless_dsn_alone() -> None:
    dsn = "postgresql://localhost:5432/transcriptor_db"
    assert masked_dsn(dsn) == dsn


def test_data_dir_defaults_inside_the_repo(settings: Settings) -> None:
    # Model weights and scratch audio live in the project, not in $HOME.
    assert settings.data_dir.is_absolute()
    assert settings.models_dir == settings.data_dir / "models"
    assert settings.scratch_dir == settings.data_dir / "tmp"


def test_the_duration_cap_is_four_hours(settings: Settings) -> None:
    """A guard against an accidental 12-hour stream VOD, not a capability limit —
    chunking removed the reason for the old ten-minute cap."""
    assert settings.max_duration_s == 14400
