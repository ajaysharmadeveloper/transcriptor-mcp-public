"""Invariant I9 and its neighbours: the dependency rule holds.

``make lint`` runs the same contracts, but keeping one in pytest means a plain
``pytest`` run also catches a violation.
"""

from __future__ import annotations

import subprocess
import sys


def test_import_contracts_hold() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "importlinter.cli", "lint"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
