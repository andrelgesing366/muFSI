"""Simulation configuration input.

TODO: choose a schema and validate SI units, model identifiers, and solver
options. Configuration loading must not trigger numerical assembly.
"""

from pathlib import Path
from typing import Any


def read_config(path: str | Path) -> dict[str, Any]:
    """Read a simulation configuration according to the future schema."""
    raise NotImplementedError("Configuration parsing and validation are pending.")
