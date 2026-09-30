"""Result serialization with units and simulation metadata.

TODO: choose a versioned format that preserves complex-valued arrays, frequency
units, DOF/grid ordering, solver settings, and numerical environment metadata.
"""

from pathlib import Path

from mufsi.solvers.eigen import EigenResult
from mufsi.solvers.frequency_response import FrequencyResponseResult

Result = FrequencyResponseResult | EigenResult


def save_results(result: Result, path: str | Path) -> None:
    """Save result arrays and the metadata needed to interpret them."""
    raise NotImplementedError("Result serialization is pending.")


def load_results(path: str | Path) -> Result:
    """Load a supported result format and check its schema version."""
    raise NotImplementedError("Result loading is pending.")
