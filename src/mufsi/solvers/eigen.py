"""Structural eigenproblem interfaces.

The initial scope is the in-vacuo generalized problem K u = omega^2 M u.
Frequency-dependent fluid-loaded eigenproblems require a separate formulation.
"""

from dataclasses import dataclass
from typing import Any

from mufsi.structure.base import StructuralModel


@dataclass(frozen=True)
class EigenResult:
    """Angular eigenfrequencies in rad/s and corresponding structural modes."""

    angular_frequencies: Any
    modes: Any


@dataclass
class EigenSolver:
    """Store the structural model for a future PETSc/SLEPc eigensolve."""

    structure: StructuralModel

    def solve(self, n_modes: int) -> EigenResult:
        """Compute the requested in-vacuo structural eigenmodes."""
        raise NotImplementedError("The structural eigenproblem is pending.")
