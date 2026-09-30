"""Evaluate structural fields on the fluid grid and project fluid forces."""

from dataclasses import dataclass
from typing import Any

from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.structure.base import StructuralModel


@dataclass
class CouplingOperator:
    """Store sparse E and fluid weights for E @ u and E.T @ (weights * p).

    After construction this operator will use numerical arrays only. Signed
    pressure and structural load conventions must be verified together.
    """

    evaluation_matrix: Any
    weights: Any

    @classmethod
    def from_structure(
        cls, structure: StructuralModel, grid: FluidGrid
    ) -> "CouplingOperator":
        """Construct E by evaluating the structural basis at grid points."""
        raise NotImplementedError("Coupling construction from DOLFINx is pending.")

    def to_fluid(self, structural_values: Any) -> Any:
        """Evaluate structural displacement or velocity as E @ values."""
        raise NotImplementedError("Structural-to-fluid evaluation is pending.")

    def to_structure(self, pressure: Any) -> Any:
        """Project pressure to force as E.T @ (weights * pressure)."""
        raise NotImplementedError("Fluid-to-structural force projection is pending.")
