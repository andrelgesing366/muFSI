"""Two-dimensional hydrodynamic model skeleton."""

from dataclasses import dataclass
from typing import Any

from mufsi.hydrodynamics.base import HydrodynamicModel
from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.models.fluid import Fluid
from mufsi.solvers.linear import LinearSolver


@dataclass
class Stokes2D(HydrodynamicModel):
    """Store fluid model setup; 2D hydrodynamic integration is pending."""

    fluid: Fluid
    grid: FluidGrid
    solver: LinearSolver | None = None

    def pressure_from_velocity(self, omega: float, velocity: Any) -> Any:
        """Solve the 2D fluid problem at angular frequency omega."""
        raise NotImplementedError("The 2D pressure solve is pending.")
