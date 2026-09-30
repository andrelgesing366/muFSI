"""Three-dimensional hydrodynamic model skeleton."""

from dataclasses import dataclass
from typing import Any

from mufsi.hydrodynamics.base import HydrodynamicModel
from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.models.fluid import Fluid
from mufsi.solvers.linear import LinearSolver


@dataclass
class Stokes3D(HydrodynamicModel):
    """Store fluid setup for the analytical panel-integration formulation.

    TODO: support factorization reuse and bounded-memory solves. No explicit
    matrix inverse or mandatory dense matrix belongs in the public interface.
    """

    fluid: Fluid
    grid: FluidGrid
    solver: LinearSolver | None = None

    def pressure_from_velocity(self, omega: float, velocity: Any) -> Any:
        """Solve the 3D fluid problem at angular frequency omega."""
        raise NotImplementedError("The 3D pressure solve is pending.")
