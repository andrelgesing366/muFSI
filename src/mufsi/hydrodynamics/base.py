"""Operator-oriented interface for fluid models."""

from abc import ABC, abstractmethod
from typing import Any

from mufsi.models.fluid import Fluid


class HydrodynamicModel(ABC):
    """Map transverse collocation velocity to sampled pressure.

    omega is angular frequency in rad/s. Each model documents its collocation
    ordering. Panel models supply a FluidGrid; coefficient models supply their
    pressure basis and independent integrated force projection. They may use
    dense, blocked, iterative, or matrix-free linear algebra.
    """

    fluid: Fluid

    @abstractmethod
    def pressure_from_velocity(self, omega: float, velocity: Any) -> Any:
        """Solve the fluid problem for velocity in m/s; return pressure in Pa."""
        raise NotImplementedError

    def assemble_matrix(self, omega: float) -> Any:
        """Optionally assemble the fluid system matrix for inspection.

        Generic coupled solvers do not require this method. Stokes2D provides
        the sparse block mobility v = B p for an accelerated block solve;
        other implementations must document their own normalization.
        """
        raise NotImplementedError("Explicit fluid matrix assembly is unavailable.")
