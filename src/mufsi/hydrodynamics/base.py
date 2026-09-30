"""Operator-oriented interface for fluid models."""

from abc import ABC, abstractmethod
from typing import Any

from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.models.fluid import Fluid


class HydrodynamicModel(ABC):
    """Map transverse fluid-grid velocity to pressure.

    omega is angular frequency in rad/s. Grid point ordering must be shared by
    velocities, pressures, and integration weights. The implementation may use
    dense, blocked, iterative, or matrix-free linear algebra.
    """

    fluid: Fluid
    grid: FluidGrid

    @abstractmethod
    def pressure_from_velocity(self, omega: float, velocity: Any) -> Any:
        """Solve the fluid problem for velocity in m/s; return pressure in Pa."""
        raise NotImplementedError

    def assemble_matrix(self, omega: float) -> Any:
        """Optionally assemble the fluid system matrix for inspection.

        Coupled solvers must not require this method. Normalization and signs
        will be specified when the kernel is validated against the reference.
        """
        raise NotImplementedError("Explicit fluid matrix assembly is unavailable.")
