"""Fluid collocation points, panel boundaries, and integration weights."""

from dataclasses import dataclass
from typing import Any

from mufsi.models.geometry import PlateGeometry


@dataclass(frozen=True)
class FluidGrid:
    """Store a fluid grid independently of any structural FEM mesh.

    points, weights, and panel_edges are provisional numerical-array fields.
    Points use physical coordinates in metres; weights represent panel areas
    in m^2 for the plate models. Their exact shapes and ordering are pending.
    """

    points: Any
    weights: Any
    panel_edges: Any
    nx: int
    ny: int

    @classmethod
    def midpoint(cls, geometry: PlateGeometry, *, nx: int, ny: int) -> "FluidGrid":
        """Construct midpoint collocation points and their panel areas."""
        raise NotImplementedError("Midpoint fluid-grid construction is pending.")

    @classmethod
    def chebyshev_gauss(
        cls, geometry: PlateGeometry, *, nx: int, ny: int
    ) -> "FluidGrid":
        """Construct Chebyshev–Gauss points with consistent panel geometry."""
        raise NotImplementedError("Chebyshev–Gauss fluid-grid construction is pending.")
