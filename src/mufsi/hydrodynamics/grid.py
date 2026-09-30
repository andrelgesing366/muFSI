"""Fluid collocation points, panel boundaries, and integration weights."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from mufsi.hydrodynamics.quadrature import (
    chebyshev_gauss_nodes, midpoint_rule, simpson_rule,
)
from mufsi.models.geometry import PlateGeometry


@dataclass(frozen=True)
class FluidGrid:
    """Tensor grid with x-major point ordering and positive area weights.

    points has shape (nx*ny, 2), weights (nx*ny,), and panel_edges (ny+1,).
    The transverse panels are shared by all x sections. Weights integrate
    pressure over area; they need not equal the collocation panel areas.
    Arrays are copied and made read-only.
    """

    points: np.ndarray
    weights: np.ndarray
    panel_edges: np.ndarray
    nx: int
    ny: int

    def __post_init__(self):
        for name in ("nx", "ny"):
            n = getattr(self, name)
            if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("points", "weights", "panel_edges"):
            a = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.isfinite(a).all():
                raise ValueError(f"{name} must be finite.")
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        if self.points.shape != (self.nx * self.ny, 2):
            raise ValueError("points must have shape (nx*ny, 2).")
        if self.weights.shape != (self.nx * self.ny,) or np.any(self.weights <= 0):
            raise ValueError("weights must contain one positive area per point.")
        if self.panel_edges.shape != (self.ny + 1,) or np.any(
            np.diff(self.panel_edges) <= 0
        ):
            raise ValueError("panel_edges must contain ny+1 increasing boundaries.")
        p = self.points.reshape(self.nx, self.ny, 2)
        if not np.array_equal(
            p[:, :, 1], np.broadcast_to(p[0, :, 1], p[:, :, 1].shape)
        ):
            raise ValueError("Every section must share the same transverse nodes.")
        if not np.array_equal(
            p[:, :, 0], np.broadcast_to(p[:, :1, 0], p[:, :, 0].shape)
        ):
            raise ValueError("Points must use x-major tensor-grid ordering.")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("Longitudinal sections must be increasing.")
        if np.any(self.y <= self.panel_edges[:-1]) or np.any(
            self.y >= self.panel_edges[1:]
        ):
            raise ValueError("Each transverse node must lie strictly inside its panel.")

    @property
    def x(self):
        return self.points.reshape(self.nx, self.ny, 2)[:, 0, 0]

    @property
    def y(self):
        return self.points[:self.ny, 1]

    @staticmethod
    def _geometry(geometry):
        if not np.isfinite([geometry.length, geometry.width]).all() or min(
            geometry.length, geometry.width
        ) <= 0:
            raise ValueError("Grid length and width must be finite and positive.")

    @classmethod
    def midpoint(cls, geometry: PlateGeometry, *, nx: int, ny: int):
        """Uniform midpoint nodes and exact rectangular panel-area weights."""
        cls._geometry(geometry)
        x, wx = midpoint_rule(0, geometry.length, nx)
        y, wy = midpoint_rule(-geometry.width / 2, geometry.width / 2, ny)
        edges = np.linspace(-geometry.width / 2, geometry.width / 2, ny + 1)
        return cls(
            np.column_stack((np.repeat(x, ny), np.tile(y, nx))),
            np.repeat(wx, ny) * np.tile(wy, nx), edges, nx, ny,
        )

    @classmethod
    def chebyshev_gauss(cls, geometry: PlateGeometry, *, nx: int, ny: int):
        """Legacy F2D rule: Simpson in x, Chebyshev–Gauss in y.

        As in the old notebook, an even nx is increased by one. nx >= 3.
        Transverse weights approximate the ordinary integral dy using the
        Chebyshev rule times sqrt((width/2)^2-y^2); they are not renormalized.
        """
        cls._geometry(geometry)
        if isinstance(nx, bool) or not isinstance(nx, Integral) or nx < 3:
            raise ValueError("nx must be an integer >= 3.")
        nx += nx % 2 == 0
        x, wx = simpson_rule(0, geometry.length, nx)
        y = chebyshev_gauss_nodes(-geometry.width / 2, geometry.width / 2, ny)
        wy = np.pi / ny * np.sqrt((geometry.width / 2)**2 - y**2)
        edges = np.concatenate((
            [-geometry.width / 2], (y[:-1] + y[1:]) / 2, [geometry.width / 2],
        ))
        return cls(
            np.column_stack((np.repeat(x, ny), np.tile(y, nx))),
            np.repeat(wx, ny) * np.tile(wy, nx), edges, nx, ny,
        )
