"""Fluid collocation points, panel boundaries, and integration weights."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from mufsi.hydrodynamics.quadrature import (
    chebyshev_gauss_nodes,
    midpoint_rule,
    simpson_rule,
)
from mufsi.models.geometry import PlateGeometry


@dataclass(frozen=True)
class FluidGrid:
    """Tensor grid with x-major point ordering and positive area weights.

    points has shape (nx*ny, 2), weights (nx*ny,), and panel_edges (ny+1,).
    The transverse panels are shared by all x sections. Weights integrate
    pressure over area; they need not equal the collocation panel areas.
    Optional x_panel_edges has shape (nx+1,) and defines the rectangles for
    3D panel integration. Arrays are copied and made read-only.
    """

    points: np.ndarray
    weights: np.ndarray
    panel_edges: np.ndarray
    nx: int
    ny: int
    x_panel_edges: np.ndarray | None = field(default=None, kw_only=True)

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
        if self.x_panel_edges is not None:
            edges = np.array(self.x_panel_edges, dtype=float, copy=True)
            if edges.shape != (self.nx + 1,) or not np.isfinite(edges).all():
                raise ValueError("x_panel_edges must contain nx+1 finite boundaries.")
            if (
                np.any(np.diff(edges) <= 0)
                or np.any(self.x <= edges[:-1])
                or np.any(self.x >= edges[1:])
            ):
                raise ValueError("Each x node must lie strictly inside its panel.")
            edges.setflags(write=False)
            object.__setattr__(self, "x_panel_edges", edges)

    @property
    def x(self):
        return self.points.reshape(self.nx, self.ny, 2)[:, 0, 0]

    @property
    def y(self):
        return self.points[: self.ny, 1]

    @property
    def panel_bounds(self):
        """Return (nx*ny,4) bounds [x_left,x_right,y_low,y_up] in grid order."""
        if self.x_panel_edges is None:
            raise ValueError(
                "3D panels need x_panel_edges. Use FluidGrid.cantilever or midpoint."
            )
        return np.column_stack(
            (
                np.repeat(self.x_panel_edges[:-1], self.ny),
                np.repeat(self.x_panel_edges[1:], self.ny),
                np.tile(self.panel_edges[:-1], self.nx),
                np.tile(self.panel_edges[1:], self.nx),
            )
        )

    @staticmethod
    def _geometry(geometry):
        if (
            not np.isfinite([geometry.length, geometry.width]).all()
            or min(geometry.length, geometry.width) <= 0
        ):
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
            np.repeat(wx, ny) * np.tile(wy, nx),
            edges,
            nx,
            ny,
            x_panel_edges=np.linspace(0, geometry.length, nx + 1),
        )

    @classmethod
    def cantilever(
        cls,
        geometry: PlateGeometry,
        *,
        nx: int,
        ny: int,
        x_uniform: bool = False,
        y_uniform: bool = False,
    ):
        """Legacy F3D grid: half Chebyshev in x, full Chebyshev in y.

        Half-Chebyshev nodes cluster at the free tip x=L. Counts are preserved,
        including even/odd counts. Uniform options use midpoint panels over
        the full physical dimensions.
        """
        cls._geometry(geometry)
        if isinstance(nx, bool) or not isinstance(nx, Integral) or nx < 1:
            raise ValueError("nx must be a positive integer.")
        if x_uniform:
            x, wx = midpoint_rule(0, geometry.length, nx)
            xedges = np.linspace(0, geometry.length, nx + 1)
        else:
            theta = (2 * np.arange(1, nx + 1) - 1) * np.pi / (4 * nx)
            x = geometry.length * np.sin(theta)
            wx = np.pi / (2 * nx) * np.sqrt(geometry.length**2 - x**2)
            xedges = np.concatenate(([0.0], (x[:-1] + x[1:]) / 2, [geometry.length]))
        if y_uniform:
            y, wy = midpoint_rule(-geometry.width / 2, geometry.width / 2, ny)
            yedges = np.linspace(-geometry.width / 2, geometry.width / 2, ny + 1)
        else:
            y = chebyshev_gauss_nodes(-geometry.width / 2, geometry.width / 2, ny)
            wy = np.pi / ny * np.sqrt((geometry.width / 2) ** 2 - y**2)
            yedges = np.concatenate(
                (
                    [-geometry.width / 2],
                    (y[:-1] + y[1:]) / 2,
                    [geometry.width / 2],
                )
            )
        return cls(
            np.column_stack((np.repeat(x, ny), np.tile(y, nx))),
            np.repeat(wx, ny) * np.tile(wy, nx),
            yedges,
            nx,
            ny,
            x_panel_edges=xedges,
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
        wy = np.pi / ny * np.sqrt((geometry.width / 2) ** 2 - y**2)
        edges = np.concatenate(
            (
                [-geometry.width / 2],
                (y[:-1] + y[1:]) / 2,
                [geometry.width / 2],
            )
        )
        return cls(
            np.column_stack((np.repeat(x, ny), np.tile(y, nx))),
            np.repeat(wx, ny) * np.tile(wy, nx),
            edges,
            nx,
            ny,
        )
