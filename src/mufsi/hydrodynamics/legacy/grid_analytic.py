"""Uniform longitudinal panels and Chebyshev transverse collocation."""

import numpy as np

from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.hydrodynamics.quadrature import chebyshev_gauss_nodes, midpoint_rule
from mufsi.models.geometry import PlateGeometry


def analytic_fluid_grid(geometry: PlateGeometry, *, nx: int, ny: int) -> FluidGrid:
    """Build the F3D_1D notebook grid in metres, with x-major ordering.

    x uses uniform midpoint panels. y uses Chebyshev--Gauss nodes, midpoint
    panel boundaries, and the original ordinary-integral Chebyshev weights.
    There is deliberately no y_uniform option. Counts, including odd ny and
    single-panel directions, are preserved. Force weights are distinct from
    panel areas, as in the legacy implementation.
    """
    FluidGrid._geometry(geometry)
    x, wx = midpoint_rule(0.0, geometry.length, nx)
    y = chebyshev_gauss_nodes(-geometry.width / 2, geometry.width / 2, ny)
    # sin(theta) avoids cancellation in sqrt((width/2)**2 - y**2).
    theta = (2 * np.arange(1, ny + 1) - 1) * np.pi / (2 * ny)
    wy = geometry.width / 2 * np.pi / ny * np.sin(theta)
    yedges = np.concatenate(
        (
            [-geometry.width / 2],
            (y[:-1] + y[1:]) / 2,
            [geometry.width / 2],
        )
    )
    return FluidGrid(
        np.column_stack((np.repeat(x, ny), np.tile(y, nx))),
        np.repeat(wx, ny) * np.tile(wy, nx),
        yedges,
        nx,
        ny,
        x_panel_edges=np.linspace(0.0, geometry.length, nx + 1),
    )
