"""Vernydub's padded fine-grid assembly with hierarchical pressure panels.

The fine lattice stores one integral for each absolute cell offset. A pressure
panel contains an odd number of unit cells in each direction, so its centroid
is a unit-cell centroid. Summing translated unit integrals gives its mobility.
All arrays use muFSI's x-major order, SI units and exp(+i omega t).
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.hydrodynamics.legacy.panel_quadrature import (
    PanelIntegrator,
    QuadratureConvergenceError,
)
from mufsi.hydrodynamics.legacy.stokes_3d import Stokes3D
from mufsi.models.geometry import PlateGeometry


def _partition_sizes(partitions, *, both_edges=False, max_unit_cells=1_000_000):
    """Panel widths in units of the smallest cell; refine the last partition."""
    p = tuple(partitions)
    if not p or any(
        isinstance(n, bool) or not isinstance(n, Integral) or n < 3 or n % 2 == 0
        for n in p
    ):
        raise ValueError("Partitions must be a nonempty sequence of odd integers >=3.")
    total = 1
    for n in p:
        total *= int(n)
    if total > max_unit_cells:
        raise ValueError("Fine lattice exceeds max_unit_cells.")
    if len(p) == 1:
        return np.ones(p[0], dtype=np.int64)
    edge, parent = [], total // p[0]
    for i, n in enumerate(p[1:]):
        parent //= n
        edge.extend([parent] * (n if i == len(p) - 2 else n - 1))
    middle = [total // p[0]] * (p[0] - (2 if both_edges else 1))
    return np.asarray(edge[::-1] + middle + edge if both_edges else middle + edge)


def multigrid_fluid_grid(
    geometry: PlateGeometry,
    *,
    x_partitions=(5, 3),
    y_partitions=(5, 3),
    max_unit_cells=1_000_000,
):
    """Edge-refined pressure grid with exact panel-area force weights.

    x refines only the free tip; y refines both edges. Single entries give a
    uniform midpoint grid. Odd factors ensure fine/coarse centroid alignment.
    Unlike a Chebyshev grid, Q[i] is exactly dx_panel[i]*dy_panel[i], in m^2.
    """
    FluidGrid._geometry(geometry)
    if (
        isinstance(max_unit_cells, bool)
        or not isinstance(max_unit_cells, Integral)
        or max_unit_cells < 1
    ):
        raise ValueError("max_unit_cells must be a positive integer.")
    xs = _partition_sizes(x_partitions, max_unit_cells=max_unit_cells)
    ys = _partition_sizes(y_partitions, both_edges=True, max_unit_cells=max_unit_cells)
    if int(xs.sum()) * int(ys.sum()) > max_unit_cells:
        raise ValueError("Fine lattice exceeds max_unit_cells.")
    xe = np.r_[0, np.cumsum(xs)] * (geometry.length / int(xs.sum()))
    ye = np.r_[0, np.cumsum(ys)] * (geometry.width / int(ys.sum())) - geometry.width / 2
    xe[-1], ye[-1] = geometry.length, geometry.width / 2
    x, y = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    return FluidGrid(
        np.column_stack((np.repeat(x, len(y)), np.tile(y, len(x)))),
        (np.diff(xe)[:, None] * np.diff(ye)[None, :]).ravel(),
        ye,
        len(x),
        len(y),
        x_panel_edges=xe,
    )


@dataclass(frozen=True)
class MultigridIntegrationReport:
    omega: float
    quadrature_backend: str
    fine_shape: tuple[int, int]
    pressure_shape: tuple[int, int]
    evaluated_unit_cells: int
    refined_entries: int
    max_relative_error: float
    max_error_ratio: float
    max_refinement: int


@dataclass(frozen=True)
class Stokes3DMultigrid(Stokes3D):
    """Padded-lattice mobility, using Stokes3D's pressure/LU/cache interface.

    Assembly is replaced completely: integrate just mx*my unit cells once,
    then sum them into the dense pressure matrix. The existing stable kernel,
    adaptive regular integration and analytic singular radial integral are
    reused. This avoids the thesis's fixed polygonal approximation of arcs.
    Dense pressure LU and full structural coupling remain unchanged.
    """

    max_unit_cells: int = 1_000_000

    def __post_init__(self):
        super().__post_init__()
        if (
            isinstance(self.max_unit_cells, bool)
            or not isinstance(self.max_unit_cells, Integral)
            or self.max_unit_cells < 1
        ):
            raise ValueError("max_unit_cells must be a positive integer.")
        lattice = []
        for edges, points in (
            (self.grid.x_panel_edges, self.grid.x),
            (self.grid.panel_edges, self.grid.y),
        ):
            widths = np.diff(edges)
            unit = widths.min()
            ratios = widths / unit
            if np.max(ratios) > self.max_unit_cells:
                raise ValueError("Fine lattice exceeds max_unit_cells.")
            sizes = np.rint(ratios).astype(np.int64)
            if (
                not np.allclose(ratios, sizes, rtol=0, atol=1e-9)
                or np.any(sizes % 2 == 0)
                or not np.allclose(
                    points,
                    (edges[:-1] + edges[1:]) / 2,
                    rtol=0,
                    atol=1e-12 * np.ptp(edges),
                )
            ):
                raise ValueError(
                    "Multigrid needs midpoint panels with odd integer unit widths; "
                    "use multigrid_fluid_grid."
                )
            starts = np.r_[0, np.cumsum(sizes)[:-1]]
            lattice.append((float(unit), sizes, starts, starts + sizes // 2))
        if int(lattice[0][1].sum()) * int(lattice[1][1].sum()) > self.max_unit_cells:
            raise ValueError("Fine lattice exceeds max_unit_cells.")
        areas = (
            np.diff(self.grid.x_panel_edges)[:, None]
            * np.diff(self.grid.panel_edges)[None, :]
        ).ravel()
        if not np.allclose(self.grid.weights, areas, rtol=1e-12, atol=0):
            raise ValueError("Multigrid force weights must equal actual panel areas.")
        object.__setattr__(self, "_lattice", lattice)

    def assemble_matrix(self, omega: float):
        """Return B in v=Bp; units m/(Pa s), complex128, x-major ordering."""
        if not np.isfinite(omega) or omega <= 0:
            raise ValueError("omega must be finite and strictly positive.")
        if self._omega == omega:
            return self._matrix
        self.clear_cache()
        dx, xs, xstarts, xc = self._lattice[0]
        dy, ys, ystarts, yc = self._lattice[1]
        mx, my = int(xs.sum()), int(ys.sum())
        ix, iy = np.meshgrid(np.arange(mx), np.arange(my), indexing="ij")
        bounds = np.column_stack(
            (
                (ix.ravel() - 0.5) * dx,
                (ix.ravel() + 0.5) * dx,
                (iy.ravel() - 0.5) * dy,
                (iy.ravel() + 0.5) * dy,
            )
        )
        # Budget absolute error across the largest assembled source panel.
        integrator = PanelIntegrator(
            self.quadrature_backend,
            self.tolerance / 4,
            self.absolute_tolerance / (int(xs.max()) * int(ys.max())),
            self.max_refinements,
            self.max_subpanels,
            self.batch_size,
        )
        lam = np.sqrt(1j * omega / self.fluid.kinematic_viscosity)
        origin = np.zeros(2)
        values, errors, refined, depth = integrator.regular(origin, bounds[1:], lam)
        diagonal, diagonal_error, diag_depth = integrator.singular(
            origin, bounds[0], lam
        )
        table = np.r_[diagonal, values].reshape(mx, my)
        error_table = np.r_[diagonal_error, errors].reshape(mx, my)
        n = self.grid.nx * self.grid.ny
        matrix = np.empty((n, n), dtype=complex)
        max_relative, max_ratio = 0.0, 0.0
        for row, (cx, cy) in enumerate((x, y) for x in xc for y in yc):
            indices = np.ix_(np.abs(np.arange(mx) - cx), np.abs(np.arange(my) - cy))
            cutout, error_cutout = table[indices], error_table[indices]
            summed = np.add.reduceat(
                np.add.reduceat(cutout, xstarts, axis=0), ystarts, axis=1
            ).ravel()
            estimated = np.add.reduceat(
                np.add.reduceat(error_cutout, xstarts, axis=0), ystarts, axis=1
            ).ravel()
            target = self.absolute_tolerance + self.tolerance * np.abs(summed)
            max_relative = max(
                max_relative,
                float(
                    np.max(estimated / np.maximum(np.abs(summed), np.finfo(float).tiny))
                ),
            )
            max_ratio = max(
                max_ratio,
                float(np.max(estimated / np.maximum(target, np.finfo(float).tiny))),
            )
            if np.any(estimated > target):
                raise QuadratureConvergenceError(
                    "Summed unit-cell error exceeds the pressure-panel tolerance; "
                    "tighten the integration tolerance or refine the pressure grid."
                )
            matrix[row] = summed / self.fluid.dynamic_viscosity
            if self.progress is not None:
                self.progress(row + 1, n)
        matrix.setflags(write=False)
        object.__setattr__(self, "_matrix", matrix)
        object.__setattr__(self, "_omega", omega)
        object.__setattr__(
            self,
            "integration_report",
            MultigridIntegrationReport(
                omega,
                self.quadrature_backend,
                (mx, my),
                (self.grid.nx, self.grid.ny),
                mx * my,
                refined,
                max_relative,
                max_ratio,
                max(depth, diag_depth),
            ),
        )
        return matrix
