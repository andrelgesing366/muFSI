"""Experimental edge grids and hybrid planar unsteady-Stokeslet integration.

Run the staged benchmark (one MPI rank, scientific muFSI environment)::

    PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        .venv/bin/python research/stokeslet_hybrid.py --output results/hybrid

Pressure remains piecewise constant on rectangular panels. This is not a
weighted-density/screen formulation. B already integrates the source area;
Q is the physical panel area for projection G = E.T @ diag(Q).
"""

from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np

from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.hydrodynamics.legacy.panel_analytic import AnalyticPanelIntegrator
from mufsi.hydrodynamics.legacy.panel_quadrature import (
    PanelIntegrator,
    QuadratureConvergenceError,
)
from mufsi.hydrodynamics.legacy.stokes_3d import Stokes3D
from mufsi.hydrodynamics.legacy.stokeslet_multigrid import (
    MultigridIntegrationReport,
    Stokes3DMultigrid,
    _partition_sizes,
)


def _area_grid(xedges, yedges):
    x, y = (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2
    return FluidGrid(
        np.column_stack((np.repeat(x, len(y)), np.tile(y, len(x)))),
        np.outer(np.diff(xedges), np.diff(yedges)).ravel(),
        yedges,
        len(x),
        len(y),
        x_panel_edges=xedges,
    )


def edge_clustered_grid(geometry, *, nx, ny, x_clustering="both"):
    """Cosine-spaced *boundaries*, midpoint collocation, exact area weights.

    x_clustering='both' resolves the root and tip; 'tip' clusters only at L;
    'uniform' enables longitudinal translation reuse. y resolves both edges.
    Unlike native Chebyshev collocation, the nodes are panel midpoints.
    """
    FluidGrid._geometry(geometry)
    for count in (nx, ny):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError("nx and ny must be positive integers.")
    t = np.linspace(0, 1, nx + 1)
    if x_clustering == "both":
        xe = geometry.length * (1 - np.cos(np.pi * t)) / 2
    elif x_clustering == "tip":
        xe = geometry.length * np.sin(np.pi * t / 2)
    elif x_clustering == "uniform":
        xe = geometry.length * t
    else:
        raise ValueError("x_clustering must be both, tip, or uniform.")
    ye = -geometry.width * np.cos(np.linspace(0, np.pi, ny + 1)) / 2
    return _area_grid(xe, ye)


def hierarchical_grid(
    geometry,
    *,
    x_partitions=(5, 3),
    y_partitions=(5, 3),
    both_x_edges=True,
    max_unit_cells=1_000_000,
):
    """Odd-width midpoint panels compatible with padded-lattice reuse."""
    FluidGrid._geometry(geometry)
    if (
        isinstance(max_unit_cells, bool)
        or not isinstance(max_unit_cells, Integral)
        or max_unit_cells < 1
    ):
        raise ValueError("max_unit_cells must be a positive integer.")
    xs = _partition_sizes(
        x_partitions, both_edges=both_x_edges, max_unit_cells=max_unit_cells
    )
    ys = _partition_sizes(y_partitions, both_edges=True, max_unit_cells=max_unit_cells)
    if int(xs.sum()) * int(ys.sum()) > max_unit_cells:
        raise ValueError("Fine lattice exceeds max_unit_cells.")
    xe = np.r_[0, np.cumsum(xs)] * (geometry.length / int(xs.sum()))
    ye = np.r_[0, np.cumsum(ys)] * (geometry.width / int(ys.sum())) - geometry.width / 2
    xe[-1], ye[-1] = geometry.length, geometry.width / 2
    return _area_grid(xe, ye)


def lattice_edge_grid(geometry, *, nx, ny, max_unit_cells=1_000_000):
    """Cosine-shaped edge clustering quantized to odd unit-cell widths.

    The smallest panel is one unit wide. All centroids align with the unit
    lattice, so the hybrid reuse model can use essentially the same pressure
    distribution as direct cosine panels. Counts nx/ny are preserved. The fine
    table grows approximately as (nx*ny)**2; max_unit_cells limits this cost.
    """
    if (
        isinstance(max_unit_cells, bool)
        or not isinstance(max_unit_cells, Integral)
        or max_unit_cells < 1
    ):
        raise ValueError("max_unit_cells must be a positive integer.")
    for count in (nx, ny):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError("nx and ny must be positive integers.")
    if int(nx) * int(ny) > max_unit_cells:
        raise ValueError("Fine lattice exceeds max_unit_cells.")
    ideal = edge_clustered_grid(geometry, nx=nx, ny=ny)
    sizes = []
    for edges in (ideal.x_panel_edges, ideal.panel_edges):
        widths = np.diff(edges)
        widths = (widths + widths[::-1]) / 2
        ratios = widths / widths.min()
        sizes.append(2 * np.rint((ratios - 1) / 2).astype(np.int64) + 1)
    xs, ys = sizes
    if int(xs.sum()) * int(ys.sum()) > max_unit_cells:
        raise ValueError("Fine lattice exceeds max_unit_cells.")
    xe = np.r_[0, np.cumsum(xs)] * (geometry.length / int(xs.sum()))
    ye = np.r_[0, np.cumsum(ys)] * (geometry.width / int(ys.sum())) - geometry.width / 2
    xe[-1], ye[-1] = geometry.length, geometry.width / 2
    return _area_grid(xe, ye)


class HybridPanelIntegrator(PanelIntegrator):
    """Radial near field, degree 2/4/6/8 cubature far field, radial fallback.

    Distance is from the observation to the rectangle, scaled by its diagonal.
    This handles aspect ratio without a fixed neighbor count. Frequency enters
    the actual complex-kernel rule comparison, so a far panel can still switch
    to radial integration. Estimates are diagnostic, not rigorous bounds.
    """

    def __init__(
        self,
        backend="quadpy",
        rtol=2e-3,
        atol=1e-15,
        max_refinements=12,
        max_subpanels=16384,
        batch_size=256,
        *,
        near_ratio=0.5,
        angular_order=8,
        angular_refinements=6,
        analytic_only=False,
    ):
        super().__init__(
            backend, rtol, atol, max_refinements, max_subpanels, batch_size
        )
        if not np.isfinite(near_ratio) or near_ratio < 0:
            raise ValueError("near_ratio must be finite and nonnegative.")
        self.near_ratio, self.analytic_only = near_ratio, analytic_only
        self.radial = AnalyticPanelIntegrator(
            angular_order, rtol, atol, angular_refinements, batch_size
        )
        self.reset_counts()

    def reset_counts(self):
        self.counts = {
            "near_radial": 0,
            "singular_radial": 0,
            "far_cubature": 0,
            "fallback_radial": 0,
            "angular_refined": 0,
            "max_angular_order": 0,
        }

    def _radial(self, observation, bounds, lam):
        values, errors, refined, order = self.radial.integrate(observation, bounds, lam)
        self.counts["angular_refined"] += refined
        self.counts["max_angular_order"] = max(order, self.counts["max_angular_order"])
        return values, errors, refined, 0

    def regular(self, observation, bounds, lam):
        bounds = np.asarray(bounds, dtype=float)
        widths = bounds[:, [1, 3]] - bounds[:, [0, 2]]
        gaps = np.maximum(
            np.maximum(
                bounds[:, [0, 2]] - observation, observation - bounds[:, [1, 3]]
            ),
            0,
        )
        near = np.linalg.norm(gaps, axis=1) <= self.near_ratio * np.linalg.norm(
            widths, axis=1
        )
        if self.analytic_only:
            near[:] = True
        values, errors = np.empty(len(bounds), complex), np.empty(len(bounds))
        refined = 0
        if np.any(near):
            values[near], errors[near], refined, _ = self._radial(
                observation, bounds[near], lam
            )
        self.counts["near_radial"] += int(near.sum())
        far = np.flatnonzero(~near)
        if len(far):
            coarse = self.rectangles(observation, bounds[far], lam, 2)
            values[far] = self.rectangles(observation, bounds[far], lam, 4)
            errors[far] = np.abs(values[far] - coarse)
            for degree in (6, 8):
                pending = far[errors[far] > self._target(values[far])]
                if not len(pending):
                    break
                new = self.rectangles(observation, bounds[pending], lam, degree)
                errors[pending] = np.abs(new - values[pending])
                values[pending] = new
            pending = far[errors[far] > self._target(values[far])]
            self.counts["far_cubature"] += len(far) - len(pending)
            self.counts["fallback_radial"] += len(pending)
            if len(pending):
                values[pending], errors[pending], extra, _ = self._radial(
                    observation, bounds[pending], lam
                )
                refined += len(pending) + extra
        return values, errors, refined, 0

    def singular(self, observation, bounds, lam):
        values, errors, _refined, depth = self._radial(
            observation, np.asarray(bounds).reshape(1, 4), lam
        )
        self.counts["singular_radial"] += 1
        return values[0], errors[0], depth


@dataclass(frozen=True)
class Stokes3DHybrid(Stokes3D):
    """Direct hybrid assembly; compatible with the existing full FE Schur solve."""

    near_ratio: float = 0.5
    angular_order: int = 8
    angular_refinements: int = 6
    analytic_only: bool = False

    def _make_integrator(self, tolerance=None, absolute_tolerance=None):
        return HybridPanelIntegrator(
            self.quadrature_backend,
            self.tolerance if tolerance is None else tolerance,
            self.absolute_tolerance
            if absolute_tolerance is None
            else absolute_tolerance,
            self.max_refinements,
            self.max_subpanels,
            self.batch_size,
            near_ratio=self.near_ratio,
            angular_order=self.angular_order,
            angular_refinements=self.angular_refinements,
            analytic_only=self.analytic_only,
        )

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "_integrator", self._make_integrator())
        object.__setattr__(self, "hybrid_report", None)

    def clear_cache(self):
        super().clear_cache()
        object.__setattr__(self, "hybrid_report", None)

    def assemble_matrix(self, omega):
        if self._omega == omega:
            return self._matrix
        self._integrator.reset_counts()
        matrix = super().assemble_matrix(omega)
        report = asdict(self.integration_report)
        report["quadrature_backend"] = "radial" if self.analytic_only else "hybrid"
        report.update(self._integrator.counts)
        object.__setattr__(self, "hybrid_report", report)
        return matrix


@dataclass(frozen=True)
class HybridMultigridIntegrationReport(MultigridIntegrationReport):
    prefix_roundoff_allowance: float = 0.0
    prefix_recomputed_entries: int = 0
    near_radial: int = 0
    singular_radial: int = 0
    far_cubature: int = 0
    fallback_radial: int = 0
    angular_refined: int = 0
    max_angular_order: int = 0


@dataclass(frozen=True)
class Stokes3DHybridMultigrid(Stokes3DMultigrid):
    """Hybrid unit-cell table, with summed-area reuse for all pressure panels.

    A prefix sum over signed offsets replaces a fine-lattice cutout per row.
    Prefix construction/queries cost O(M + N**2), storage O(M + N**2); M is
    the offset-table cell count, N the pressure unknown count. Entries whose
    prefix roundoff allowance is too large are recomputed by direct sums.
    Dense LU/Schur costs remain unchanged.
    """

    near_ratio: float = 0.5
    angular_order: int = 8
    angular_refinements: int = 6
    analytic_only: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Also validate the research controls at construction time.
        object.__setattr__(self, "_integrator", Stokes3DHybrid._make_integrator(self))
        object.__setattr__(self, "hybrid_report", None)

    def clear_cache(self):
        super().clear_cache()
        object.__setattr__(self, "hybrid_report", None)

    def assemble_matrix(self, omega):
        if not np.isfinite(omega) or omega <= 0:
            raise ValueError("omega must be finite and strictly positive.")
        if self._omega == omega:
            return self._matrix
        self.clear_cache()
        dx, xs, starts_x, centers_x = self._lattice[0]
        dy, ys, starts_y, centers_y = self._lattice[1]
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
        integrator = Stokes3DHybrid._make_integrator(
            self,
            self.tolerance / 4,
            self.absolute_tolerance / (int(xs.max()) * int(ys.max())),
        )
        lam = np.sqrt(1j * omega / self.fluid.kinematic_viscosity)
        values, errors, refined, _ = integrator.regular(np.zeros(2), bounds[1:], lam)
        diagonal, diagonal_error, _ = integrator.singular(np.zeros(2), bounds[0], lam)
        table = np.r_[diagonal, values].reshape(mx, my)
        error_table = np.r_[diagonal_error, errors].reshape(mx, my)
        signed = np.ix_(np.abs(np.arange(1 - mx, mx)), np.abs(np.arange(1 - my, my)))

        def prefix(array):
            return np.pad(array[signed], ((1, 0), (1, 0))).cumsum(0).cumsum(1)

        summed_area, summed_error = prefix(table), prefix(error_table)
        sx, sy = np.repeat(starts_x, len(ys)), np.tile(starts_y, len(xs))
        ex, ey = sx + np.repeat(xs, len(ys)), sy + np.tile(ys, len(xs))
        cx, cy = np.repeat(centers_x, len(ys)), np.tile(centers_y, len(xs))
        n = len(cx)
        matrix = np.empty((n, n), complex)
        max_relative, max_ratio, recomputed = 0.0, 0.0, 0
        # Conservative roundoff allowance for prefix subtraction, in metres.
        roundoff = 8 * np.finfo(float).eps * (mx + my) * np.abs(table).sum() * 4
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            x0, x1 = (
                sx[None, :] - cx[start:stop, None] + mx - 1,
                ex[None, :] - cx[start:stop, None] + mx - 1,
            )
            y0, y1 = (
                sy[None, :] - cy[start:stop, None] + my - 1,
                ey[None, :] - cy[start:stop, None] + my - 1,
            )

            def rectangles(array, x0=x0, x1=x1, y0=y0, y1=y1):
                return array[x1, y1] - array[x0, y1] - array[x1, y0] + array[x0, y0]

            value = rectangles(summed_area)
            estimated = np.maximum(rectangles(summed_error), 0) + roundoff
            target = self.absolute_tolerance + self.tolerance * np.abs(value)
            # Prefix subtraction can lose digits for tiny distant panels. Sum
            # their cells directly rather than weakening the requested target.
            for local_row, column in zip(*np.nonzero(estimated > target), strict=True):
                row = start + local_row
                xindices = np.abs(np.arange(sx[column], ex[column]) - cx[row])
                yindices = np.abs(np.arange(sy[column], ey[column]) - cy[row])
                indices = np.ix_(xindices, yindices)
                cells = table[indices]
                value[local_row, column] = cells.sum()
                estimate = error_table[indices].sum()
                estimate += 8 * np.finfo(float).eps * cells.size * abs(cells).sum()
                estimated[local_row, column] = estimate
                recomputed += 1
            target = self.absolute_tolerance + self.tolerance * np.abs(value)
            max_relative = max(
                max_relative,
                float(np.max(estimated / np.maximum(abs(value), np.finfo(float).tiny))),
            )
            max_ratio = max(
                max_ratio,
                float(np.max(estimated / np.maximum(target, np.finfo(float).tiny))),
            )
            if np.any(estimated > target):
                raise QuadratureConvergenceError(
                    "Aggregated unit-cell estimate (including prefix roundoff) exceeds "
                    "the requested pressure-panel tolerance."
                )
            matrix[start:stop] = value / self.fluid.dynamic_viscosity
            if self.progress is not None:
                self.progress(stop, n)
        matrix.setflags(write=False)
        object.__setattr__(self, "_matrix", matrix)
        object.__setattr__(self, "_omega", omega)
        report = dict(
            omega=float(omega),
            quadrature_backend="hybrid_multigrid",
            fine_shape=(mx, my),
            pressure_shape=(self.grid.nx, self.grid.ny),
            evaluated_unit_cells=mx * my,
            refined_entries=refined,
            max_relative_error=max_relative,
            max_error_ratio=max_ratio,
            max_refinement=0,
            prefix_roundoff_allowance=roundoff,
            prefix_recomputed_entries=recomputed,
            **integrator.counts,
        )
        object.__setattr__(
            self, "integration_report", HybridMultigridIntegrationReport(**report)
        )
        object.__setattr__(self, "hybrid_report", report)
        return matrix

