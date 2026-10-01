"""Adaptive 3D Stokeslet panel formulation ported from Fluid/F3D.py."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mufsi.hydrodynamics.base import HydrodynamicModel
from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.hydrodynamics.panel_quadrature import IntegrationReport, PanelIntegrator
from mufsi.models.fluid import Fluid
from mufsi.solvers.linear import LinearSolver, SciPyLUSolver


@dataclass(frozen=True)
class Stokes3D(HydrodynamicModel):
    """Thin-plate single-layer mobility with all longitudinal interactions.

    v = B p; p is resisting traction in Pa with exp(+i omega t).
    B integrates Szz/(8*pi*mu) over piecewise-constant pressure panels.
    The fluid mobility is dense, complex128, and only the latest frequency is
    cached. LU solves replace the legacy inverse. This is the adaptive 2D panel
    quadrature version, not the later general analytical 1D panel reduction.
    """

    fluid: Fluid
    grid: FluidGrid
    solver: LinearSolver | None = None
    tolerance: float = 2e-3
    absolute_tolerance: float = 1e-15
    quadrature_backend: str = "quadpy"
    max_refinements: int = 12
    max_subpanels: int = 16384
    batch_size: int = 256
    use_symmetry: bool = True
    progress: Callable[[int, int], None] | None = None

    def __post_init__(self):
        if (
            not np.isfinite([self.fluid.density, self.fluid.dynamic_viscosity]).all()
            or min(self.fluid.density, self.fluid.dynamic_viscosity) <= 0
        ):
            raise ValueError("Fluid density and dynamic viscosity must be positive.")
        _ = self.grid.panel_bounds  # Validate the longitudinal panel geometry.
        object.__setattr__(
            self,
            "_integrator",
            PanelIntegrator(
                self.quadrature_backend,
                self.tolerance,
                self.absolute_tolerance,
                self.max_refinements,
                self.max_subpanels,
                self.batch_size,
            ),
        )
        object.__setattr__(
            self, "solver", SciPyLUSolver() if self.solver is None else self.solver
        )
        object.__setattr__(self, "_omega", None)
        object.__setattr__(self, "_matrix", None)
        object.__setattr__(self, "_factorized_omega", None)
        object.__setattr__(self, "integration_report", None)

    def assemble_matrix(self, omega: float):
        """Return read-only dense mobility B, in m/(Pa s).

        Reuses transverse reflection when the grid is symmetric. Uniform x
        panels also reuse translation/reflection across x; nonuniform grids
        evaluate each longitudinal observation section independently.
        """
        if not np.isfinite(omega) or omega <= 0:
            raise ValueError("omega must be finite and strictly positive.")
        if self._omega == omega:
            return self._matrix
        g = self.grid
        bounds = g.panel_bounds
        nx, ny = g.nx, g.ny
        matrix = np.empty((nx * ny, nx * ny), dtype=complex)
        lam = np.sqrt(1j * omega / self.fluid.kinematic_viscosity)
        atol = 1e-12 * max(np.ptp(g.x_panel_edges), np.ptp(g.panel_edges))
        mirror_y = (
            self.use_symmetry
            and np.allclose(
                g.y,
                g.panel_edges[0] + g.panel_edges[-1] - g.y[::-1],
                rtol=0,
                atol=atol,
            )
            and np.allclose(
                g.panel_edges,
                g.panel_edges[0] + g.panel_edges[-1] - g.panel_edges[::-1],
                rtol=0,
                atol=atol,
            )
        )
        dx = np.diff(g.x_panel_edges)
        uniform_x = self.use_symmetry and np.allclose(dx, dx[0], rtol=1e-12, atol=atol)
        uniform_x = uniform_x and np.allclose(
            g.x,
            (g.x_panel_edges[:-1] + g.x_panel_edges[1:]) / 2,
            rtol=0,
            atol=atol,
        )
        x_count, y_count = (1 if uniform_x else nx), ((ny + 1) // 2 if mirror_y else ny)
        total = x_count * y_count
        columns_reflected_y = np.arange(nx * ny).reshape(nx, ny)[:, ::-1].ravel()
        refined, max_error, max_ratio, max_depth, count = 0, 0.0, 0.0, 0, 0
        for ix in range(x_count):
            for iy in range(y_count):
                row = ix * ny + iy
                observation = g.points[row]
                regular = np.arange(nx * ny) != row
                values, errors, n_refined, depth = self._integrator.regular(
                    observation,
                    bounds[regular],
                    lam,
                )
                diagonal, diagonal_error, diag_depth = self._integrator.singular(
                    observation,
                    bounds[row],
                    lam,
                )
                matrix[row, regular], matrix[row, row] = values, diagonal
                denom = np.maximum(np.abs(values), np.finfo(float).tiny)
                relative = np.max(errors / denom, initial=0.0)
                targets = np.maximum(
                    self._integrator._target(values), np.finfo(float).tiny
                )
                ratio = np.max(errors / targets, initial=0.0)
                max_ratio = max(
                    max_ratio,
                    ratio,
                    diagonal_error
                    / max(self._integrator._target(diagonal), np.finfo(float).tiny),
                )
                max_error = max(
                    max_error,
                    relative,
                    diagonal_error / max(abs(diagonal), np.finfo(float).tiny),
                )
                refined += n_refined
                max_depth = max(max_depth, depth, diag_depth)
                if mirror_y and iy != ny - 1 - iy:
                    matrix[ix * ny + ny - 1 - iy] = matrix[row, columns_reflected_y]
                count += 1
                if self.progress is not None:
                    self.progress(count, total)
        if uniform_x:
            blocks = matrix[:ny].reshape(ny, nx, ny).transpose(1, 0, 2).copy()
            for ix in range(nx):
                matrix[ix * ny : (ix + 1) * ny] = np.concatenate(
                    [blocks[abs(jx - ix)] for jx in range(nx)],
                    axis=1,
                )
        matrix /= self.fluid.dynamic_viscosity
        matrix.setflags(write=False)
        object.__setattr__(self, "_matrix", matrix)
        object.__setattr__(self, "_omega", omega)
        object.__setattr__(self, "_factorized_omega", None)
        object.__setattr__(
            self,
            "integration_report",
            IntegrationReport(
                omega,
                self.quadrature_backend,
                count,
                refined,
                float(max_error),
                float(max_ratio),
                max_depth,
            ),
        )
        return matrix

    def pressure_from_velocity(self, omega: float, velocity):
        """Solve for one velocity vector or batch (nx*ny[, nrhs])."""
        v = np.asarray(velocity, dtype=complex)
        n = self.grid.nx * self.grid.ny
        if v.ndim not in (1, 2) or v.shape[0] != n or not np.isfinite(v).all():
            raise ValueError("velocity must be finite with shape (nx*ny[, nrhs]).")
        if self._factorized_omega != omega:
            self.solver.factorize(self.assemble_matrix(omega))
            object.__setattr__(self, "_factorized_omega", omega)
        return self.solver.solve(v)

    def clear_cache(self):
        """Release cached mobility/LU factors before a large new calculation."""
        object.__setattr__(self, "_matrix", None)
        object.__setattr__(self, "_omega", None)
        object.__setattr__(self, "_factorized_omega", None)
        object.__setattr__(self, "integration_report", None)
        if isinstance(self.solver, SciPyLUSolver) and hasattr(self.solver, "_factors"):
            del self.solver._factors
