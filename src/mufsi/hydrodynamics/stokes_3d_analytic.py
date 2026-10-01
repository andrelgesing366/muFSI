"""F3D_1D analytic radial integration with uniform-x block reuse."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mufsi.hydrodynamics.base import HydrodynamicModel
from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.hydrodynamics.grid_analytic import analytic_fluid_grid
from mufsi.hydrodynamics.panel_analytic import AnalyticPanelIntegrator
from mufsi.models.fluid import Fluid
from mufsi.solvers.linear import LinearSolver, SciPyLUSolver

__all__ = ["AnalyticIntegrationReport", "Stokes3DAnalytic", "analytic_fluid_grid"]


@dataclass(frozen=True)
class AnalyticIntegrationReport:
    """Latest angular error estimates, before division by viscosity."""

    omega: float
    evaluated_rows: int
    evaluated_panels: int
    refined_panels: int
    max_relative_error: float
    max_error_ratio: float
    max_order: int
    reflected_y: bool
    reused_x: bool
    backend: str = "analytic_radial"


@dataclass(frozen=True)
class Stokes3DAnalytic(HydrodynamicModel):
    """Uniform-x planar single-layer mobility v = B p.

    p is resisting traction in Pa, v is m/s, omega is rad/s, and the harmonic
    convention is exp(+i*omega*t), matching the rewritten coupled solvers.
    Radial integration is analytic; angular Gauss quadrature remains.
    Uniform midpoint x-panels give B[i,j] = blocks[abs(i-j)]. y can be
    nonuniform and asymmetric; reflection is used only when geometrically
    valid. No y-translation or y-uniform special case is used.

    Only the latest frequency is cached, in complex128. assemble_blocks and
    apply_mobility avoid storing the full dense matrix. Pressure solves use
    a dense LU of B, without explicitly forming an inverse. use_symmetry=False
    provides independent row assembly for small correctness checks.
    """

    fluid: Fluid
    grid: FluidGrid
    solver: LinearSolver | None = None
    angular_order: int = 8
    tolerance: float = 2e-3
    absolute_tolerance: float = 1e-15
    max_refinements: int = 4
    batch_size: int = 256
    use_symmetry: bool = True
    progress: Callable[[int, int], None] | None = None

    def __post_init__(self):
        if (
            not np.isfinite([self.fluid.density, self.fluid.dynamic_viscosity]).all()
            or min(
                self.fluid.density,
                self.fluid.dynamic_viscosity,
            )
            <= 0
        ):
            raise ValueError("Fluid density and dynamic viscosity must be positive.")
        if self.grid.x_panel_edges is None:
            raise ValueError("3D panels require explicit x_panel_edges.")
        dx = np.diff(self.grid.x_panel_edges)
        atol = 1e-12 * np.ptp(self.grid.x_panel_edges)
        if not np.allclose(dx, dx[0], rtol=1e-12, atol=atol) or not np.allclose(
            self.grid.x,
            (self.grid.x_panel_edges[:-1] + self.grid.x_panel_edges[1:]) / 2,
            rtol=0,
            atol=atol,
        ):
            raise ValueError(
                "Stokes3DAnalytic requires uniform midpoint x-panels; "
                "use analytic_fluid_grid(geometry, nx=..., ny=...)."
            )
        object.__setattr__(
            self,
            "_integrator",
            AnalyticPanelIntegrator(
                self.angular_order,
                self.tolerance,
                self.absolute_tolerance,
                self.max_refinements,
                self.batch_size,
            ),
        )
        object.__setattr__(
            self, "solver", SciPyLUSolver() if self.solver is None else self.solver
        )
        self.clear_cache()

    @staticmethod
    def _validate_omega(omega):
        if not np.isfinite(omega) or omega < 0:
            raise ValueError("omega must be finite and nonnegative.")

    def _assemble(self, omega):
        self._validate_omega(omega)
        if self._omega == omega:
            return
        # Release the previous frequency before allocating its replacement.
        self.clear_cache()
        g = self.grid
        nx, ny = g.nx, g.ny
        atol_y = 1e-12 * np.ptp(g.panel_edges)
        center_sum = g.panel_edges[0] + g.panel_edges[-1]
        mirror_y = (
            self.use_symmetry
            and np.allclose(
                g.y,
                center_sum - g.y[::-1],
                rtol=0,
                atol=atol_y,
            )
            and np.allclose(
                g.panel_edges,
                center_sum - g.panel_edges[::-1],
                rtol=0,
                atol=atol_y,
            )
        )
        x_count = 1 if self.use_symmetry else nx
        y_count = (ny + 1) // 2 if mirror_y else ny
        total = x_count * y_count
        rows = np.empty((x_count * ny, nx * ny), dtype=complex)
        bounds = g.panel_bounds
        lam = np.sqrt(1j * omega / self.fluid.kinematic_viscosity)
        reflected_columns = np.arange(nx * ny).reshape(nx, ny)[:, ::-1].ravel()
        refined, relative_error, error_ratio, max_order, count = 0, 0.0, 0.0, 0, 0
        for ix in range(x_count):
            for iy in range(y_count):
                row = ix * ny + iy
                values, errors, n_refined, order = self._integrator.integrate(
                    g.points[row],
                    bounds,
                    lam,
                )
                rows[row] = values
                if mirror_y and iy != ny - 1 - iy:
                    rows[ix * ny + ny - 1 - iy] = values[reflected_columns]
                target = np.maximum(
                    self.absolute_tolerance, self.tolerance * np.abs(values)
                )
                relative_error = max(
                    relative_error,
                    float(
                        np.max(
                            errors / np.maximum(np.abs(values), np.finfo(float).tiny),
                        )
                    ),
                )
                error_ratio = max(
                    error_ratio,
                    float(
                        np.max(
                            errors / np.maximum(target, np.finfo(float).tiny),
                        )
                    ),
                )
                refined += n_refined
                max_order = max(max_order, order)
                count += 1
                if self.progress is not None:
                    self.progress(count, total)
        rows /= self.fluid.dynamic_viscosity
        if self.use_symmetry:
            blocks = rows.reshape(ny, nx, ny).transpose(1, 0, 2).copy()
            blocks.setflags(write=False)
            object.__setattr__(self, "_blocks", blocks)
        else:
            rows.setflags(write=False)
            object.__setattr__(self, "_matrix", rows)
        object.__setattr__(self, "_omega", omega)
        object.__setattr__(
            self,
            "integration_report",
            AnalyticIntegrationReport(
                omega,
                count,
                count * nx * ny,
                refined,
                relative_error,
                error_ratio,
                max_order,
                bool(mirror_y),
                bool(self.use_symmetry),
            ),
        )

    def assemble_blocks(self, omega):
        """Read-only (nx,ny,ny) mobility blocks, indexed by |source_x-target_x|.

        Storage and panel evaluations scale as nx*ny**2 instead of
        nx**2*ny**2. Requires use_symmetry=True.
        """
        if not self.use_symmetry:
            raise ValueError("assemble_blocks requires use_symmetry=True.")
        self._assemble(omega)
        return self._blocks

    def assemble_matrix(self, omega):
        """Return read-only dense B in m/(Pa*s), shape (nx*ny,nx*ny)."""
        self._assemble(omega)
        if self._matrix is None:
            nx, ny = self.grid.nx, self.grid.ny
            matrix = np.empty((nx * ny, nx * ny), dtype=complex)
            for ix in range(nx):
                for jx in range(nx):
                    matrix[ix * ny : (ix + 1) * ny, jx * ny : (jx + 1) * ny] = (
                        self._blocks[abs(ix - jx)]
                    )
            matrix.setflags(write=False)
            object.__setattr__(self, "_matrix", matrix)
        return self._matrix

    def _field(self, values, name):
        field = np.asarray(values, dtype=complex)
        n = self.grid.nx * self.grid.ny
        if (
            field.ndim not in (1, 2)
            or field.shape[0] != n
            or (field.ndim == 2 and field.shape[1] == 0)
            or not np.isfinite(field).all()
        ):
            raise ValueError(f"{name} must be finite with shape (nx*ny[, nrhs]).")
        return field

    def apply_mobility(self, omega, pressure):
        """Apply B to one/batched pressure field, without dense matrix storage."""
        p = self._field(pressure, "pressure")
        if not self.use_symmetry:
            return self.assemble_matrix(omega) @ p
        blocks = self.assemble_blocks(omega)
        nx, ny = self.grid.nx, self.grid.ny
        sections = p.reshape(nx, ny, -1)
        result = np.einsum("ij,xjk->xik", blocks[0], sections)
        for offset in range(1, nx):
            result[:-offset] += np.einsum(
                "ij,xjk->xik", blocks[offset], sections[offset:]
            )
            result[offset:] += np.einsum(
                "ij,xjk->xik", blocks[offset], sections[:-offset]
            )
        return result.reshape(p.shape)

    def pressure_from_velocity(self, omega, velocity):
        """Solve B p = v using cached dense LU, for one or many RHSs."""
        v = self._field(velocity, "velocity")
        if self._factorized_omega != omega:
            self.solver.factorize(self.assemble_matrix(omega))
            object.__setattr__(self, "_factorized_omega", omega)
        return self.solver.solve(v)

    def clear_cache(self):
        """Release latest matrix/blocks/report and invalidate the pressure LU."""
        for name in (
            "_omega",
            "_matrix",
            "_blocks",
            "_factorized_omega",
            "integration_report",
        ):
            object.__setattr__(self, name, None)
        if isinstance(self.solver, SciPyLUSolver) and hasattr(self.solver, "_factors"):
            del self.solver._factors
