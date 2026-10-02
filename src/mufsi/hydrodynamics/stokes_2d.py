"""Legacy F2D unsteady Stokes panel model, independent of DOLFINx."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import keip, kerp

from mufsi.hydrodynamics.base import HydrodynamicModel
from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.models.fluid import Fluid
from mufsi.solvers.linear import LinearSolver, SciPyLUSolver


def _kelvin_primitive(z):
    """Odd extension of 1/z + ker'(z) + i kei'(z).

    The small-argument series avoids subtracting the cancelling 1/z terms.
    Collocation nodes lie inside panels, so zero arguments are excluded.
    """
    x = np.abs(z)
    if np.any(x == 0):
        raise ValueError("A collocation node coincides with a panel boundary.")
    f = np.empty(x.shape, dtype=np.complex128)
    small = x < 1e-3
    t = np.log(x[small] / 2) + np.euler_gamma + 1j * np.pi / 4
    f[small] = -1j * x[small] / 2 * (t - 0.5)
    f[small] += x[small]**3 / 16 * (t - 1.25)
    f[~small] = 1 / x[~small] + kerp(x[~small]) + 1j * keip(x[~small])
    return np.sign(z) * f


@dataclass(frozen=True)
class Stokes2D(HydrodynamicModel):
    """Independent infinitely long transverse fluid sections.

    With exp(+i omega t), positive p is resisting traction: fluid force is -p.
    The section mobility B satisfies v = B p, B = A/mu, using the old F2D
    panel-integrated Kelvin kernel. No thickness, walls, or longitudinal
    fluid interactions are included. Frequency must be strictly positive.
    """

    fluid: Fluid
    grid: FluidGrid
    solver: LinearSolver | None = None

    def __post_init__(self):
        if not np.isfinite([
            self.fluid.density, self.fluid.dynamic_viscosity
        ]).all() or min(self.fluid.density, self.fluid.dynamic_viscosity) <= 0:
            raise ValueError("Fluid density and dynamic viscosity must be positive.")
        object.__setattr__(
            self, "solver", SciPyLUSolver() if self.solver is None else self.solver,
        )
        object.__setattr__(self, "_factorized_omega", None)

    def section_mobility(self, omega: float):
        """Return the ny-by-ny complex mobility B, in m/(Pa s)."""
        if not np.isfinite(omega) or omega <= 0:
            raise ValueError("omega must be finite and strictly positive.")
        nu = self.fluid.kinematic_viscosity
        scale = np.sqrt(omega / nu)
        upper = scale * (self.grid.panel_edges[None, 1:] - self.grid.y[:, None])
        lower = scale * (self.grid.panel_edges[None, :-1] - self.grid.y[:, None])
        return (
            (_kelvin_primitive(upper) - _kelvin_primitive(lower))
            / (2j * np.pi * scale * self.fluid.dynamic_viscosity)
        )

    def pressure_from_velocity(self, omega: float, velocity: Any):
        """Solve B p = v for vector (nx*ny,) or batch (nx*ny, nrhs).

        One section LU factorization is reused for all x sections and RHSs.
        Pressure has the same shape as velocity, in Pa.
        """
        v = np.asarray(velocity, dtype=np.complex128)
        n = self.grid.nx * self.grid.ny
        if v.ndim not in (1, 2) or v.shape[0] != n or not np.isfinite(v).all():
            raise ValueError("velocity must be finite with shape (nx*ny[, nrhs]).")
        if self._factorized_omega != omega:
            self.solver.factorize(self.section_mobility(omega))
            object.__setattr__(self, "_factorized_omega", omega)
        shape = v.shape
        rhs = v.reshape(self.grid.nx, self.grid.ny, -1).transpose(1, 0, 2)
        p = self.solver.solve(rhs.reshape(self.grid.ny, -1))
        return (
            p.reshape(self.grid.ny, self.grid.nx, -1)
            .transpose(1, 0, 2).reshape(shape)
        )

    def assemble_matrix(self, omega: float):
        """Optional sparse block mobility B such that v = B p."""
        return sparse.kron(
            sparse.eye(self.grid.nx, format="csr"),
            sparse.csr_matrix(self.section_mobility(omega)), format="csr",
        )
