"""Complex driven response with exp(+i omega t) and full FE displacement."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres

from mufsi.coupling.operator import CouplingOperator
from mufsi.coupling.weighted import WeightedCouplingOperator
from mufsi.hydrodynamics.legacy.stokes_3d import Stokes3D as PanelStokes3D
from mufsi.hydrodynamics.stokes_2d import Stokes2D
from mufsi.hydrodynamics.stokes_3d import Stokes3D
from mufsi.solvers.linear import SciPyLUSolver


@dataclass(frozen=True)
class FrequencyResponseResult:
    """Frequency-first complex arrays in full structural/grid order.

    Frequencies are Hz, displacement metres, pressure Pa (resisting traction).
    relative_errors measures equilibrium against the applied force; fluid_errors
    measures the no-slip residual against velocity (NaN for action-only models).
    Weighted 3D also returns coefficient phasors [Pa], their pressure evaluation
    points [m], integrated resisting FE forces [N], and force-projection error.
    """

    frequencies: np.ndarray
    displacement: np.ndarray
    pressure: np.ndarray | None = None
    relative_errors: np.ndarray | None = None
    fluid_errors: np.ndarray | None = None
    pressure_coefficients: np.ndarray | None = None
    pressure_points: np.ndarray | None = None
    fluid_force: np.ndarray | None = None
    force_projection_error: float | None = None


def _csr(matrix):
    """Copy a serial PETSc matrix into SciPy; destroy the caller-owned object."""
    try:
        indptr, indices, data = matrix.getValuesCSR()
        return sparse.csr_matrix(
            (data.copy(), indices.copy(), indptr.copy()),
            shape=matrix.getSize(),
        )
    finally:
        matrix.destroy()


def _mixed_block_solve(D, G, E, B, F, omega, k_scale):
    """Scaled joint displacement/pressure solve, also valid at vacuum poles."""
    tiny = np.finfo(float).tiny
    b_scale = max(np.max(np.asarray(np.abs(B).sum(axis=1))), tiny)
    p_per_u = omega / b_scale
    system = sparse.bmat(
        [
            [D / k_scale, G * (p_per_u / k_scale)],
            [-1j * E, sparse.csr_matrix(B) * (p_per_u / omega)],
        ],
        format="csc",
    )
    solver = SciPyLUSolver()
    solver.factorize(system)
    rhs = np.concatenate((F / k_scale, np.zeros(B.shape[0])))
    z = solver.solve(rhs)
    for _ in range(2):
        z += solver.solve(rhs - system @ z)
    return z[: len(F)], z[len(F) :] * p_per_u


def _fluid_schur_solve(D, G, E, B, F, omega, k_scale, batch_size=64):
    """Exact pressure Schur solve; only bounded batches of structural RHSs.

    (B + i*omega*E*D^-1*G) p = i*omega*E*D^-1*F.
    The full displacement is recovered without modal truncation. Falls back
    to a joint solve if D cannot be factored at a vacuum resonance.
    """
    structural = SciPyLUSolver()
    try:
        structural.factorize(D / k_scale)
    except RuntimeError:
        return _mixed_block_solve(D, G, E, B, F, omega, k_scale)
    b_scale = max(np.max(np.abs(B).sum(axis=1)), np.finfo(float).tiny)
    system = np.array(B / b_scale, dtype=complex, copy=True)
    for start in range(0, B.shape[1], batch_size):
        stop = min(start + batch_size, B.shape[1])
        influence = structural.solve(G[:, start:stop].toarray() / k_scale)
        system[:, start:stop] += (1j * omega / b_scale) * (E @ influence)
    pressure_solver = SciPyLUSolver()
    pressure_solver.factorize(system)
    dry = structural.solve(F / k_scale)
    p = pressure_solver.solve((1j * omega / b_scale) * (E @ dry))
    u = structural.solve((F - G @ p) / k_scale)
    for _ in range(2):
        r_force = F - D @ u - G @ p
        r_velocity = 1j * omega * (E @ u) - B @ p
        correction = structural.solve(r_force / k_scale)
        dp = pressure_solver.solve(
            (r_velocity + 1j * omega * (E @ correction)) / b_scale,
        )
        u += structural.solve((r_force - G @ dp) / k_scale)
        p += dp
    return u, p


@dataclass
class FrequencyResponseSolver:
    """Eliminate fixed DOFs and solve each complex frequency system.

    F2D uses a sparse displacement/pressure block system, avoiding a dense
    structural impedance or a global inverse. Weighted F3D uses a coefficient
    Schur solve with independently integrated pressure forces. Legacy panel
    F3D uses a pressure Schur solve. Both retain full FE displacement.
    Other hydrodynamic models use
    pressure_from_velocity through a matrix-free Schur operator. Initial
    coupled solves require one MPI rank; structural eigen solves support MPI.
    """

    problem: Any

    def solve(
        self,
        frequencies,
        load,
        *,
        progress: Callable[[int, int], None] | None = None,
    ):
        f = np.atleast_1d(np.array(frequencies, dtype=float, copy=True))
        if f.ndim != 1 or f.size == 0 or not np.isfinite(f).all() or np.any(f <= 0):
            raise ValueError("frequencies must be a nonempty positive finite 1D array.")
        structure, hydro = self.problem.structure, self.problem.hydrodynamics
        if structure.mesh.comm.size != 1:
            raise NotImplementedError(
                "Coupled frequency response requires one MPI rank."
            )
        if isinstance(hydro, Stokes3D):
            return self._solve_weighted(f, load, progress=progress)
        coupling = self.problem.coupling
        if coupling is None:
            coupling = CouplingOperator.from_structure(structure, hydro.grid)
            self.problem.coupling = coupling
        K, M = _csr(structure.stiffness_matrix()), _csr(structure.mass_matrix())
        force = structure.force_vector(load)
        try:
            F = force.getArray(readonly=True).copy().astype(complex)
        finally:
            force.destroy()
        ndofs = K.shape[0]
        E = coupling.evaluation_matrix
        if E.shape != (len(hydro.grid.points), ndofs) or not np.array_equal(
            coupling.weights, hydro.grid.weights
        ):
            raise ValueError(
                "Coupling dimensions and weights must match structure/grid."
            )
        free = np.setdiff1d(np.arange(ndofs), structure.constrained_dofs)
        E = E[:, free].tocsr()
        G = (E.T @ sparse.diags(coupling.weights)).tocsr()
        K, M, F = K[free][:, free], M[free][:, free], F[free]
        nf = len(hydro.grid.points)
        displacement = np.zeros((len(f), ndofs), dtype=complex)
        pressure = np.zeros((len(f), nf), dtype=complex)
        errors, fluid_errors = np.zeros(len(f)), np.full(len(f), np.nan)
        # Normalize displacement and pressure units to balance the block system.
        force_scale = np.linalg.norm(F)
        k_scale = max(np.max(np.abs(K.diagonal())), np.finfo(float).tiny)
        tiny = np.finfo(float).tiny
        for i, hz in enumerate(f):
            omega = 2 * np.pi * hz
            D = K - omega**2 * M
            if isinstance(hydro, (Stokes2D, PanelStokes3D)):
                B = hydro.assemble_matrix(omega)
                if isinstance(hydro, PanelStokes3D):
                    u, p = _fluid_schur_solve(D, G, E, B, F, omega, k_scale)
                else:
                    u, p = _mixed_block_solve(D, G, E, B, F, omega, k_scale)
                v = 1j * omega * (E @ u)
                fluid_residual = B @ p - v
            else:

                def action(u, D=D, omega=omega):
                    return D @ u + G @ hydro.pressure_from_velocity(
                        omega,
                        1j * omega * (E @ u),
                    )

                pre = SciPyLUSolver()
                pre.factorize(D)
                operator = LinearOperator(D.shape, matvec=action, dtype=complex)
                preconditioner = LinearOperator(
                    D.shape,
                    matvec=pre.solve,
                    dtype=complex,
                )
                u, info = gmres(
                    operator,
                    F,
                    M=preconditioner,
                    rtol=1e-9,
                    atol=0,
                    restart=80,
                    maxiter=300,
                )
                if info != 0:
                    raise RuntimeError(
                        f"Coupled iterative solve failed at {hz:g} Hz: {info}."
                    )
                p = hydro.pressure_from_velocity(omega, 1j * omega * (E @ u))
                fluid_residual = None
            errors[i] = np.linalg.norm(D @ u + G @ p - F) / max(force_scale, tiny)
            if fluid_residual is not None:
                fluid_errors[i] = np.linalg.norm(fluid_residual) / max(
                    np.linalg.norm(v),
                    tiny,
                )
            if not np.isfinite(u).all() or not np.isfinite(p).all():
                raise RuntimeError(f"Non-finite coupled solution at {hz:g} Hz.")
            displacement[i, free], pressure[i] = u, p
            if progress is not None:
                progress(i + 1, len(f))
        return FrequencyResponseResult(f, displacement, pressure, errors, fluid_errors)

    def _solve_weighted(self, frequencies, load, *, progress=None):
        """Full FE response with a small pressure-coefficient Schur system.

        a = i omega L E u, D u + C a = F, where L is the least-squares
        action of rectangular H. The joint fallback handles dry poles.
        No-slip residual reports finite-basis error, not just linear-solve error.
        """
        structure, hydro = self.problem.structure, self.problem.hydrodynamics
        coupling = self.problem.coupling
        if coupling is None:
            coupling = WeightedCouplingOperator.from_structure(structure, hydro)
            self.problem.coupling = coupling
        if not isinstance(coupling, WeightedCouplingOperator):
            raise TypeError("Stokes3D requires WeightedCouplingOperator.")
        K, M = _csr(structure.stiffness_matrix()), _csr(structure.mass_matrix())
        vector = structure.force_vector(load)
        try:
            force = vector.getArray(readonly=True).copy().astype(complex)
        finally:
            vector.destroy()
        ndofs = len(force)
        if coupling.evaluation_matrix.shape != (len(hydro.collocation_points), ndofs):
            raise ValueError("Collocation evaluation dimensions do not match.")
        if coupling.force_projection.shape != (ndofs, hydro.coefficient_count):
            raise ValueError("Pressure force projection dimensions do not match.")
        free = np.setdiff1d(np.arange(ndofs), structure.constrained_dofs)
        E = coupling.evaluation_matrix[:, free].tocsr()
        C = sparse.csr_matrix(coupling.force_projection[free])
        K, M, F = K[free][:, free], M[free][:, free], force[free]
        count = hydro.coefficient_count
        displacement = np.zeros((len(frequencies), ndofs), complex)
        coefficients = np.zeros((len(frequencies), count), complex)
        pressure = np.zeros((len(frequencies), len(hydro.collocation_points)), complex)
        fluid_force = np.zeros_like(displacement)
        errors, fluid_errors = np.zeros(len(frequencies)), np.zeros(len(frequencies))
        scale = max(np.max(abs(K.diagonal())), np.finfo(float).tiny)
        tiny = np.finfo(float).tiny
        for i, hz in enumerate(frequencies):
            omega = 2 * np.pi * hz
            H = hydro.assemble_matrix(omega)
            LE = sparse.csr_matrix(hydro.coefficient_action(omega) @ E)
            D = K - omega**2 * M
            u, a = _fluid_schur_solve(D, C, LE, np.eye(count), F, omega, scale)
            velocity = 1j * omega * (E @ u)
            errors[i] = np.linalg.norm(D @ u + C @ a - F) / max(np.linalg.norm(F), tiny)
            fluid_errors[i] = np.linalg.norm(H @ a - velocity) / max(
                np.linalg.norm(velocity), tiny
            )
            if not np.isfinite(u).all() or not np.isfinite(a).all():
                raise RuntimeError(f"Non-finite weighted response at {hz:g} Hz.")
            displacement[i, free], coefficients[i] = u, a
            pressure[i] = hydro.pressure_from_coefficients(a)
            fluid_force[i] = coupling.to_structure(a)
            if progress is not None:
                progress(i + 1, len(frequencies))
        return FrequencyResponseResult(
            frequencies,
            displacement,
            pressure,
            errors,
            fluid_errors,
            coefficients,
            hydro.collocation_points,
            fluid_force,
            coupling.projection_error,
        )
