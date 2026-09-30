"""Complex driven response with exp(+i omega t) and full FE displacement."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres

from mufsi.coupling.operator import CouplingOperator
from mufsi.hydrodynamics.stokes_2d import Stokes2D
from mufsi.solvers.linear import SciPyLUSolver


@dataclass(frozen=True)
class FrequencyResponseResult:
    """Frequency-first complex arrays in full structural/grid order.

    Frequencies are Hz, displacement metres, pressure Pa (resisting traction).
    relative_errors measures equilibrium against the applied force; fluid_errors
    measures the F2D no-slip residual against velocity (NaN for action-only models).
    """

    frequencies: np.ndarray
    displacement: np.ndarray
    pressure: np.ndarray | None = None
    relative_errors: np.ndarray | None = None
    fluid_errors: np.ndarray | None = None


def _csr(matrix):
    """Copy a serial PETSc matrix into SciPy; destroy the caller-owned object."""
    try:
        indptr, indices, data = matrix.getValuesCSR()
        return sparse.csr_matrix(
            (data.copy(), indices.copy(), indptr.copy()), shape=matrix.getSize(),
        )
    finally:
        matrix.destroy()


@dataclass
class FrequencyResponseSolver:
    """Eliminate fixed DOFs and solve each complex frequency system.

    F2D uses a sparse displacement/pressure block system, avoiding a dense
    structural impedance or a global inverse. Other hydrodynamic models use
    pressure_from_velocity through a matrix-free Schur operator. Initial
    coupled solves require one MPI rank; structural eigen solves support MPI.
    """

    problem: Any

    def solve(
        self, frequencies, load, *, progress: Callable[[int, int], None] | None = None,
    ):
        f = np.atleast_1d(np.array(frequencies, dtype=float, copy=True))
        if f.ndim != 1 or f.size == 0 or not np.isfinite(f).all() or np.any(f <= 0):
            raise ValueError("frequencies must be a nonempty positive finite 1D array.")
        structure, hydro = self.problem.structure, self.problem.hydrodynamics
        if structure.mesh.comm.size != 1:
            raise NotImplementedError(
                "Coupled frequency response requires one MPI rank."
            )
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
            if isinstance(hydro, Stokes2D):
                B = hydro.assemble_matrix(omega)
                b_scale = max(np.max(np.asarray(np.abs(B).sum(axis=1))), tiny)
                p_per_u = omega / b_scale
                row_scale = omega
                system = sparse.bmat([
                    [D / k_scale, G * (p_per_u / k_scale)],
                    [-1j * omega * E / row_scale, B * (p_per_u / row_scale)],
                ], format="csc")
                solver = SciPyLUSolver()
                solver.factorize(system)
                rhs = np.concatenate((F / k_scale, np.zeros(nf)))
                z = solver.solve(rhs)
                # Reuse LU for refinement of the high-contrast FE/fluid system.
                for _ in range(2):
                    z += solver.solve(rhs - system @ z)
                u, p = z[:len(free)], z[len(free):] * p_per_u
                v = 1j * omega * (E @ u)
                fluid_residual = B @ p - v
            else:
                def action(u):
                    return D @ u + G @ hydro.pressure_from_velocity(
                        omega, 1j * omega * (E @ u),
                    )

                pre = SciPyLUSolver()
                pre.factorize(D)
                operator = LinearOperator(D.shape, matvec=action, dtype=complex)
                preconditioner = LinearOperator(
                    D.shape, matvec=pre.solve, dtype=complex,
                )
                u, info = gmres(
                    operator, F, M=preconditioner, rtol=1e-9, atol=0,
                    restart=80, maxiter=300,
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
                    np.linalg.norm(v), tiny,
                )
            if not np.isfinite(u).all() or not np.isfinite(p).all():
                raise RuntimeError(f"Non-finite coupled solution at {hz:g} Hz.")
            displacement[i, free], pressure[i] = u, p
            if progress is not None:
                progress(i + 1, len(f))
        return FrequencyResponseResult(f, displacement, pressure, errors, fluid_errors)
