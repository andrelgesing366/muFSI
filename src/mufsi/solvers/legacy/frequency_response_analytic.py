"""Direct coupled response for the legacy analytic constant-panel model.

The legacy Stokes3DAnalytic model uses the shared scaled fluid Schur algebra
and result type, bounded structural RHS batches, and a joint-solve fallback
at an in-vacuo resonance. This entry point preserves its direct solve for
reference comparisons with the active weighted-pressure formulation.
"""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from scipy import sparse

from mufsi.coupling.operator import CouplingOperator
from mufsi.hydrodynamics.legacy.stokes_3d_analytic import Stokes3DAnalytic
from mufsi.solvers.frequency_response import (
    FrequencyResponseResult,
    _csr,
    _fluid_schur_solve,
)


@dataclass
class AnalyticFrequencyResponseSolver:
    """Full structural response with analytic fluid mobility, on one MPI rank.

    Accepts the same CoupledProblem and load objects as FrequencyResponseSolver.
    Call this class explicitly to select the dense fluid Schur solve. The generic
    CoupledProblem.frequency_response interface also accepts Stokes3DAnalytic,
    but currently routes it through its iterative operator path.
    """

    problem: Any
    structural_batch_size: int = 64

    def solve(
        self, frequencies, load, *, progress: Callable[[int, int], None] | None = None
    ):
        """Return displacement, pressure, and equilibrium/no-slip residuals.

        frequencies is in Hz. Rows in the returned arrays follow frequency
        order; displacement includes constrained DOFs, set to zero, and
        pressure follows the analytic grid's x-major order.
        """
        f = np.atleast_1d(np.array(frequencies, dtype=float, copy=True))
        if f.ndim != 1 or f.size == 0 or not np.isfinite(f).all() or np.any(f <= 0):
            raise ValueError("frequencies must be a nonempty positive finite 1D array.")
        batch = self.structural_batch_size
        if isinstance(batch, bool) or not isinstance(batch, Integral) or batch < 1:
            raise ValueError("structural_batch_size must be a positive integer.")
        structure, hydro = self.problem.structure, self.problem.hydrodynamics
        if not isinstance(hydro, Stokes3DAnalytic):
            raise TypeError(
                "AnalyticFrequencyResponseSolver requires Stokes3DAnalytic."
            )
        if structure.mesh.comm.size != 1:
            raise NotImplementedError(
                "Coupled analytic response requires one MPI rank."
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
            coupling.weights,
            hydro.grid.weights,
        ):
            raise ValueError(
                "Coupling dimensions and weights must match structure/grid."
            )
        free = np.setdiff1d(np.arange(ndofs), structure.constrained_dofs)
        if free.size == 0:
            raise ValueError(
                "The coupled problem requires unconstrained structural DOFs."
            )
        E = E[:, free].tocsr()
        G = (E.T @ sparse.diags(coupling.weights)).tocsr()
        K, M, F = K[free][:, free], M[free][:, free], F[free]
        nf = len(hydro.grid.points)
        displacement = np.zeros((len(f), ndofs), dtype=complex)
        pressure = np.zeros((len(f), nf), dtype=complex)
        force_errors, fluid_errors = np.empty(len(f)), np.empty(len(f))
        tiny = np.finfo(float).tiny
        force_scale = max(np.linalg.norm(F), tiny)
        k_scale = max(np.max(np.abs(K.diagonal())), tiny)
        for i, hz in enumerate(f):
            omega = 2 * np.pi * hz
            D = K - omega**2 * M
            B = hydro.assemble_matrix(omega)
            u, p = _fluid_schur_solve(
                D,
                G,
                E,
                B,
                F,
                omega,
                k_scale,
                batch_size=batch,
            )
            velocity = 1j * omega * (E @ u)
            force_errors[i] = np.linalg.norm(D @ u + G @ p - F) / force_scale
            fluid_errors[i] = np.linalg.norm(B @ p - velocity) / max(
                np.linalg.norm(velocity),
                tiny,
            )
            if not np.isfinite(u).all() or not np.isfinite(p).all():
                raise RuntimeError(
                    f"Non-finite coupled analytic solution at {hz:g} Hz."
                )
            displacement[i, free], pressure[i] = u, p
            if progress is not None:
                progress(i + 1, len(f))
        return FrequencyResponseResult(
            f, displacement, pressure, force_errors, fluid_errors
        )
