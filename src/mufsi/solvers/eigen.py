"""SLEPc solution of the in-vacuo generalized problem K u = omega^2 M u."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from numbers import Integral
from typing import Any

from mufsi.structure.base import StructuralModel


@dataclass(frozen=True)
class EigenResult:
    """Mass-normalized structural modes and ascending eigenfrequencies.

    modes has shape (local DOFs including ghosts, n_modes), in DOLFINx order.
    In serial this is the full mode matrix. Under MPI it is rank-local, while
    frequencies/errors are replicated. mode_functions contain the same fields
    as DOLFINx Functions with synchronized ghost values.

    relative_errors = ||K u - lambda M u|| / (||K u|| + |lambda| ||M u||).
    """

    angular_frequencies: Any
    modes: Any
    eigenvalues: Any
    relative_errors: Any
    mode_functions: tuple[Any, ...]
    iterations: int

    @property
    def frequencies(self) -> Any:
        """Natural frequencies in Hz."""
        return self.angular_frequencies / (2 * pi)


@dataclass(frozen=True)
class EigenSolver:
    """Find the lowest modes with Krylov–Schur and zero-shift inversion.

    Both matrices are restricted to unconstrained displacement DOFs first.
    factor_solver_type can select an installed backend, e.g. MUMPS for MPI LU;
    otherwise PETSc chooses its available default.
    """

    structure: StructuralModel
    tolerance: float = 1e-10
    max_iterations: int = 1000
    factor_solver_type: str | None = None
    options_prefix: str = "mufsi_eigen_"

    def __post_init__(self) -> None:
        if not isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive.")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, Integral)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer.")

    def solve(self, n_modes: int) -> EigenResult:
        """Solve collectively on the structural mesh communicator.

        Failure to converge or a non-positive/complex eigenvalue is reported
        explicitly instead of taking the absolute value of an invalid result.
        Temporary PETSc/SLEPc objects are destroyed before returning.
        """
        if (
            isinstance(n_modes, bool)
            or not isinstance(n_modes, Integral)
            or n_modes < 1
        ):
            raise ValueError("n_modes must be a positive integer.")
        try:
            import numpy as np
            from dolfinx import fem
            from mpi4py import MPI
            from petsc4py import PETSc
            from slepc4py import SLEPc
        except ImportError as error:
            raise ImportError(
                "EigenSolver requires compatible DOLFINx, PETSc, and SLEPc Python "
                "bindings. See docs/plate_eigenproblem.md."
            ) from error

        V = self.structure.function_space
        comm = V.mesh.comm
        if V.dofmap.index_map_bs != 1:
            raise ValueError("The initial eigen solver requires a scalar space.")
        n_owned = V.dofmap.index_map.size_local
        free_mask = np.ones(n_owned, dtype=bool)
        constrained = np.asarray(self.structure.constrained_dofs)
        free_mask[constrained[constrained < n_owned]] = False
        free_local = np.flatnonzero(free_mask)
        n_free = comm.allreduce(free_local.size, op=MPI.SUM)
        if n_modes >= n_free:
            raise ValueError(f"n_modes must be smaller than the {n_free} free DOFs.")

        objects: list[Any] = []
        try:
            K = self.structure.stiffness_matrix()
            objects.append(K)
            M = self.structure.mass_matrix()
            objects.append(M)
            start, stop = K.getOwnershipRange()
            if stop - start != n_owned or M.getOwnershipRange() != (start, stop):
                raise ValueError("Matrix ownership must match the structural DOF map.")
            free_global = (start + free_local).astype(PETSc.IntType)
            free_is = PETSc.IS().createGeneral(free_global, comm=comm)
            objects.append(free_is)
            K_free = K.createSubMatrix(free_is, free_is)
            objects.append(K_free)
            M_free = M.createSubMatrix(free_is, free_is)
            objects.append(M_free)

            # Common scaling preserves the spectrum and prevents tiny physical
            # mass entries from producing misleading convergence estimates.
            mass_norm = M_free.norm(PETSc.NormType.INFINITY)
            if not np.isfinite(mass_norm) or mass_norm <= 0:
                raise RuntimeError("The reduced mass matrix must have positive norm.")
            pencil_scale = 1.0 / mass_norm
            K_free.scale(pencil_scale)
            M_free.scale(pencil_scale)

            eps = SLEPc.EPS().create(comm=comm)
            objects.append(eps)
            eps.setOptionsPrefix(self.options_prefix)
            eps.setOperators(K_free, M_free)
            eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
            eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
            eps.setDimensions(nev=int(n_modes))
            eps.setTarget(0.0)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.setTolerances(self.tolerance, int(self.max_iterations))
            st = eps.getST()
            st.setType(SLEPc.ST.Type.SINVERT)
            ksp = st.getKSP()
            ksp.setType(PETSc.KSP.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.LU)
            if self.factor_solver_type is not None:
                pc.setFactorSolverType(self.factor_solver_type)
            eps.setFromOptions()
            eps.solve()
            if eps.getConvergedReason() <= 0 or eps.getConverged() < n_modes:
                raise RuntimeError(
                    f"SLEPc converged {eps.getConverged()} of {n_modes} modes "
                    f"after {eps.getIterationNumber()} iterations "
                    f"(reason {eps.getConvergedReason()})."
                )

            real = K_free.createVecRight()
            imag = real.duplicate()
            weighted = real.duplicate()
            stiffness_action = real.duplicate()
            residual = real.duplicate()
            objects.extend((real, imag, weighted, stiffness_action, residual))
            eigenvalues, errors, functions = [], [], []
            for i in range(n_modes):
                value = complex(eps.getEigenpair(i, real, imag))
                if (
                    not np.isfinite(value)
                    or abs(value.imag) > self.tolerance * max(1.0, abs(value.real))
                    or value.real <= 0
                ):
                    raise RuntimeError(
                        f"Invalid structural eigenvalue {value}; check the formulation, "
                        "constraints, and penalty before interpreting a frequency."
                    )
                M_free.mult(real, weighted)
                norm_squared = complex(real.dot(weighted)) / pencil_scale
                if norm_squared.real <= 0 or abs(norm_squared.imag) > (
                    self.tolerance * norm_squared.real
                ):
                    raise RuntimeError("The modal mass must be real and positive.")
                real.scale(1.0 / np.sqrt(norm_squared.real))
                local_values = real.getArray(readonly=True)
                if local_values.size != free_local.size:
                    raise RuntimeError("Reduced eigenvector ownership is inconsistent.")
                # Fix an arbitrary sign/phase using the largest global coefficient.
                if local_values.size:
                    j = int(np.argmax(np.abs(local_values)))
                    pivot = (
                        float(abs(local_values[j])), int(free_global[j]), local_values[j]
                    )
                else:
                    pivot = (0.0, 0, 0.0)
                _, _, coefficient = max(
                    comm.allgather(pivot), key=lambda item: (item[0], -item[1])
                )
                phase = np.conj(coefficient) / abs(coefficient)
                function = fem.Function(V, name=f"mode_{i + 1}")
                function.x.array[free_local] = local_values * phase
                function.x.scatter_forward()
                functions.append(function)
                eigenvalues.append(value.real)
                K_free.mult(real, stiffness_action)
                M_free.mult(real, weighted)
                stiffness_action.copy(residual)
                residual.axpy(-value.real, weighted)
                denominator = stiffness_action.norm() + value.real * weighted.norm()
                errors.append(residual.norm() / denominator)

            order = np.argsort(eigenvalues)
            values = np.asarray(eigenvalues)[order]
            mode_functions = tuple(functions[i] for i in order)
            for i, function in enumerate(mode_functions):
                function.name = f"mode_{i + 1}"
            return EigenResult(
                angular_frequencies=np.sqrt(values),
                modes=np.column_stack([f.x.array.copy() for f in mode_functions]),
                eigenvalues=values,
                relative_errors=np.asarray(errors)[order],
                mode_functions=mode_functions,
                iterations=eps.getIterationNumber(),
            )
        finally:
            for obj in reversed(objects):
                obj.destroy()
