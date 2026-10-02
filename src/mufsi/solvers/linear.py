"""Reusable dense and sparse SciPy LU behind a common solver interface."""

from abc import ABC, abstractmethod
from typing import Any


class LinearSolver(ABC):
    """Separate reusable linear algebra from the physical formulation."""

    @abstractmethod
    def factorize(self, matrix: Any) -> None:
        """Prepare a system for repeated solves without forming an inverse."""
        raise NotImplementedError

    @abstractmethod
    def solve(self, rhs: Any) -> Any:
        """Solve the prepared system for one or several right-hand sides."""
        raise NotImplementedError


class SciPyLUSolver(LinearSolver):
    """Reusable dense or sparse SciPy LU, including complex systems."""

    def factorize(self, matrix: Any) -> None:
        """Prepare an LU factorization for subsequent solves."""
        import numpy as np
        from scipy import linalg, sparse
        from scipy.sparse.linalg import splu

        self._sparse = sparse.issparse(matrix)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("LU requires a square matrix.")
        if self._sparse:
            self._factors = splu(matrix.astype(np.complex128).tocsc())
        else:
            a = np.asarray(matrix, dtype=np.complex128)
            if not np.isfinite(a).all():
                raise ValueError("LU matrix must be finite.")
            self._factors = linalg.lu_factor(a)

    def solve(self, rhs: Any) -> Any:
        """Solve using the stored LU factors."""
        from scipy.linalg import lu_solve

        if not hasattr(self, "_factors"):
            raise RuntimeError("Call factorize before solve.")
        return (
            self._factors.solve(rhs) if self._sparse
            else lu_solve(self._factors, rhs)
        )
