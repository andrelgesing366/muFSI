"""Linear solver interfaces shared by numerical models.

TODO: implement SciPy LU and PETSc backends with repeated/blocked right-hand
sides. Import backend libraries only when the corresponding backend is used.
"""

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
    """Placeholder for a reusable SciPy LU factorization."""

    def factorize(self, matrix: Any) -> None:
        """Prepare an LU factorization for subsequent solves."""
        raise NotImplementedError("SciPy LU factorization is pending.")

    def solve(self, rhs: Any) -> Any:
        """Solve using the stored LU factors."""
        raise NotImplementedError("The SciPy linear solve is pending.")


class PETScSolver(LinearSolver):
    """Placeholder for a configurable PETSc direct or iterative solver."""

    def factorize(self, matrix: Any) -> None:
        """Prepare PETSc operators and solver state."""
        raise NotImplementedError("PETSc solver setup is pending.")

    def solve(self, rhs: Any) -> Any:
        """Solve using the configured PETSc backend."""
        raise NotImplementedError("The PETSc linear solve is pending.")
