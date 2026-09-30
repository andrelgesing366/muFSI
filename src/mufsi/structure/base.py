"""Interface for structural discretizations.

Array and matrix annotations use Any until the numerical backends are chosen.
DOLFINx objects will be created by concrete implementations, never at import.
"""

from abc import ABC, abstractmethod
from typing import Any

from mufsi.structure.loads import Load


class StructuralModel(ABC):
    """Expose structural operators without assembling the coupled FSI problem."""

    @property
    def mesh(self) -> Any:
        """Return the structural mesh once mesh construction is implemented."""
        raise NotImplementedError("Structural mesh construction is pending.")

    @property
    def function_space(self) -> Any:
        """Return the FEM space used for structural displacement."""
        raise NotImplementedError("Structural function-space setup is pending.")

    @abstractmethod
    def stiffness_matrix(self) -> Any:
        """Assemble the structural stiffness matrix K."""
        raise NotImplementedError

    @abstractmethod
    def mass_matrix(self) -> Any:
        """Assemble the structural mass matrix M."""
        raise NotImplementedError

    @abstractmethod
    def force_vector(self, load: Load) -> Any:
        """Assemble a load vector in structural DOF order."""
        raise NotImplementedError
