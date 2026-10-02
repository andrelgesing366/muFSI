"""Interface for structural discretizations.

Concrete models construct DOLFINx objects lazily, never at import. Backend
object annotations use Any so the interface imports without the FEM runtime.
"""

from abc import ABC, abstractmethod
from typing import Any

from mufsi.structure.loads import Load


class StructuralModel(ABC):
    """Expose structural operators without assembling the coupled FSI problem."""

    @property
    def mesh(self) -> Any:
        """Return the concrete model's structural mesh."""
        raise NotImplementedError("Structural models must provide a mesh.")

    @property
    def function_space(self) -> Any:
        """Return the FEM space used for structural displacement."""
        raise NotImplementedError("Structural models must provide a function space.")

    @property
    def constrained_dofs(self) -> Any:
        """Local zero-displacement DOFs, including ghosts.

        Matrices use the full DOF space; solvers eliminate constraints from
        both operators rather than introducing arbitrary diagonal entries.
        """
        raise NotImplementedError("The model must identify constrained DOFs.")

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
