"""Kirchhoff–Love plate skeleton for the DOLFINx C0 interior-penalty method."""

from dataclasses import dataclass
from typing import Any

from mufsi.models.geometry import PlateGeometry
from mufsi.models.material import Material
from mufsi.structure.base import StructuralModel
from mufsi.structure.loads import Load


@dataclass
class KirchhoffPlate(StructuralModel):
    """Store plate setup; FEM construction and assembly are pending.

    mesh_resolution denotes cell counts, while the inherited mesh property
    will expose the actual DOLFINx mesh. Boundary-condition names are provisional.
    """

    geometry: PlateGeometry
    material: Material
    mesh_resolution: tuple[int, int] = (64, 32)
    boundary_condition: str = "cantilever"
    element_degree: int = 2

    def stiffness_matrix(self) -> Any:
        """Assemble bending stiffness and interior-penalty terms."""
        raise NotImplementedError("Kirchhoff plate stiffness assembly is pending.")

    def mass_matrix(self) -> Any:
        """Assemble the plate mass matrix."""
        raise NotImplementedError("Kirchhoff plate mass assembly is pending.")

    def force_vector(self, load: Load) -> Any:
        """Assemble distributed or point loads with consistent DOF ordering."""
        raise NotImplementedError("Kirchhoff plate load assembly is pending.")
