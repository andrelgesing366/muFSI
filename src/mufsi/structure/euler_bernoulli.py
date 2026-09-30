"""Euler–Bernoulli beam skeleton for the DOLFINx structural backend."""

from dataclasses import dataclass
from typing import Any

from mufsi.models.geometry import BeamGeometry
from mufsi.models.material import Material
from mufsi.structure.base import StructuralModel
from mufsi.structure.loads import Load


@dataclass
class EulerBernoulliBeam(StructuralModel):
    """Store beam setup; FEM construction and assembly are pending."""

    geometry: BeamGeometry
    material: Material
    mesh_resolution: int = 64
    boundary_condition: str = "cantilever"
    element_degree: int = 2

    def stiffness_matrix(self) -> Any:
        """Assemble beam bending stiffness."""
        raise NotImplementedError("Euler–Bernoulli stiffness assembly is pending.")

    def mass_matrix(self) -> Any:
        """Assemble the beam mass matrix."""
        raise NotImplementedError("Euler–Bernoulli mass assembly is pending.")

    def force_vector(self, load: Load) -> Any:
        """Assemble distributed or point beam loads."""
        raise NotImplementedError("Euler–Bernoulli load assembly is pending.")
