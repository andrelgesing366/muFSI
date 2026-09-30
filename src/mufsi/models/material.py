"""Structural material parameters in SI units."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Isotropic material data; validation and anisotropic models are pending."""

    young_modulus: float  # Pa
    density: float  # kg/m^3
    poisson_ratio: float  # Dimensionless
