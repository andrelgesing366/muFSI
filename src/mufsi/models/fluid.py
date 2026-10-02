"""Fluid parameters in SI units."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fluid:
    """Newtonian fluid data; input validation is pending."""

    density: float  # kg/m^3
    dynamic_viscosity: float  # Pa s

    @property
    def kinematic_viscosity(self) -> float:
        """Return dynamic viscosity divided by density, in m^2/s."""
        return self.dynamic_viscosity / self.density
