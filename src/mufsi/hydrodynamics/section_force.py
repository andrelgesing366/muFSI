"""Local beam loading from Sader or a rigid Tuck/F2D strip section."""

from dataclasses import dataclass
from functools import cached_property
from numbers import Integral

import numpy as np
from scipy.integrate import trapezoid

from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.hydrodynamics.sader import gamma_function
from mufsi.hydrodynamics.stokes_2d import Stokes2D
from mufsi.models.fluid import Fluid
from mufsi.models.geometry import BeamGeometry, PlateGeometry


@dataclass(frozen=True)
class SectionForce2D:
    """Section impedance Z: resisting line force = Z(omega)*velocity.

    Z has units N s/m^2; the actual fluid force on the beam is its negative.
    Sections move uniformly across their width, independently in x.
    'sader' uses the rectangular hydrodynamic function. 'tuck' solves the
    Kelvin-panel problem on cosine edges with midpoint collocation, matching
    the numeric-force helper referenced by the legacy beam main.py.

    'panel' integrates piecewise-constant pressure over full panel widths.
    'legacy_trapezoid' reproduces the old centre-to-centre trapezoid integral,
    which omits the edge half-panels. No explicit inverse or dummy velocity.
    """

    geometry: BeamGeometry | PlateGeometry
    fluid: Fluid
    method: str = "tuck"
    ny: int = 32
    integration: str = "panel"

    def __post_init__(self):
        parameters = (
            self.geometry.length,
            self.geometry.width,
            self.geometry.thickness,
            self.fluid.density,
            self.fluid.dynamic_viscosity,
        )
        if not np.isfinite(parameters).all() or min(parameters) <= 0:
            raise ValueError("Section geometry and fluid parameters must be positive.")
        if self.method not in {"sader", "tuck"}:
            raise ValueError("method must be 'sader' or 'tuck'.")
        if (
            isinstance(self.ny, bool)
            or not isinstance(self.ny, Integral)
            or self.ny < 2
        ):
            raise ValueError("ny must be an integer >= 2.")
        if self.integration not in {"panel", "legacy_trapezoid"}:
            raise ValueError("integration must be 'panel' or 'legacy_trapezoid'.")

    @staticmethod
    def _omega(omega):
        if np.ndim(omega) != 0 or not np.isfinite(omega) or omega <= 0:
            raise ValueError("omega must be a finite, strictly positive scalar.")

    @cached_property
    def section_grid(self):
        """One section with cosine-spaced edges and physical midpoint nodes."""
        length, width = self.geometry.length, self.geometry.width
        edges = width / 2 * np.cos(np.linspace(-np.pi, 0, self.ny + 1))
        y = (edges[:-1] + edges[1:]) / 2
        return FluidGrid(
            np.column_stack((np.full(self.ny, length / 2), y)),
            length * np.diff(edges),
            edges,
            1,
            self.ny,
            x_panel_edges=np.array([0.0, length]),
        )

    @cached_property
    def _section(self):
        return Stokes2D(self.fluid, self.section_grid)

    def section_pressure(self, omega):
        """Read-only Pa/(m/s) pressure for unit rigid velocity ('tuck' only)."""
        self._omega(omega)
        if self.method != "tuck":
            raise NotImplementedError(
                "Sader supplies an integrated force, not a pressure field."
            )
        if getattr(self, "_pressure_omega", None) != omega:
            pressure = self._section.pressure_from_velocity(omega, np.ones(self.ny))
            pressure.setflags(write=False)
            object.__setattr__(self, "_pressure", pressure)
            object.__setattr__(self, "_pressure_omega", omega)
        return self._pressure

    def line_impedance(self, omega):
        """Return resisting force per unit length / velocity in N s/m^2."""
        self._omega(omega)
        if self.method == "sader":
            re = omega * self.geometry.width**2 / (4 * self.fluid.kinematic_viscosity)
            gamma = gamma_function(re).conjugate()
            return (
                1j
                * omega
                * np.pi
                * self.fluid.density
                * self.geometry.width**2
                / 4
                * gamma
            )
        pressure = self.section_pressure(omega)
        if self.integration == "legacy_trapezoid":
            return trapezoid(pressure, x=self.section_grid.y)
        return np.dot(np.diff(self.section_grid.panel_edges), pressure)

    def dynamic_stiffness(self, omega):
        """Return i*omega*Z: resisting line force / displacement in N/m^2."""
        return 1j * omega * self.line_impedance(omega)

    def resisting_force_from_velocity(self, omega, velocity):
        """Return local resisting line force in N/m, preserving array shape."""
        velocity = np.asarray(velocity, dtype=complex)
        if not np.isfinite(velocity).all():
            raise ValueError("velocity must be finite.")
        return self.line_impedance(omega) * velocity
