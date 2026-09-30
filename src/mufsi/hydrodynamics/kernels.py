"""Internal panel integration and operator application.

TODO: port the analytical 1D reduction, singular-panel treatment, geometric
cases, and uniform-grid symmetry. Keep temporary allocations bounded.
"""

from typing import Any

from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.models.fluid import Fluid


def _integrate_panel_zz(
    observation_point: Any, panel_bounds: Any, omega: float, fluid: Fluid
) -> complex:
    """Integrate the transverse Stokeslet over one fluid panel."""
    raise NotImplementedError("Analytically reduced panel integration is pending.")


def _apply_panel_operator(
    omega: float, pressure: Any, *, grid: FluidGrid, fluid: Fluid
) -> Any:
    """Apply the fluid panel operator without requiring a stored dense matrix."""
    raise NotImplementedError("Fluid panel-operator application is pending.")
