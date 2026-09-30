"""Recover fluid fields after a pressure solution has been obtained."""

from typing import Any

from mufsi.hydrodynamics.base import HydrodynamicModel


def reconstruct_flow(
    omega: float, points: Any, pressure: Any, hydrodynamics: HydrodynamicModel
) -> Any:
    """Recover fluid velocity at physical observation points.

    TODO: define the required tensor kernels and singular-point handling.
    """
    raise NotImplementedError("Fluid-flow reconstruction is pending.")
