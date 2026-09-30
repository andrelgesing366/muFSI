"""Unsteady Stokeslet primitives; no FEM dependencies belong in this module."""

from typing import Any

from mufsi.models.fluid import Fluid


def unsteady_stokeslet_zz(separation: Any, omega: float, fluid: Fluid) -> Any:
    """Evaluate the transverse kernel at separation vectors in metres.

    TODO: establish the harmonic convention, zero-frequency limit, and singular
    handling before implementing the expression from the reference code.
    """
    raise NotImplementedError("The unsteady Stokeslet kernel is pending.")
