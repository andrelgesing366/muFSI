"""Quadrature rules and integration helpers for fluid panels.

TODO: specify node ordering, weight normalization, and endpoint treatment
against the reference implementation before adding numerical code.
"""

from typing import Any


def midpoint_rule(start: float, stop: float, n_points: int) -> tuple[Any, Any]:
    """Return midpoint nodes and ordinary integration weights on an interval."""
    raise NotImplementedError("Midpoint quadrature is pending.")


def chebyshev_gauss_nodes(start: float, stop: float, n_points: int) -> Any:
    """Return mapped Chebyshev–Gauss nodes; panel areas belong to FluidGrid."""
    raise NotImplementedError("Chebyshev–Gauss node generation is pending.")
