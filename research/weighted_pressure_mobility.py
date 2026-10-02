"""Research compatibility imports; numerical implementation lives in mufsi."""

from pathlib import Path

from mufsi.hydrodynamics.weighted_pressure import (
    DEFAULT_K,
    DEFAULT_M,
    Fluid,
    IntegrationConvergenceError,
    PlateGeometry,
    Quadrature,
    WeightedBasis,
    WeightedMobility,
    _integer,
    line_rule,
    solve_coefficients,
)

ROOT = Path(__file__).resolve().parents[1]

__all__ = [
    "DEFAULT_K",
    "DEFAULT_M",
    "ROOT",
    "Fluid",
    "IntegrationConvergenceError",
    "PlateGeometry",
    "Quadrature",
    "WeightedBasis",
    "WeightedMobility",
    "_integer",
    "line_rule",
    "solve_coefficients",
]
