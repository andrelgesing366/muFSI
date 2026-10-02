"""One-dimensional quadrature in physical SI coordinates."""

from numbers import Integral

import numpy as np


def _interval(start: float, stop: float, n_points: int) -> None:
    if not np.isfinite([start, stop]).all() or stop <= start:
        raise ValueError("The quadrature interval must be finite and increasing.")
    if isinstance(n_points, bool) or not isinstance(n_points, Integral) or n_points < 1:
        raise ValueError("n_points must be a positive integer.")


def midpoint_rule(start: float, stop: float, n_points: int):
    """Return midpoint nodes and weights for an ordinary integral."""
    _interval(start, stop, n_points)
    h = (stop - start) / n_points
    return start + h * (np.arange(n_points) + 0.5), np.full(n_points, h)


def chebyshev_gauss_nodes(start: float, stop: float, n_points: int):
    """Return ascending mapped Chebyshev–Gauss nodes."""
    _interval(start, stop, n_points)
    t = -np.cos((2 * np.arange(1, n_points + 1) - 1) * np.pi / (2 * n_points))
    return (start + stop) / 2 + (stop - start) / 2 * t


def simpson_rule(start: float, stop: float, n_points: int):
    """Composite Simpson rule, including endpoints; require an odd count >= 3."""
    _interval(start, stop, n_points)
    if n_points < 3 or n_points % 2 == 0:
        raise ValueError("Simpson quadrature requires an odd n_points >= 3.")
    nodes = np.linspace(start, stop, n_points)
    weights = np.ones(n_points)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    weights *= (stop - start) / (3 * (n_points - 1))
    return nodes, weights
