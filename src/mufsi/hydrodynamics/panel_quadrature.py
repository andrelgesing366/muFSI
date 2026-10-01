"""Tolerance-controlled rectangle cubature for the original F3D method."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

from mufsi.hydrodynamics.stokeslet import _kernel_zz, _radial_integral


class QuadratureConvergenceError(RuntimeError):
    """Requested panel tolerance was not reached within the refinement limits."""


@dataclass(frozen=True)
class IntegrationReport:
    omega: float
    quadrature_backend: str
    evaluated_rows: int
    refined_entries: int
    max_relative_error: float
    max_error_ratio: float
    max_refinement: int


@lru_cache(maxsize=16)
def _rectangle_rule(backend, degree):
    if backend == "quadpy":
        try:
            import quadpy

            scheme = quadpy.c2.get_good_scheme(degree)
        except (ImportError, RuntimeError) as error:
            raise ImportError(
                "Quadpy cubature is unavailable. Install a working Quadpy runtime "
                "or legacy-quadpy, or select quadrature_backend='gauss'. "
                "See docs/f3d_spectrum.md."
            ) from error
        return np.asarray(scheme.points.T), 4 * np.asarray(scheme.weights)
    # Tensor Gauss rule integrates polynomials through at least this degree.
    x, w = leggauss((degree + 2) // 2)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return np.column_stack((xx.ravel(), yy.ravel())), np.outer(w, w).ravel()


@lru_cache(maxsize=32)
def _line_rule(backend, order):
    if backend == "quadpy":
        import quadpy

        scheme = quadpy.c1.gauss_legendre(order)
        return np.asarray(scheme.points), np.asarray(scheme.weights)
    return leggauss(order)


class PanelIntegrator:
    """Raise rule degree 2/4/6/8, then adaptively bisect the longer panel side.

    Relative tolerance controls each complex integral estimate, with an absolute
    tolerance in metres before dividing by mu. The estimate is not a rigorous
    error bound or a fluid-grid convergence criterion. No silent truncation.
    """

    def __init__(
        self,
        backend="quadpy",
        rtol=2e-3,
        atol=1e-15,
        max_refinements=12,
        max_subpanels=16384,
        batch_size=256,
    ):
        if backend not in {"quadpy", "gauss"}:
            raise ValueError("quadrature_backend must be 'quadpy' or 'gauss'.")
        if not np.isfinite([rtol, atol]).all() or rtol <= 0 or atol < 0:
            raise ValueError("rtol must be positive and atol nonnegative.")
        for name, value, minimum in (
            ("max_refinements", max_refinements, 0),
            ("max_subpanels", max_subpanels, 2),
            ("batch_size", batch_size, 1),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        self.backend, self.rtol, self.atol = backend, rtol, atol
        self.max_refinements, self.max_subpanels = max_refinements, max_subpanels
        self.batch_size = batch_size
        _rectangle_rule(backend, 2)  # Fail early on a missing/licensed backend.

    def rectangles(self, observation, bounds, lam, degree):
        """Vectorized cubature of nonsingular panels (x0,x1,y0,y1)."""
        nodes, weights = _rectangle_rule(self.backend, degree)
        result = np.empty(len(bounds), dtype=complex)
        for start in range(0, len(bounds), self.batch_size):
            b = bounds[start : start + self.batch_size]
            hx, hy = (b[:, 1] - b[:, 0]) / 2, (b[:, 3] - b[:, 2]) / 2
            dx = (
                (b[:, 0] + b[:, 1])[:, None] / 2
                + hx[:, None] * nodes[:, 0]
                - observation[0]
            )
            dy = (
                (b[:, 2] + b[:, 3])[:, None] / 2
                + hy[:, None] * nodes[:, 1]
                - observation[1]
            )
            result[start : start + len(b)] = (
                (_kernel_zz(dx, dy, 0.0, lam) @ weights) * hx * hy
            )
        return result

    def _target(self, value):
        return self.atol + self.rtol * np.abs(value)

    @staticmethod
    def _bisect(bounds):
        b = np.asarray(bounds)
        left, right = b.copy(), b.copy()
        along_x = (b[:, 1] - b[:, 0]) >= (b[:, 3] - b[:, 2])
        midx, midy = (b[:, 0] + b[:, 1]) / 2, (b[:, 2] + b[:, 3]) / 2
        left[along_x, 1], right[along_x, 0] = midx[along_x], midx[along_x]
        left[~along_x, 3], right[~along_x, 2] = midy[~along_x], midy[~along_x]
        return np.concatenate((left, right))

    def _refine(self, observation, bounds, lam, previous):
        leaves = self._bisect(np.asarray(bounds)[None, :])
        coarse = self.rectangles(observation, leaves, lam, 4)
        fine = self.rectangles(observation, leaves, lam, 8)
        error = np.abs(fine - coarse)
        for level in range(1, self.max_refinements + 1):
            value = fine.sum()
            estimated = max(error.sum(), abs(value - previous))
            if estimated <= self._target(value):
                return value, estimated, level
            previous = value
            # Retain already accurate leaves, refining those that dominate error.
            split = error > self._target(value) / (2 * len(leaves))
            if not np.any(split):
                split[:] = True
            if len(leaves) + np.count_nonzero(split) > self.max_subpanels:
                break
            children = self._bisect(leaves[split])
            c = self.rectangles(observation, children, lam, 4)
            f = self.rectangles(observation, children, lam, 8)
            leaves = np.concatenate((leaves[~split], children))
            fine = np.concatenate((fine[~split], f))
            error = np.concatenate((error[~split], np.abs(f - c)))
        raise QuadratureConvergenceError(
            f"Regular panel integration failed at {observation.tolist()}, "
            f"bounds={np.asarray(bounds).tolist()}, rtol={self.rtol:g}; "
            "increase max_refinements/max_subpanels or loosen the tolerance."
        )

    def regular(self, observation, bounds, lam):
        coarse = self.rectangles(observation, bounds, lam, 2)
        value = self.rectangles(observation, bounds, lam, 4)
        error = np.abs(value - coarse)
        for degree in (6, 8):
            bad = error > self._target(value)
            if not np.any(bad):
                break
            new = self.rectangles(observation, bounds[bad], lam, degree)
            error[bad] = np.abs(new - value[bad])
            value[bad] = new
        bad_indices = np.flatnonzero(error > self._target(value))
        depth = 0
        for i in bad_indices:
            value[i], error[i], level = self._refine(
                observation, bounds[i], lam, value[i]
            )
            depth = max(depth, level)
        return value, error, len(bad_indices), depth

    def _singular_rule(self, observation, bounds, lam, order):
        distances_x = np.array([observation[0] - bounds[0], bounds[1] - observation[0]])
        distances_y = np.array([observation[1] - bounds[2], bounds[3] - observation[1]])
        if min(*distances_x, *distances_y) <= 0:
            raise ValueError("A singular collocation point must lie inside its panel.")
        radius = min(*distances_x, *distances_y)
        radial_circle = _radial_integral(radius, lam)
        disk = 2 * np.pi * radial_circle
        nodes, weights = _line_rule(self.backend, order)
        value = disk
        for x in distances_x:
            for y in distances_y:
                corner = np.arctan2(y, x)
                for lower, upper, direction in (
                    (0.0, corner, "x"),
                    (corner, np.pi / 2, "y"),
                ):
                    theta = (lower + upper) / 2 + (upper - lower) / 2 * nodes
                    outer = x / np.cos(theta) if direction == "x" else y / np.sin(theta)
                    remainder = _radial_integral(outer, lam) - radial_circle
                    value += (upper - lower) / 2 * (weights @ remainder)
        return value

    def singular(self, observation, bounds, lam):
        """Analytic inscribed disk plus angular integration of the remainder.

        The radial primitive is exact; only the smooth angular integral needs
        quadrature. Rule doubling controls its error, unlike the old fixed rule.
        """
        previous = self._singular_rule(observation, bounds, lam, 15)
        for level in range(self.max_refinements + 1):
            order = 30 * 2**level
            if order > 960:
                break
            value = self._singular_rule(observation, bounds, lam, order)
            error = abs(value - previous)
            if error <= self._target(value):
                return value, error, level
            previous = value
        raise QuadratureConvergenceError(
            f"Singular panel integration failed at {observation.tolist()}, "
            f"bounds={np.asarray(bounds).tolist()}, rtol={self.rtol:g}."
        )
