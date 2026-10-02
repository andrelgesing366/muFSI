"""Continuous weighted-Chebyshev pressure mobility and singular quadrature.

H here denotes the dense coefficient-to-velocity matrix, NOT a hierarchical
H-matrix. exp(+i omega t); pressure is traction applied TO the fluid.
M is the maximum x degree; K is the maximum even-y index (y degree = 2*K).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss

from mufsi.hydrodynamics.stokeslet import unsteady_stokeslet_zz
from mufsi.models.fluid import Fluid
from mufsi.models.geometry import PlateGeometry

DEFAULT_M = 16
DEFAULT_K = 4


class IntegrationConvergenceError(RuntimeError):
    """Numerical integration exhausted its permitted orders."""


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


@lru_cache(maxsize=32)
def line_rule(backend, order):
    """Open Gauss-Legendre rule on [0,1], from Quadpy or NumPy."""
    _integer("order", order, 2)
    if backend == "quadpy":
        try:
            import quadpy

            scheme = quadpy.c1.gauss_legendre(order)
        except (ImportError, RuntimeError) as error:
            raise ImportError(
                "A working Quadpy is required for backend='quadpy'. The repository "
                "uses legacy-quadpy==0.16.10. Or explicitly choose backend='gauss'."
            ) from error
        nodes = np.asarray(scheme.points, dtype=float).reshape(-1)
        weights = np.asarray(scheme.weights, dtype=float).reshape(-1)
    elif backend == "gauss":
        nodes, weights = leggauss(order)
    else:
        raise ValueError("backend must be 'quadpy' or 'gauss'.")
    # Do not assume a backend normalizes its line weights in the same way.
    if not np.isclose(weights.sum(), 2.0) or len(nodes) != order:
        raise ValueError("Unexpected Gauss-Legendre rule normalization/size.")
    return (nodes + 1) / 2, weights / 2


@dataclass(frozen=True)
class WeightedBasis:
    geometry: PlateGeometry
    M: int = DEFAULT_M
    K: int = DEFAULT_K

    def __post_init__(self):
        _integer("M", self.M)
        _integer("K", self.K)
        dims = [self.geometry.length, self.geometry.width, self.geometry.thickness]
        if not np.isfinite(dims).all() or min(dims) <= 0:
            raise ValueError("Geometry dimensions must be finite and positive.")

    @property
    def count(self):
        return (self.M + 1) * (self.K + 1)

    @property
    def transverse_degrees(self):
        """Even-only beam basis; plate experiments override this property."""
        return 2 * np.arange(self.K + 1)

    @property
    def reflection_signs(self):
        return np.tile((-1.0) ** self.transverse_degrees, self.M + 1)

    def angles(self, points):
        points = np.asarray(points, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) == 0
            or not np.isfinite(points).all()
        ):
            raise ValueError("points must be a finite (n,2) array in metres.")
        xi = 2 * points[:, 0] / self.geometry.length - 1
        eta = 2 * points[:, 1] / self.geometry.width
        if np.any(np.abs(xi) >= 1) or np.any(np.abs(eta) >= 1):
            raise ValueError("Collocation/pressure points must lie strictly inside.")
        return np.column_stack((np.arccos(xi), np.arccos(eta)))

    def values(self, points, *, remainder=False):
        angles = self.angles(points)
        x = np.cos(angles[:, :1] * np.arange(self.M + 1))
        y = np.cos(angles[:, 1:] * self.transverse_degrees)
        values = (x[:, :, None] * y[:, None, :]).reshape(-1, self.count)
        if not remainder:
            values /= (np.sin(angles[:, 0]) * np.sin(angles[:, 1]))[:, None]
        return values

    def collocation(self, nx=None, ny_half=None):
        """Interior Chebyshev points; positive y only, no duplicate mirror rows."""
        nx = 2 * (self.M + 1) if nx is None else nx
        ny_half = 2 * (self.K + 1) if ny_half is None else ny_half
        _integer("nx", nx, self.M + 1)
        _integer("ny_half", ny_half, self.K + 1)
        x = self.geometry.length / 2 * (1 - np.cos(np.pi * (np.arange(nx) + 0.5) / nx))
        y = (
            self.geometry.width
            / 2
            * np.sin(np.pi / 2 * (np.arange(ny_half) + 0.5) / ny_half)
        )
        return np.column_stack((np.repeat(x, ny_half), np.tile(y, nx)))


@dataclass(frozen=True)
class Quadrature:
    backend: str = "quadpy"
    orders: tuple[int, ...] = (12, 20, 32, 48, 72, 104)
    rtol: float = 2e-5
    atol: float = 1e-10  # mobility units m/(Pa s), after viscosity division

    def __post_init__(self):
        if self.backend not in {"quadpy", "gauss"}:
            raise ValueError("Unknown quadrature backend.")
        if len(self.orders) < 2:
            raise ValueError("At least two increasing quadrature orders required.")
        for order in self.orders:
            _integer("quadrature order", order, 2)
        if any(a >= b for a, b in zip(self.orders, self.orders[1:])):
            raise ValueError("Quadrature orders must increase strictly.")
        if (
            not np.isfinite([self.rtol, self.atol]).all()
            or self.rtol <= 0
            or self.atol < 0
        ):
            raise ValueError("rtol must be positive and atol nonnegative.")


def _graded_intervals(crossover):
    """Resolve anisotropic distances in the Duffy directional coordinate."""
    c = np.clip(crossover, 1e-6, 1.0)
    edges = [0.0, float(c)]
    while edges[-1] < 1:
        edges.append(min(1.0, 2 * edges[-1]))
    return np.array(edges)


class WeightedMobility:
    """Integrate in cosine coordinates, then split into eight Duffy triangles.

    dA / [sqrt(1-xi²)*sqrt(1-eta²)] = L*W/4 d(alpha)d(beta).
    Each target is the common vertex of the triangles. Duffy supplies a radial
    u Jacobian which cancels the interior 1/r singularity. All integration is
    numerical; no radial primitive or artificial kernel regularization is used.
    """

    def __init__(
        self, basis: WeightedBasis, fluid: Fluid, quadrature: Quadrature | None = None
    ):
        self.basis, self.fluid = basis, fluid
        self.quadrature = Quadrature() if quadrature is None else quadrature
        params = [fluid.density, fluid.dynamic_viscosity]
        if not np.isfinite(params).all() or min(params) <= 0:
            raise ValueError("Fluid density and viscosity must be positive.")
        line_rule(self.quadrature.backend, self.quadrature.orders[0])

    def _row(self, angle, omega, order):
        a0, b0 = angle
        L, W = self.basis.geometry.length, self.basis.geometry.width
        nodes, weights = line_rule(self.quadrature.backend, order)
        result = np.zeros((self.basis.M + 1, self.basis.K + 1), complex)
        evaluations = 0
        for sa, A in ((-1, a0), (1, np.pi - a0)):
            for sb, B in ((-1, b0), (1, np.pi - b0)):
                # Local physical distance scales, used only to grade v.
                dx_scale = L / 2 * np.sin(a0) * A
                dy_scale = W / 2 * np.sin(b0) * B
                for swap in (False, True):
                    ratio = dx_scale / dy_scale if not swap else dy_scale / dx_scale
                    edges = _graded_intervals(ratio)
                    v = (
                        edges[:-1, None] + np.diff(edges)[:, None] * nodes[None, :]
                    ).ravel()
                    vw = (np.diff(edges)[:, None] * weights[None, :]).ravel()
                    u, vv = np.broadcast_arrays(nodes[:, None], v[None, :])
                    da = sa * A * (u * vv if swap else u)
                    db = sb * B * (u if swap else u * vv)
                    alpha, beta = (a0 + da).ravel(), (b0 + db).ravel()
                    # Stable differences of cosines near the target.
                    dx = -L * np.sin(a0 + da / 2) * np.sin(da / 2)
                    dy = -W * np.sin(b0 + db / 2) * np.sin(db / 2)
                    separation = np.column_stack((dx.ravel(), dy.ravel()))
                    kernel = unsteady_stokeslet_zz(separation, omega, self.fluid)
                    jac_weights = (
                        L * W / 4 * A * B * u * weights[:, None] * vw[None, :]
                    ).ravel()
                    bx = np.cos(alpha[:, None] * np.arange(self.basis.M + 1))
                    by = np.cos(beta[:, None] * self.basis.transverse_degrees)
                    result += bx.T @ ((kernel * jac_weights)[:, None] * by)
                    evaluations += len(kernel)
        return result.ravel(), evaluations

    def assemble(self, omega, points=None, *, progress=None):
        """Return H and diagnostics; fail rather than accept unconverged rows.

        Successive-order max column differences use a row-scaled tolerance.
        Estimates are empirical, not rigorous bounds, especially for tiny
        cancelling entries. Tightening tolerance is an independent check.
        """
        if not np.isfinite(omega) or omega < 0:
            raise ValueError("omega must be finite and nonnegative (rad/s).")
        points = self.basis.collocation() if points is None else np.asarray(points)
        angles = self.basis.angles(points)
        H = np.empty((len(points), self.basis.count), complex)
        reports = []
        started = perf_counter()
        for i, angle in enumerate(angles):
            previous, evaluations = self._row(angle, omega, self.quadrature.orders[0])
            for order in self.quadrature.orders[1:]:
                value, count = self._row(angle, omega, order)
                evaluations += count
                error = float(np.max(np.abs(value - previous)))
                target = self.quadrature.atol + self.quadrature.rtol * np.max(
                    abs(value)
                )
                if error <= target:
                    H[i] = value
                    reports.append(
                        {
                            "order": order,
                            "estimated_error": error,
                            "target": float(target),
                            "evaluations": evaluations,
                        }
                    )
                    break
                previous = value
            else:
                raise IntegrationConvergenceError(
                    f"Row {i} at {points[i].tolist()} failed: estimated error "
                    f"{error:g} > {target:g} at order {order}. Increase orders."
                )
            if progress is not None:
                progress(i + 1, len(points))
        return H, {
            "seconds": perf_counter() - started,
            "backend": self.quadrature.backend,
            "omega": float(omega),
            "shape": list(H.shape),
            "rows": reports,
            "max_order": max(r["order"] for r in reports),
            "kernel_evaluations": sum(r["evaluations"] for r in reports),
            "max_estimated_error_ratio": max(
                r["estimated_error"] / r["target"] for r in reports
            ),
        }


def solve_coefficients(H, velocity, *, row_weights=None):
    """Column-scaled complex least squares, supports multiple velocity RHSs."""
    H, velocity = np.asarray(H, complex), np.asarray(velocity, complex)
    if H.ndim != 2 or H.shape[0] < H.shape[1] or not np.isfinite(H).all():
        raise ValueError("H must be finite with at least as many rows as columns.")
    if velocity.ndim not in (1, 2) or velocity.shape[0] != len(H):
        raise ValueError("velocity must have one value per collocation row.")
    if not np.isfinite(velocity).all():
        raise ValueError("velocity must be finite.")
    w = np.ones(len(H)) if row_weights is None else np.asarray(row_weights, float)
    if w.shape != (len(H),) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("row_weights must be finite, positive, and match H rows.")
    matrix = np.sqrt(w)[:, None] * H
    scales = np.linalg.norm(matrix, axis=0)
    if np.any(scales == 0):
        raise ValueError("Zero mobility column.")
    rhs = (
        np.sqrt(w) * velocity if velocity.ndim == 1 else np.sqrt(w)[:, None] * velocity
    )
    scaled, _, rank, singular = np.linalg.lstsq(matrix / scales, rhs, rcond=None)
    if rank < H.shape[1]:
        raise ValueError(f"Rank deficient mobility: {rank}/{H.shape[1]}.")
    coefficients = scaled / scales if scaled.ndim == 1 else scaled / scales[:, None]
    residual = H @ coefficients - velocity
    relative = np.linalg.norm(residual, axis=0) / np.maximum(
        np.linalg.norm(velocity, axis=0), np.finfo(float).tiny
    )
    return coefficients, {
        "rank": int(rank),
        "scaled_condition": float(singular[0] / singular[-1]),
        "relative_residual": np.asarray(relative).tolist(),
    }


@dataclass(frozen=True)
class PlatePressureBasis(WeightedBasis):
    """K is the maximum y degree, including ALL degrees 0..K (not 2*k)."""

    M: int = 4
    K: int = 6

    @property
    def transverse_degrees(self):
        return np.arange(self.K + 1)

    @property
    def reflection_signs(self):
        return np.tile((-1.0) ** self.transverse_degrees, self.M + 1)

    def collocation(self, nx=None, ny=None):
        """Full-width symmetric Chebyshev grid, even ny for paired rows."""
        nx = 2 * (self.M + 1) if nx is None else nx
        ny = 2 * (self.K + 1) if ny is None else ny
        _integer("nx", nx, self.M + 1)
        _integer("ny", ny, self.K + 1)
        x = self.geometry.length / 2 * (1 - np.cos(np.pi * (np.arange(nx) + 0.5) / nx))
        # Construct exact mirrored pairs, avoiding rounding differences.
        positive = np.sin(np.pi / 2 * (np.arange(ny // 2) + 0.5) / (ny // 2))
        eta = np.concatenate((-positive[::-1], [0.0] if ny % 2 else [], positive))
        y = self.geometry.width / 2 * eta
        return np.column_stack((np.repeat(x, ny), np.tile(y, nx)))


class PlateMobility(WeightedMobility):
    """Same Duffy quadrature, reusing reflected rows for a symmetric fluid sheet.

    This reflection uses the unbounded isotropic kernel and rectangular domain,
    not symmetry of the prescribed velocity. Odd-pressure columns change sign.
    """

    def assemble(self, omega, points=None, *, progress=None):
        points = self.basis.collocation() if points is None else np.asarray(points)
        self.basis.angles(points)  # Validate before reflecting.
        reflected = points.copy()
        reflected[:, 1] = abs(reflected[:, 1])
        unique, inverse = np.unique(reflected, axis=0, return_inverse=True)
        half, report = super().assemble(omega, unique, progress=progress)
        half[unique[:, 1] == 0] *= (1 + self.basis.reflection_signs) / 2
        H = half[inverse].copy()
        H[points[:, 1] < 0] *= self.basis.reflection_signs
        report.update(
            shape=list(H.shape),
            integrated_rows=len(unique),
            reflected_rows=len(points) - len(unique),
        )
        return H, report
