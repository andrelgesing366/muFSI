"""Analytic radial reduction of the planar unsteady Stokeslet.

This is the F3D_1D / h_matrix_app polar approach, with the cancelling Ei
terms removed. Only a one-dimensional angular integral remains. Reflection
and splitting at the coordinate axes cover every rectangle, including a
singular panel and observations on its boundary, without sampling r=0.
The kernel includes 1/(8*pi) but excludes dynamic viscosity.
"""

from math import factorial
from numbers import Integral

import numpy as np
from numpy.polynomial.legendre import leggauss

# A(z) = 2 exp(-z) (1 + 1/z + 1/z**2) - 2/z**2.
_RADIAL_COEFFICIENTS = np.array(
    [2 * (-1) ** n * (n + 1) / factorial(n + 2) for n in range(19)]
)


class AnalyticConvergenceError(RuntimeError):
    """The remaining angular quadrature did not meet its error target."""


def radial_stokeslet_integral(lower, upper, lam: complex):
    """Return integral_lower^upper A(lambda*r)/(8*pi) dr, in metres.

    The primitive is [1-(1+lambda*r)*exp(-lambda*r)]/(4*pi*lambda**2*r),
    with limit zero at r=0. A Taylor series handles small lambda*r, including
    lambda=0. An interval formula avoids subtracting nearly equal primitives
    for distant, thin panels. lower and upper may be broadcastable arrays.
    """
    a, b = np.broadcast_arrays(
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0):
        raise ValueError("Radial bounds must be finite and nonnegative.")
    if np.any(b < a):
        raise ValueError("Upper radial bounds must be at least lower bounds.")
    if not np.isfinite(lam) or np.real(lam) < 0:
        raise ValueError("lambda must be finite with nonnegative real part.")
    qa, qb = lam * a, lam * b
    value = np.empty(a.shape, dtype=np.complex128)
    small = np.abs(qb) < 0.5
    if np.any(small):
        # (b**(n+1)-a**(n+1))/(b-a), expressed in dimensionless powers.
        u, v = qa[small], qb[small]
        power_sum = np.ones(u.shape, dtype=complex)
        u_power = np.ones(u.shape, dtype=complex)
        series = np.full(u.shape, _RADIAL_COEFFICIENTS[0], dtype=complex)
        for coefficient in _RADIAL_COEFFICIENTS[1:]:
            u_power *= u
            power_sum = v * power_sum + u_power
            series += coefficient * power_sum
        value[small] = (b[small] - a[small]) * series / (8 * np.pi)
    origin = ~small & (a == 0)
    if np.any(origin):
        z = qb[origin]
        value[origin] = (
            b[origin] / (4 * np.pi) * (-np.expm1(-z) - z * np.exp(-z)) / z**2
        )
    regular = ~small & ~origin
    if np.any(regular):
        ar, br = a[regular], b[regular]
        dz = lam * (br - ar)
        z = qa[regular]
        # Algebraically F(b)-F(a), using expm1 in the radial increment.
        value[regular] = (
            np.expm1(-z) * ((br - ar) / br) / ar
            - np.exp(-z) * (1 / br + lam) * np.expm1(-dz)
        ) / (4 * np.pi * lam**2)
    return value


class AnalyticPanelIntegrator:
    """Batched rectangle integration with adaptive angular Gauss rules.

    order=8 starts from the legacy angular rule. Each result is checked
    against twice that order; only unconverged panels are evaluated again.
    max_refinements bounds further doublings after the first comparison.
    Temporary arrays are bounded by batch_size, angular order, and at most
    three angular sectors per reflected quadrant.
    """

    def __init__(
        self, order=8, rtol=2e-3, atol=1e-15, max_refinements=4, batch_size=256
    ):
        for name, value, minimum in (
            ("order", order, 2),
            ("max_refinements", max_refinements, 0),
            ("batch_size", batch_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer >= {minimum}.")
            if value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not np.isfinite([rtol, atol]).all() or rtol <= 0 or atol < 0:
            raise ValueError("rtol must be positive and atol nonnegative, finite.")
        self.order, self.rtol, self.atol = order, rtol, atol
        self.max_refinements, self.batch_size = max_refinements, batch_size
        self._rules = {}

    def _rule(self, order):
        if order not in self._rules:
            self._rules[order] = leggauss(order)
        return self._rules[order]

    def _quadrant(self, rectangles, lam, order):
        """Rectangles [a,b,c,d] in the first quadrant, with a,c >= 0."""
        a, b, c, d = rectangles.T
        angles = np.sort(
            np.column_stack(
                (
                    np.arctan2(c, b),
                    np.arctan2(c, a),
                    np.arctan2(d, b),
                    np.arctan2(d, a),
                )
            ),
            axis=1,
        )
        nodes, weights = self._rule(order)
        half = np.diff(angles, axis=1) / 2
        theta = (angles[:, :-1] + half)[..., None] + half[..., None] * nodes
        cosine, sine = np.cos(theta), np.sin(theta)
        # Degenerate sectors carry zero weight. Their angles can be 0/pi/2.
        valid = half > 0
        cosine = np.where(valid[..., None], cosine, 1.0)
        sine = np.where(valid[..., None], sine, 1.0)
        lower = np.maximum(a[:, None, None] / cosine, c[:, None, None] / sine)
        upper = np.minimum(b[:, None, None] / cosine, d[:, None, None] / sine)
        upper = np.maximum(lower, upper)
        radial = radial_stokeslet_integral(lower, upper, lam)
        return np.sum(radial * half[..., None] * weights, axis=(1, 2))

    def _evaluate(self, relative_bounds, lam, order):
        xlo, xhi, ylo, yhi = relative_bounds.T
        result = np.zeros(len(relative_bounds), dtype=complex)
        # Reflect each nonempty part into the first quadrant. No subtraction
        # of large corner integrals, no dependence on y-panel uniformity.
        for a, b in ((np.maximum(xlo, 0), xhi), (np.maximum(-xhi, 0), -xlo)):
            for c, d in ((np.maximum(ylo, 0), yhi), (np.maximum(-yhi, 0), -ylo)):
                active = (b > a) & (d > c)
                if np.any(active):
                    rectangles = np.column_stack(
                        (a[active], b[active], c[active], d[active])
                    )
                    result[active] += self._quadrant(rectangles, lam, order)
        return result

    def integrate(self, observation, bounds, lam):
        """Return values, absolute error estimates, refined count, max order.

        observation is (2,), bounds is (n,4) in [xlo,xhi,ylo,yhi] order.
        Values include the geometric integral and 1/(8*pi), not viscosity.
        Error estimates compare successive angular orders; they are not
        rigorous bounds. Failure raises AnalyticConvergenceError.
        """
        point = np.asarray(observation, dtype=float)
        panels = np.asarray(bounds, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("observation must contain two finite coordinates.")
        if panels.ndim != 2 or panels.shape[1] != 4:
            raise ValueError("bounds must have shape (n,4).")
        if not np.isfinite(panels).all() or np.any(panels[:, 1] <= panels[:, 0]):
            raise ValueError("Panels must have finite, increasing x bounds.")
        if np.any(panels[:, 3] <= panels[:, 2]):
            raise ValueError("Panels must have increasing y bounds.")
        if not np.isfinite(lam) or np.real(lam) < 0:
            raise ValueError("lambda must be finite with nonnegative real part.")
        relative = panels - point[[0, 0, 1, 1]]
        values = np.empty(len(panels), dtype=complex)
        errors = np.empty(len(panels))
        refined, max_order = 0, 2 * self.order
        for start in range(0, len(panels), self.batch_size):
            stop = min(start + self.batch_size, len(panels))
            batch = relative[start:stop]
            coarse = self._evaluate(batch, lam, self.order)
            pending = np.arange(len(batch))
            for depth in range(self.max_refinements + 1):
                order = self.order * 2 ** (depth + 1)
                fine = self._evaluate(batch[pending], lam, order)
                error = np.abs(fine - coarse)
                target = np.maximum(self.atol, self.rtol * np.abs(fine))
                accepted = np.isfinite(fine) & (error <= target)
                values[start + pending[accepted]] = fine[accepted]
                errors[start + pending[accepted]] = error[accepted]
                max_order = max(max_order, order)
                if np.all(accepted):
                    break
                if depth == self.max_refinements:
                    raise AnalyticConvergenceError(
                        f"Angular quadrature failed for {np.count_nonzero(~accepted)} "
                        f"panels at order {order}; increase max_refinements/order "
                        "or relax tolerance."
                    )
                if depth == 0:
                    refined += np.count_nonzero(~accepted)
                pending, coarse = pending[~accepted], fine[~accepted]
        return values, errors, int(refined), max_order


def integrate_panel_zz(observation, bounds, omega, fluid, **quadrature_options):
    """Integrate Szz/(8*pi*mu) over one rectangle in SI units.

    omega is nonnegative rad/s, with exp(+i*omega*t); omega=0 is supported.
    quadrature_options are passed to AnalyticPanelIntegrator.
    """
    if not np.isfinite(omega) or omega < 0:
        raise ValueError("omega must be finite and nonnegative.")
    if (
        not np.isfinite([fluid.density, fluid.dynamic_viscosity]).all()
        or min(
            fluid.density,
            fluid.dynamic_viscosity,
        )
        <= 0
    ):
        raise ValueError("Fluid density and dynamic viscosity must be positive.")
    lam = np.sqrt(1j * omega / fluid.kinematic_viscosity)
    values, _, _, _ = AnalyticPanelIntegrator(**quadrature_options).integrate(
        observation,
        np.asarray(bounds, dtype=float).reshape(1, 4),
        lam,
    )
    return complex(values[0] / fluid.dynamic_viscosity)
