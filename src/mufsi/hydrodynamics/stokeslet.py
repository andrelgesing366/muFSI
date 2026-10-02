"""Unsteady 3D Stokeslet with exp(+i omega t), in physical SI coordinates."""

from math import factorial

import numpy as np

from mufsi.models.fluid import Fluid

# Taylor coefficients eliminate cancellation of the 1/(lambda*r)^2 terms.
_A = np.array(
    [
        2
        * (
            (-1) ** n / factorial(n)
            + (-1) ** (n + 1) / factorial(n + 1)
            + (-1) ** (n + 2) / factorial(n + 2)
        )
        for n in range(11)
    ]
)
_B = np.array(
    [
        -2
        * (
            (-1) ** n / factorial(n)
            + 3 * (-1) ** (n + 1) / factorial(n + 1)
            + 3 * (-1) ** (n + 2) / factorial(n + 2)
        )
        for n in range(11)
    ]
)


def _coefficients(z):
    a, b = np.empty_like(z, dtype=complex), np.empty_like(z, dtype=complex)
    small = np.abs(z) < 0.05
    a[small] = np.polynomial.polynomial.polyval(z[small], _A)
    b[small] = np.polynomial.polynomial.polyval(z[small], _B)
    q = z[~small]
    e = np.exp(-q)
    a[~small] = 2 * e * (1 + 1 / q + 1 / q**2) - 2 / q**2
    b[~small] = -2 * e * (1 + 3 / q + 3 / q**2) + 6 / q**2
    return a, b


def _kernel_zz(dx, dy, dz, lam):
    """Kernel including 1/(8*pi), excluding viscosity; units 1/m."""
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    if np.any(r == 0):
        raise ValueError("The point Stokeslet is singular at zero separation.")
    a, b = _coefficients(np.asarray(lam * r, dtype=complex))
    return (a + b * (dz / r) ** 2) / (8 * np.pi * r)


def _radial_integral(radius, lam):
    """Integral of planar Szz*r dr from zero to radius, excluding viscosity."""
    radius = np.asarray(radius, dtype=float)
    q = lam * radius
    value = np.empty(q.shape, dtype=complex)
    small = np.abs(q) < 0.05
    value[small] = (
        radius[small]
        / (8 * np.pi)
        * np.polynomial.polynomial.polyval(
            q[small],
            _A / np.arange(1, len(_A) + 1),
        )
    )
    z = q[~small]
    value[~small] = (
        radius[~small] / (4 * np.pi) * (-np.expm1(-z) - z * np.exp(-z)) / z**2
    )
    return value


def unsteady_stokeslet_zz(separation, omega: float, fluid: Fluid):
    """Return Szz/(8*pi*mu) for separation (...,2) or (...,3), in metres.

    Multiply by a point force in N to obtain velocity in m/s. omega is rad/s
    and may be zero for the steady limit. A zero separation raises ValueError;
    singular panels must be integrated, not sampled at their singular point.
    lambda = sqrt(+i*omega/nu), consistent with exp(+i*omega*t).
    """
    if not np.isfinite(omega) or omega < 0:
        raise ValueError("omega must be finite and nonnegative.")
    if (
        not np.isfinite([fluid.density, fluid.dynamic_viscosity]).all()
        or min(fluid.density, fluid.dynamic_viscosity) <= 0
    ):
        raise ValueError("Fluid density and viscosity must be positive.")
    d = np.asarray(separation, dtype=float)
    if d.ndim < 1 or d.shape[-1] not in (2, 3) or not np.isfinite(d).all():
        raise ValueError("separation must be finite with last dimension 2 or 3.")
    dz = d[..., 2] if d.shape[-1] == 3 else np.zeros(d.shape[:-1])
    lam = np.sqrt(1j * omega / fluid.kinematic_viscosity)
    return _kernel_zz(d[..., 0], d[..., 1], dz, lam) / fluid.dynamic_viscosity
