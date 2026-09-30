"""Sader (1998), rectangular-beam hydrodynamic function and driven compliance.

Reference: J. Appl. Phys. 84, 64–76, doi:10.1063/1.368002.
This is a deterministic uniform-load response, not a thermal noise PSD.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.special import kve

from mufsi.models.fluid import Fluid
from mufsi.models.geometry import BeamGeometry, PlateGeometry
from mufsi.models.material import Material


def gamma_function(reynolds):
    """Rectangular hydrodynamic function, paper's exp(-i omega t) convention.

    Implements Eqs. (18), (20)–(22). The rational fit was validated in the
    paper over 1e-6 <= Re <= 1e4. Arguments must be finite and positive.
    """
    re = np.asarray(reynolds, dtype=float)
    if not np.isfinite(re).all() or np.any(re <= 0):
        raise ValueError("Reynolds numbers must be finite and strictly positive.")
    root = np.sqrt(1j * re)
    z = -1j * root
    circular = 1 + 4j * kve(1, z) / (root * kve(0, z))
    t = np.log10(re)
    real = polyval(t, [
        .91324, -.48274, .46842, -.12886, .044055, -.0035117, .00069085,
    ]) / polyval(t, [
        1, -.56964, .48690, -.13444, .045155, -.0035862, .00069085,
    ])
    imag = polyval(t, [
        -.024134, -.029256, .016294, -.00010961, .000064577, -.000044510,
    ]) / polyval(t, [
        1, -.59702, .55182, -.18357, .079156, -.014369, .0028361,
    ])
    return (real + 1j * imag) * circular


def _static_inverse(coefficients):
    """Polynomial inverse of d^4/dr^4 with clamped/free end conditions."""
    c = np.zeros(len(coefficients) + 4)
    k = np.arange(len(coefficients))
    c[4:] = coefficients / ((k + 1) * (k + 2) * (k + 3) * (k + 4))
    j = np.arange(4, len(c))
    c[3] = -np.sum(j * (j - 1) * (j - 2) * c[4:]) / 6
    c[2] = -(np.sum(j * (j - 1) * c[4:]) + 6 * c[3]) / 2
    return c


def _uniform_transfer(b4, r):
    """Solve w'''' - b4*w = 1, w(0)=w'(0)=w''(1)=w'''(1)=0."""
    b = complex(b4)**0.25
    if abs(b) < 0.5:
        # Neumann series about the exact static solution avoids 1/b^4 cancellation.
        c = _static_inverse(np.array([1.0]))
        result = polyval(r, c).astype(complex)
        power = 1.0 + 0j
        for _ in range(8):
            c = _static_inverse(c)
            power *= b4
            result += power * polyval(r, c)
        return result
    e = np.exp(-b)
    cb, sb = np.cos(b), np.sin(b)
    # Bounded real exponentials avoid the large cosh/sinh terms in Eq. (B4).
    system = np.array([
        [1, 0, 1, e],
        [0, 1, -1, e],
        [-cb, -sb, e, 1],
        [sb, -cb, -e, 1],
    ], dtype=complex)
    c = np.linalg.solve(system, np.array([1 / b4, 0, 0, 0], dtype=complex))
    value = (
        c[0] * np.cos(b * r) + c[1] * np.sin(b * r)
        + c[2] * np.exp(-b * r) + c[3] * np.exp(-b * (1 - r)) - 1 / b4
    )
    value[r == 0] = 0
    return value


@dataclass(frozen=True)
class SaderMethod:
    """Euler–Bernoulli cantilever reference with uniform line/pressure loading.

    Responses use exp(+i omega t), by conjugating the paper's Gamma. Geometry
    must satisfy the slender, thin rectangular beam assumptions of the model.
    Units are SI; frequency arrays are in Hz and positions are in metres.
    """

    geometry: BeamGeometry | PlateGeometry
    material: Material
    fluid: Fluid

    def __post_init__(self):
        values = [
            self.geometry.length, self.geometry.width, self.geometry.thickness,
            self.material.young_modulus, self.material.density,
            self.fluid.density, self.fluid.dynamic_viscosity,
        ]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Sader geometry and physical parameters must be positive.")

    @property
    def flexural_rigidity(self):
        return (
            self.material.young_modulus * self.geometry.width
            * self.geometry.thickness**3 / 12
        )

    def hydrodynamic_function(self, frequencies):
        """Return Gamma in the library's exp(+i omega t) convention."""
        f = np.asarray(frequencies, dtype=float)
        if not np.isfinite(f).all() or np.any(f <= 0):
            raise ValueError("frequencies must be finite and strictly positive.")
        re = (
            self.fluid.density * 2 * np.pi * f * self.geometry.width**2
            / (4 * self.fluid.dynamic_viscosity)
        )
        return gamma_function(re).conjugate()

    def displacement_per_line_force(self, frequencies, positions=None):
        """Compliance (m/(N/m)), shape (nfrequencies, npositions).

        Uniform line force along the whole length. Positions default to the
        tip. Scalar frequencies/positions are treated as one-element arrays.
        """
        f = np.atleast_1d(np.asarray(frequencies, dtype=float))
        x = np.atleast_1d(np.asarray(
            self.geometry.length if positions is None else positions, dtype=float,
        ))
        if f.ndim != 1 or x.ndim != 1 or f.size == 0 or x.size == 0:
            raise ValueError("frequencies and positions must be nonempty 1D arrays.")
        if not np.isfinite(x).all() or np.any((x < 0) | (x > self.geometry.length)):
            raise ValueError("positions must lie on the beam, between 0 and length.")
        omega = 2 * np.pi * f
        gamma = self.hydrodynamic_function(f)
        g = self.geometry
        line_mass = self.material.density * g.width * g.thickness
        fluid_mass = np.pi * self.fluid.density * g.width**2 / 4 * gamma
        b4 = omega**2 * g.length**4 / self.flexural_rigidity * (line_mass + fluid_mass)
        return np.array([
            _uniform_transfer(b, x / g.length) for b in b4
        ]) * (g.length**4 / self.flexural_rigidity)

    def displacement_per_pressure(self, frequencies, positions=None):
        """Compliance in m/Pa for uniform pressure, with the same output shape."""
        return self.geometry.width * self.displacement_per_line_force(
            frequencies, positions,
        )
