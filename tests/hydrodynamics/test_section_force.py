"""Local line-force normalization against independent Kelvin and Sader formulas."""

import unittest

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import keip, kerp

from mufsi import BeamGeometry, Fluid, SectionForce2D
from mufsi.hydrodynamics.sader import gamma_function


class SectionForceTests(unittest.TestCase):
    def setUp(self):
        self.geometry = BeamGeometry(800e-6, 50e-6, 10e-6)
        self.fluid = Fluid(997, 890e-6)

    def test_sader_force_units_and_dissipation(self):
        h = SectionForce2D(self.geometry, self.fluid, method="sader")
        omega = 2 * np.pi * 1e4
        re = omega * self.geometry.width**2 / (4 * self.fluid.kinematic_viscosity)
        mass = (
            np.pi
            * self.fluid.density
            * self.geometry.width**2
            / 4
            * gamma_function(re).conjugate()
        )
        np.testing.assert_allclose(h.dynamic_stiffness(omega), -(omega**2) * mass)
        v = np.array([1.0, 2j])
        np.testing.assert_allclose(
            h.resisting_force_from_velocity(omega, v), 1j * omega * mass * v
        )
        self.assertGreater(h.line_impedance(omega).real, 0)
        self.assertGreater(h.line_impedance(omega).imag / omega, 0)

    def test_legacy_numeric_force_formula_and_harmonic_sign(self):
        ny, mu = 12, self.fluid.dynamic_viscosity
        h = SectionForce2D(
            self.geometry, self.fluid, ny=ny, integration="legacy_trapezoid"
        )
        xi = np.cos(np.linspace(-np.pi, 0, ny + 1))
        centers = (xi[:-1] + xi[1:]) / 2
        for hz in (1e3, 2e4, 1e5):
            omega = 2 * np.pi * hz
            beta = (
                omega * (self.geometry.width / 2) ** 2 / self.fluid.kinematic_viscosity
            )
            upper = np.sqrt(beta) * (xi[None, 1:] - centers[:, None])
            lower = np.sqrt(beta) * (xi[None, :-1] - centers[:, None])

            def primitive(z):
                a = abs(z)
                return np.sign(z) * (1 / a + kerp(a) + 1j * keip(a))

            A = (primitive(upper) - primitive(lower)) / (2j * np.pi * np.sqrt(beta))
            # Original helper: arbitrary dummy v0, inverse replaced with solve.
            v0 = 10
            delta = np.linalg.solve(A, np.ones(ny) / v0)
            actual_old_force_per_displacement = (
                -mu * 1j * omega * v0 * trapezoid(delta, x=centers)
            )
            np.testing.assert_allclose(
                h.dynamic_stiffness(omega),
                -actual_old_force_per_displacement,
                rtol=2e-12,
            )

    def test_full_panel_force_near_sader_and_pressure_residual(self):
        h = SectionForce2D(self.geometry, self.fluid, ny=64)
        reference = SectionForce2D(self.geometry, self.fluid, method="sader")
        for hz in (1e3, 1e4, 1e5):
            omega = 2 * np.pi * hz
            p = h.section_pressure(omega)
            np.testing.assert_allclose(
                h._section.section_mobility(omega) @ p, 1, atol=1e-13
            )
            self.assertLess(
                abs(h.line_impedance(omega) / reference.line_impedance(omega) - 1), 0.03
            )
            self.assertGreater(h.line_impedance(omega).real, 0)
            self.assertFalse(p.flags.writeable)
            self.assertIs(h.section_pressure(omega), p)
        edges = h.section_grid.panel_edges
        self.assertAlmostEqual(np.diff(edges).sum() / self.geometry.width, 1)

    def test_invalid_controls_and_omega(self):
        for controls in (
            {"ny": 1},
            {"ny": True},
            {"method": "3d"},
            {"integration": "unknown"},
        ):
            with self.assertRaises(ValueError):
                SectionForce2D(self.geometry, self.fluid, **controls)
        h = SectionForce2D(self.geometry, self.fluid)
        for omega in (0, -1, np.nan, [1, 2]):
            with self.assertRaises(ValueError):
                h.line_impedance(omega)


if __name__ == "__main__":
    unittest.main()
