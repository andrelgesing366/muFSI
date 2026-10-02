"""Independent checks of continuous weighted mobility and force projection."""

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ellipk
from weighted_beam_comparison import local_modal_impedance, uniform_modal_force
from weighted_beam_spectrum import CantileverModes, solve_modal_response
from weighted_pressure_mobility import (
    Fluid,
    IntegrationConvergenceError,
    PlateGeometry,
    Quadrature,
    WeightedBasis,
    WeightedMobility,
    solve_coefficients,
)
from weighted_pressure_projection import modal_projection


class MobilityChecks(unittest.TestCase):
    def setUp(self):
        self.geometry = PlateGeometry(3.0, 1.0, 0.01)
        self.fluid = Fluid(1.0, 1.0)

    def test_steady_center_against_elliptic_integral(self):
        # Integrate x analytically with the complete elliptic integral, then
        # independent adaptive 1D quadrature in beta. No Duffy implementation.
        a, b = self.geometry.length / 2, self.geometry.width / 2
        integral = quad(
            lambda beta: (
                ellipk(a * a / (a * a + b * b * np.cos(beta) ** 2))
                / np.sqrt(a * a + b * b * np.cos(beta) ** 2)
            ),
            0,
            np.pi / 2,
            epsabs=1e-9,
            epsrel=1e-9,
        )[0]
        expected = 4 * a * b * integral / (8 * np.pi * self.fluid.dynamic_viscosity)
        basis = WeightedBasis(self.geometry, M=2, K=1)
        hydro = WeightedMobility(basis, self.fluid, Quadrature(rtol=1e-7, atol=1e-10))
        H, _ = hydro.assemble(0, np.array([[a, 0.0]]))
        self.assertAlmostEqual(H[0, 0].real / expected, 1, places=6)
        self.assertLess(np.max(abs(H[0, 2:4])), 1e-12)  # odd longitudinal terms

    def test_mirror_and_backend_agreement_unsteady(self):
        basis = WeightedBasis(self.geometry, M=1, K=1)
        points = np.array([[0.7, 0.17], [0.7, -0.17]])
        quadpy = WeightedMobility(basis, self.fluid, Quadrature(rtol=1e-6))
        gauss = WeightedMobility(
            basis, self.fluid, Quadrature(backend="gauss", rtol=1e-6)
        )
        a, _ = quadpy.assemble(3.0, points)
        b, _ = gauss.assemble(3.0, points[:1])
        np.testing.assert_allclose(a[0], a[1], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(a[:1], b, rtol=1e-8, atol=1e-10)

    def test_weighted_force_moments(self):
        basis = WeightedBasis(self.geometry, M=3, K=3)
        C = modal_projection(basis, lambda x: np.ones_like(x))
        expected = self.geometry.length * self.geometry.width * np.pi**2 / 4
        self.assertAlmostEqual(C[0, 0], expected, places=11)
        np.testing.assert_allclose(C[0, 1:], 0, atol=1e-12)

    def test_manufactured_complex_pressure_recovery(self):
        basis = WeightedBasis(self.geometry, M=1, K=1)
        solver = WeightedMobility(basis, self.fluid, Quadrature(rtol=1e-6))
        points = basis.collocation()
        H, _ = solver.assemble(2, points)
        exact = np.array([1 + 2j, -0.4j, 0.3, 0.1 + 0.2j])
        recovered, report = solve_coefficients(H, H @ exact)
        np.testing.assert_allclose(recovered, exact, rtol=1e-10, atol=1e-10)
        self.assertEqual(report["rank"], 4)
        check, _ = solver.assemble(2, np.array([[0.34, 0.05], [2.77, 0.23]]))
        np.testing.assert_allclose(check @ recovered, check @ exact, rtol=1e-10)

    def test_failure_is_explicit(self):
        basis = WeightedBasis(self.geometry, M=2, K=2)
        solver = WeightedMobility(
            basis, self.fluid, Quadrature(orders=(2, 3), rtol=1e-15, atol=0)
        )
        with self.assertRaises(IntegrationConvergenceError):
            solver.assemble(3, np.array([[0.1, 0.49]]))
        with self.assertRaises(ValueError):
            basis.angles(np.array([[0, 0]]))
        with self.assertRaises(ValueError):
            solve_coefficients(np.ones((5, 4)), np.ones(5))

    def test_eb_static_tip_compliance_and_mode_mass(self):
        beam = CantileverModes(self.geometry, young_modulus=7, density=2, count=6)
        expected = self.geometry.length**3 / (3 * beam.EI)
        d, error = solve_modal_response(beam, 0, np.zeros((6, 6)), np.ones(6))
        # Six modes omit a known positive high-mode compliance tail (~0.019%).
        self.assertLess(abs(d.sum() / expected - 1), 2e-4)
        self.assertLess(error, 1e-12)
        np.testing.assert_allclose(
            beam.mass, beam.line_mass * self.geometry.length / 4, rtol=1e-7
        )
        np.testing.assert_allclose(
            beam.values(np.array([0, self.geometry.length])),
            np.array([np.zeros(6), np.ones(6)]),
            atol=1e-10,
        )

    def test_fluid_sign_adds_mass_and_damping(self):
        beam = CantileverModes(self.geometry, young_modulus=7, density=2, count=1)
        added_mass = 0.3 * beam.mass[0]
        damping = 0.04 * beam.mass[0] * beam.dry_omega[0]
        omega = 0.8 * beam.dry_omega[0]
        impedance = np.array([[damping + 1j * omega * added_mass]])
        d, error = solve_modal_response(beam, omega, impedance, np.array([1.0]))
        expected = 1 / (
            beam.stiffness[0]
            - omega**2 * (beam.mass[0] + added_mass)
            + 1j * omega * damping
        )
        np.testing.assert_allclose(d[0], expected, rtol=1e-13)
        self.assertLess(error, 1e-12)

    def test_uniform_loading_static_compliance(self):
        beam = CantileverModes(self.geometry, young_modulus=7, density=2, count=6)
        force = uniform_modal_force(beam, 0.3)
        d, _ = solve_modal_response(beam, 0, np.zeros((6, 6)), force)
        exact = 0.3 * self.geometry.length**4 / (8 * beam.EI)
        self.assertLess(abs(d.sum() / exact - 1), 1e-5)

    def test_local_fluid_projection_matches_independent_integral(self):
        beam = CantileverModes(self.geometry, young_modulus=7, density=2, count=3)

        class ConstantSection:
            @staticmethod
            def line_impedance(omega):
                return 2 + 0.4j

        t, w = np.polynomial.legendre.leggauss(80)
        shapes = beam.values(self.geometry.length * (t + 1) / 2)
        expected = (
            (2 + 0.4j) * self.geometry.length / 2 * (shapes.T @ (w[:, None] * shapes))
        )
        np.testing.assert_allclose(
            local_modal_impedance(beam, ConstantSection(), 1),
            expected,
            rtol=1e-11,
            atol=1e-11,
        )


if __name__ == "__main__":
    unittest.main()
