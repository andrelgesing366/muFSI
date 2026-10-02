"""Independent even/odd integration and KL modal-coupling checks."""

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ellipkm1
from weighted_beam_spectrum import solve_modal_response
from weighted_plate_mobility import (
    PlateMobility,
    PlateModes,
    PlatePressureBasis,
    plate_modal_projection,
)
from weighted_pressure_mobility import (
    Fluid,
    PlateGeometry,
    Quadrature,
    WeightedMobility,
    solve_coefficients,
)


class PlateFluidChecks(unittest.TestCase):
    def setUp(self):
        self.basis = PlatePressureBasis(PlateGeometry(2.0, 1.0, 0.01), M=2, K=3)
        self.hydro = PlateMobility(
            self.basis, Fluid(1, 1), Quadrature(backend="gauss", rtol=1e-7, atol=1e-10)
        )

    def test_odd_column_against_independent_elliptic_integral(self):
        # x=center allows analytical x integration. Only adaptive 1D beta
        # integration remains: an independent route, with no Duffy mapping.
        a, b, Y = 1.0, 0.5, 0.13

        def integrand(beta):
            d = Y - b * np.cos(beta)
            denominator = a * a + d * d
            return (
                np.cos(beta) * 2 * ellipkm1(d * d / denominator) / np.sqrt(denominator)
            )

        integral = quad(
            integrand, 0, np.pi, points=[np.arccos(Y / b)], epsabs=1e-8, epsrel=1e-8
        )[0]
        expected = a * b / (8 * np.pi) * integral
        H, _ = self.hydro.assemble(0, np.array([[a, Y]]))
        np.testing.assert_allclose(H[0, 1], expected, rtol=2e-7)
        self.assertGreater(abs(expected), 0.001)

    def test_reflection_reuse_against_independently_integrated_negative_rows(self):
        points = np.array([[0.6, 0.19], [0.6, -0.19]])
        H, report = self.hydro.assemble(2, points)
        # Direct base integrator integrates the negative target separately.
        direct, _ = WeightedMobility(
            self.basis, self.hydro.fluid, self.hydro.quadrature
        ).assemble(2, points)
        np.testing.assert_allclose(H, direct, rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(H[1], H[0] * self.basis.reflection_signs, atol=1e-14)
        self.assertEqual(report["integrated_rows"], 1)

    def test_2d_projection_has_nonzero_odd_generalized_force(self):
        # Manufactured phi=eta=T1(eta) gives C_0,1=ab*pi^2/2.
        g = self.basis.geometry
        C = plate_modal_projection(
            self.basis, lambda p: 2 * p[:, 1] / g.width, order=40
        )
        expected = g.length * g.width * np.pi**2 / 8
        np.testing.assert_allclose(C[0, 1], expected, rtol=1e-12)
        masked = C.copy()
        masked[0, 1] = 0
        np.testing.assert_allclose(masked, 0, atol=1e-12)

    def test_antisymmetric_velocity_recovers_odd_pressure(self):
        points = self.basis.collocation()
        H, _ = self.hydro.assemble(2, points)
        velocity = points[:, 1] * (1 + points[:, 0])
        coefficients, fit = solve_coefficients(H, velocity)
        even = self.basis.reflection_signs == 1
        self.assertLess(
            np.linalg.norm(coefficients[even]) / np.linalg.norm(coefficients), 1e-10
        )
        self.assertEqual(fit["rank"], self.basis.count)
        fields = self.basis.values(points) @ coefficients
        mirrored = self.basis.values(points * [1, -1]) @ coefficients
        np.testing.assert_allclose(fields, -mirrored, rtol=1e-9, atol=1e-10)


class PlateStructuralChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mufsi import KirchhoffPlate, Material

        cls.plate = KirchhoffPlate(
            PlateGeometry(1, 0.7, 0.01),
            Material(1e7, 1500, 0.3),
            mesh_resolution=(16, 12),
            boundary_condition="simply_supported",
        )
        cls.modes = PlateModes(cls.plate, count=3)

    def test_simply_supported_frequencies_against_exact_plate(self):
        p = self.plate
        exact = np.sort(
            [
                np.pi**2
                * np.sqrt(p.bending_rigidity / p.surface_density)
                * ((m / p.geometry.length) ** 2 + (n / p.geometry.width) ** 2)
                for m in range(1, 4)
                for n in range(1, 4)
            ]
        )[:3]
        np.testing.assert_allclose(self.modes.dry_omega, exact, rtol=0.045)
        self.assertLess(np.max(self.modes.eigen_residuals), 1e-7)

    def test_modal_normalization_and_dry_response(self):
        M = self.plate.mass_matrix()
        try:
            row, col, data = M.getValuesCSR()
            from scipy.sparse import csr_matrix

            matrix = csr_matrix((data, col, row), shape=M.getSize())
            projected = self.modes.dof_modes.T @ matrix @ self.modes.dof_modes
            np.testing.assert_allclose(
                projected, np.diag(self.modes.mass), rtol=1e-8, atol=1e-10
            )
        finally:
            M.destroy()
        force = np.array([1.0, 0.3, -0.2])
        omega = 0.23 * self.modes.dry_omega[0]
        d, error = solve_modal_response(self.modes, omega, np.zeros((3, 3)), force)
        expected = force / (self.modes.stiffness - omega**2 * self.modes.mass)
        np.testing.assert_allclose(d, expected, rtol=1e-12)
        self.assertLess(error, 1e-12)

    def test_supported_edge_and_point_force_virtual_work(self):
        from mufsi import PointLoad

        points = np.column_stack((np.zeros(9), np.linspace(-0.35, 0.35, 9)))
        np.testing.assert_allclose(self.modes.values(points), 0, atol=1e-9)
        position = (0.43, 0.13)
        F = self.plate.force_vector(PointLoad(position, 2.0))
        try:
            projected = self.modes.dof_modes.T @ F.array
            np.testing.assert_allclose(
                projected, 2 * self.modes.values(np.array([position]))[0], rtol=1e-10
            )
        finally:
            F.destroy()


if __name__ == "__main__":
    unittest.main()
