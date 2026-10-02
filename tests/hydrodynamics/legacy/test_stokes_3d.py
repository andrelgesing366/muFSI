"""Independent Stokeslet/panel checks and coarse F3D regression tests."""

import importlib.util
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

from mufsi import Fluid, FluidGrid, PlateGeometry, Stokes2D
from mufsi.hydrodynamics.legacy.panel_quadrature import (
    PanelIntegrator,
    QuadratureConvergenceError,
)
from mufsi.hydrodynamics.legacy.stokes_3d import Stokes3D
from mufsi.hydrodynamics.stokeslet import unsteady_stokeslet_zz


class Kernel3DTests(unittest.TestCase):
    def setUp(self):
        self.fluid = Fluid(997, 890e-6)

    def test_legacy_kernel_with_harmonic_conversion(self):
        separation = np.array([[1e-5, 2e-5, 0], [5e-5, 1e-5, 2e-5]])
        omega = 2 * np.pi * 1e4
        r = np.linalg.norm(separation, axis=1)
        # Original Fluid.py used sqrt(-i*omega/nu); conjugate for exp(+i*omega*t).
        z = np.sqrt(-1j * omega / self.fluid.kinematic_viscosity) * r
        a = 2 * np.exp(-z) * (1 + 1 / z + 1 / z**2) - 2 / z**2
        b = -2 * np.exp(-z) * (1 + 3 / z + 3 / z**2) + 6 / z**2
        expected = (
            (a + b * separation[:, 2] ** 2 / r**2)
            / (8 * np.pi * r * self.fluid.dynamic_viscosity)
        ).conjugate()
        np.testing.assert_allclose(
            unsteady_stokeslet_zz(separation, omega, self.fluid),
            expected,
            rtol=1e-13,
        )

    def test_steady_limit_and_small_argument(self):
        d = np.array([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0]])
        fluid = Fluid(1, 1)
        r = np.linalg.norm(d, axis=1)
        expected = (1 + (d[:, 2] / r) ** 2) / (8 * np.pi * r)
        np.testing.assert_allclose(unsteady_stokeslet_zz(d, 0, fluid), expected)
        np.testing.assert_allclose(
            unsteady_stokeslet_zz(d, 1e-20, fluid),
            expected,
            rtol=1e-8,
        )
        with self.assertRaises(ValueError):
            unsteady_stokeslet_zz([0.0, 0.0], 1, fluid)

    def test_singular_steady_rectangle_exact_integral(self):
        # Four quadrant integrals of 1/r, including an off-centre collocation point.
        observation = np.array([0.3, 0.7])
        bounds = np.array([0.0, 1.0, 0.0, 2.0])
        expected = sum(
            x * np.arcsinh(y / x) + y * np.arcsinh(x / y)
            for x in (0.3, 0.7)
            for y in (0.7, 1.3)
        ) / (8 * np.pi)
        value, error, _ = PanelIntegrator("gauss", rtol=1e-10).singular(
            observation,
            bounds,
            0j,
        )
        np.testing.assert_allclose(value, expected, rtol=1e-10)
        self.assertLess(error / abs(value), 1e-10)

    def test_singular_unsteady_against_duffy_triangles(self):
        fluid = self.fluid
        observation = np.array([0.4e-5, 0.6e-5])
        bounds = np.array([0.0, 1e-5, 0.0, 1.4e-5])
        omega = 2 * np.pi * 2e4
        lam = np.sqrt(1j * omega / fluid.kinematic_viscosity)
        actual, _, _ = PanelIntegrator("gauss", rtol=1e-9).singular(
            observation,
            bounds,
            lam,
        )
        nodes, weights = leggauss(60)
        s, w = (nodes + 1) / 2, weights / 2
        corners = (
            np.array(
                [
                    [bounds[0], bounds[2]],
                    [bounds[1], bounds[2]],
                    [bounds[1], bounds[3]],
                    [bounds[0], bounds[3]],
                ]
            )
            - observation
        )
        expected = 0j
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            edge = (1 - s[:, None]) * a + s[:, None] * b
            displacement = s[:, None, None] * edge[None, :, :]
            jacobian = abs(a[0] * b[1] - a[1] * b[0]) * s[:, None]
            integrand = unsteady_stokeslet_zz(displacement, omega, fluid) * jacobian
            expected += np.sum(w[:, None] * w[None, :] * integrand)
        np.testing.assert_allclose(
            actual / fluid.dynamic_viscosity, expected, rtol=1e-9
        )

    def test_regular_near_panel_and_refinement_limit(self):
        observation = np.array([0.0, 0.0])
        bounds = np.array([[0.02, 1.0, -0.02, 0.02]])
        lam = 3 + 3j
        loose = PanelIntegrator("gauss", rtol=2e-3)
        fine = PanelIntegrator("gauss", rtol=1e-7, max_refinements=20)
        a, _, refined, _ = loose.regular(observation, bounds, lam)
        reference, _, _, _ = fine.regular(observation, bounds, lam)
        self.assertGreater(refined, 0)
        self.assertLess(abs(a[0] / reference[0] - 1), 2e-3)
        with self.assertRaises(QuadratureConvergenceError):
            PanelIntegrator("gauss", rtol=1e-12, max_refinements=0).regular(
                observation,
                bounds,
                lam,
            )

    def test_elongated_self_panel_at_example_grid_limit(self):
        grid = FluidGrid.cantilever(PlateGeometry(500e-6, 50e-6, 5e-6), nx=32, ny=64)
        point, bounds = grid.points[0], grid.panel_bounds[0]
        expected = sum(
            x * np.arcsinh(y / x) + y * np.arcsinh(x / y)
            for x in (point[0] - bounds[0], bounds[1] - point[0])
            for y in (point[1] - bounds[2], bounds[3] - point[1])
        ) / (8 * np.pi)
        integrator = PanelIntegrator("gauss")
        value, error, depth = integrator.singular(point, bounds, 0j)
        self.assertLess(abs(value / expected - 1), integrator.rtol)
        self.assertLessEqual(error, integrator._target(value))
        self.assertGreater(depth, 0)

    @unittest.skipUnless(importlib.util.find_spec("quadpy"), "Quadpy unavailable")
    def test_quadpy_rules_and_gauss_agree(self):
        observation = np.array([0.0, 0.0])
        bounds = np.array([[0.02, 1.0, -0.02, 0.02], [-0.2, 0.4, 0.6, 0.8]])
        quadpy = PanelIntegrator("quadpy", rtol=1e-7, max_refinements=20)
        gauss = PanelIntegrator("gauss", rtol=1e-7, max_refinements=20)
        q, _, _, _ = quadpy.regular(observation, bounds, 3 + 3j)
        g, _, _, _ = gauss.regular(observation, bounds, 3 + 3j)
        np.testing.assert_allclose(q, g, rtol=2e-7)


class Grid3DTests(unittest.TestCase):
    def test_half_chebyshev_rule_and_full_domain(self):
        geometry = PlateGeometry(500e-6, 50e-6, 5e-6)
        g = FluidGrid.cantilever(geometry, nx=8, ny=16)
        self.assertEqual((g.nx, g.ny), (8, 16))
        np.testing.assert_allclose(
            g.x / geometry.length,
            np.sin(
                (2 * np.arange(1, 9) - 1) * np.pi / 32,
            ),
        )
        self.assertEqual(g.x_panel_edges[0], 0)
        self.assertEqual(g.x_panel_edges[-1], geometry.length)
        self.assertLess(
            abs(g.weights.sum() / (geometry.length * geometry.width) - 1), 0.004
        )
        uniform = FluidGrid.cantilever(
            geometry,
            nx=4,
            ny=5,
            x_uniform=True,
            y_uniform=True,
        )
        self.assertEqual(uniform.panel_edges[0], -geometry.width / 2)
        self.assertEqual(uniform.panel_edges[-1], geometry.width / 2)
        self.assertAlmostEqual(uniform.weights.sum(), geometry.length * geometry.width)
        self.assertFalse(g.x_panel_edges.flags.writeable)
        with self.assertRaises(ValueError):
            Stokes3D(
                Fluid(997, 890e-6),
                FluidGrid.chebyshev_gauss(geometry, nx=5, ny=8),
                quadrature_backend="gauss",
            )


class Mobility3DTests(unittest.TestCase):
    def setUp(self):
        self.geometry = PlateGeometry(500e-6, 50e-6, 5e-6)
        self.fluid = Fluid(997, 890e-6)

    def test_symmetry_translation_and_odd_y_counts(self):
        for uniform_x in (False, True):
            g = FluidGrid.cantilever(
                self.geometry,
                nx=3,
                ny=5,
                x_uniform=uniform_x,
            )
            fast = Stokes3D(self.fluid, g, quadrature_backend="gauss")
            full = Stokes3D(
                self.fluid,
                g,
                quadrature_backend="gauss",
                use_symmetry=False,
            )
            omega = 2 * np.pi * 1e4
            a, b = fast.assemble_matrix(omega), full.assemble_matrix(omega)
            np.testing.assert_allclose(a, b, rtol=2e-10, atol=1e-17)
            self.assertTrue(np.isfinite(a).all())
            self.assertLess(fast.integration_report.evaluated_rows, len(g.points))
            self.assertFalse(a.flags.writeable)

    def test_pressure_batches_cache_and_passivity(self):
        g = FluidGrid.cantilever(self.geometry, nx=3, ny=6)
        h = Stokes3D(self.fluid, g, quadrature_backend="gauss")
        omega = 2 * np.pi * 1e4
        v = np.column_stack((np.ones(len(g.points)), 1j * np.ones(len(g.points))))
        p = h.pressure_from_velocity(omega, v)
        matrix = h.assemble_matrix(omega)
        np.testing.assert_allclose(matrix @ p, v, atol=1e-13)
        np.testing.assert_allclose(h.pressure_from_velocity(omega, v[:, 0]), p[:, 0])
        self.assertGreater(np.dot(g.weights, p[:, 0]).real, 0)
        self.assertIs(h.assemble_matrix(omega), matrix)
        h.clear_cache()
        self.assertIsNone(h.integration_report)
        self.assertIsNot(h.assemble_matrix(omega), matrix)
        with self.assertRaises(ValueError):
            h.assemble_matrix(0)

    def test_slender_rigid_force_near_2d_limit(self):
        g = FluidGrid.cantilever(self.geometry, nx=6, ny=16)
        h3 = Stokes3D(self.fluid, g, quadrature_backend="gauss")
        g2 = FluidGrid.chebyshev_gauss(self.geometry, nx=7, ny=16)
        h2 = Stokes2D(self.fluid, g2)
        omega = 2 * np.pi * 1e4
        force3 = np.dot(
            g.weights, h3.pressure_from_velocity(omega, np.ones(len(g.points)))
        )
        force2 = np.dot(
            g2.weights, h2.pressure_from_velocity(omega, np.ones(len(g2.points)))
        )
        self.assertLess(abs(force3 / force2 - 1), 0.1)


if __name__ == "__main__":
    unittest.main()
