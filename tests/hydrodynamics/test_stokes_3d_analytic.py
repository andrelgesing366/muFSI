"""Small independent checks; no spectrum examples or performance benchmarks."""

import unittest
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.hydrodynamics.panel_analytic import (
    AnalyticConvergenceError,
    AnalyticPanelIntegrator,
    integrate_panel_zz,
    radial_stokeslet_integral,
)
from mufsi.hydrodynamics.stokes_3d_analytic import (
    Stokes3DAnalytic,
    analytic_fluid_grid,
)
from mufsi.models.fluid import Fluid
from mufsi.models.geometry import PlateGeometry


def reference_kernel(radius, lam):
    """Independent direct point kernel, with its small-argument series."""
    z = lam * np.asarray(radius)
    a = np.empty_like(z, dtype=complex)
    small = abs(z) < 0.1
    coefficients = [2 * (-1) ** n * (n + 1) ** 2 / factorial(n + 2) for n in range(20)]
    a[small] = np.polynomial.polynomial.polyval(z[small], coefficients)
    q = z[~small]
    a[~small] = 2 * np.exp(-q) * (1 + 1 / q + 1 / q**2) - 2 / q**2
    return a / (8 * np.pi * radius)


def direct_rectangle(observation, bounds, lam, order=80):
    """Independent Cartesian tensor quadrature, for nonsingular rectangles."""
    nodes, weights = leggauss(order)
    xlo, xhi, ylo, yhi = bounds
    x = (xlo + xhi) / 2 + (xhi - xlo) / 2 * nodes - observation[0]
    y = (ylo + yhi) / 2 + (yhi - ylo) / 2 * nodes - observation[1]
    radius = np.hypot(x[:, None], y[None, :])
    return (
        np.sum(
            weights[:, None] * weights[None, :] * reference_kernel(radius, lam),
        )
        * (xhi - xlo)
        * (yhi - ylo)
        / 4
    )


class AnalyticPanelTests(unittest.TestCase):
    def test_radial_integral_including_steady_and_thin_intervals(self):
        for lam in (0j, 1e-10 + 1e-10j, 0.3 + 0.3j, 20 + 20j):
            for lower, upper in ((0.0, 1.0), (0.4, 1.0), (2.0, 2.000001)):
                # Integrate A rather than sampling the singular point kernel.
                def integrand(r, lam=lam):
                    if r == 0:
                        return 1 / (8 * np.pi)
                    return reference_kernel(np.asarray(r), lam) * r

                expected = (
                    quad(lambda r: integrand(r).real, lower, upper, epsabs=1e-15)[0]
                    + 1j
                    * quad(
                        lambda r: integrand(r).imag,
                        lower,
                        upper,
                        epsabs=1e-15,
                    )[0]
                )
                np.testing.assert_allclose(
                    radial_stokeslet_integral(lower, upper, lam),
                    expected,
                    rtol=2e-9,
                    atol=2e-16,
                )

    def test_steady_singular_rectangle_exact(self):
        observation = np.array([0.3, 0.7])
        bounds = np.array([[0.0, 1.0, 0.0, 2.0]])
        expected = sum(
            x * np.arcsinh(y / x) + y * np.arcsinh(x / y)
            for x in (0.3, 0.7)
            for y in (0.7, 1.3)
        ) / (8 * np.pi)
        value, error, _, _ = AnalyticPanelIntegrator(rtol=1e-11).integrate(
            observation,
            bounds,
            0j,
        )
        np.testing.assert_allclose(value[0], expected, rtol=1e-11)
        self.assertLess(error[0] / abs(value[0]), 1e-11)

    def test_unsteady_singular_rectangle_duffy_reference(self):
        observation = np.array([0.3, 0.7])
        bounds = np.array([0.0, 1.0, 0.0, 2.0])
        lam = 3 + 3j
        nodes, weights = leggauss(70)
        s, w = (nodes + 1) / 2, weights / 2
        corners = (
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]]) - observation
        )
        expected = 0j
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            edge = (1 - s[:, None]) * a + s[:, None] * b
            radius = s[:, None] * np.linalg.norm(edge, axis=1)[None, :]
            jacobian = abs(a[0] * b[1] - a[1] * b[0]) * s[:, None]
            expected += np.sum(
                w[:, None] * w[None, :] * reference_kernel(radius, lam) * jacobian
            )
        actual, _, _, _ = AnalyticPanelIntegrator(rtol=1e-10).integrate(
            observation,
            bounds[None, :],
            lam,
        )
        np.testing.assert_allclose(actual[0], expected, rtol=2e-10)

    def test_regular_rectangles_all_geometric_regions(self):
        bounds = np.array(
            [
                [0.1, 0.2, 0.4, 2.0],
                [0.4, 2.0, 0.1, 0.2],
                [-2.0, -0.4, 0.1, 0.5],
                [-0.8, -0.1, -0.9, -0.4],
                [0.1, 0.7, -0.9, -0.4],
                [-0.3, 0.7, 0.2, 0.6],
                [0.1, 0.3, -0.3, 0.6],
                [10.0, 10.00001, 0.1, 0.2],
            ]
        )
        integrator = AnalyticPanelIntegrator(rtol=1e-10, batch_size=3)
        for lam in (0j, 1e-9 + 1e-9j, 3 + 3j, 40 + 40j):
            actual, _, _, _ = integrator.integrate([0.0, 0.0], bounds, lam)
            expected = [direct_rectangle([0.0, 0.0], b, lam) for b in bounds]
            np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-14)

    def test_edge_and_corner_observations(self):
        bounds = np.array([[0.0, 1.0, 0.0, 2.0]])
        for observation, widths, heights in (
            ([0.0, 0.0], [1.0], [2.0]),
            ([0.0, 0.7], [1.0], [0.7, 1.3]),
        ):
            expected = sum(
                x * np.arcsinh(y / x) + y * np.arcsinh(x / y)
                for x in widths
                for y in heights
            ) / (8 * np.pi)
            actual, _, _, _ = AnalyticPanelIntegrator(rtol=1e-10).integrate(
                observation,
                bounds,
                0j,
            )
            np.testing.assert_allclose(actual[0], expected, rtol=1e-10)

    def test_validation_convergence_failure_and_physical_units(self):
        fluid = Fluid(997, 890e-6)
        bounds = [0.0, 1e-5, 0.0, 2e-5]
        value = integrate_panel_zz([0.0, 0.0], bounds, 0, fluid, rtol=1e-10)
        expected = 1e-5 * np.arcsinh(2.0) + 2e-5 * np.arcsinh(0.5)
        np.testing.assert_allclose(
            value, expected / (8 * np.pi * fluid.dynamic_viscosity)
        )
        with self.assertRaises(AnalyticConvergenceError):
            AnalyticPanelIntegrator(
                order=2, rtol=1e-14, atol=0, max_refinements=0
            ).integrate(
                [0.0, 0.0],
                np.array([[0.0, 1.0, 0.0, 20.0]]),
                3 + 3j,
            )
        for options in ({"order": True}, {"batch_size": 0}, {"rtol": np.nan}):
            with self.assertRaises((TypeError, ValueError)):
                AnalyticPanelIntegrator(**options)
        with self.assertRaises(ValueError):
            radial_stokeslet_integral(2, 1, 1j)


class AnalyticMobilityTests(unittest.TestCase):
    def setUp(self):
        self.geometry = PlateGeometry(500e-6, 50e-6, 5e-6)
        self.fluid = Fluid(997, 890e-6)
        self.omega = 2 * np.pi * 1e4

    def test_notebook_grid_weights_and_counts(self):
        for nx, ny in ((1, 1), (3, 5), (4, 6)):
            g = analytic_fluid_grid(self.geometry, nx=nx, ny=ny)
            np.testing.assert_allclose(
                g.x, (np.arange(nx) + 0.5) * self.geometry.length / nx
            )
            t = (2 * np.arange(1, ny + 1) - 1) * np.pi / (2 * ny)
            np.testing.assert_allclose(g.y, -np.cos(t) * self.geometry.width / 2)
            expected = (
                self.geometry.length
                / nx
                * self.geometry.width
                / 2
                * np.pi
                / ny
                * np.sin(t)
            )
            np.testing.assert_allclose(g.weights, np.tile(expected, nx))
            self.assertEqual(g.points.shape, (nx * ny, 2))
        g = analytic_fluid_grid(self.geometry, nx=3, ny=5)
        self.assertGreater(np.ptp(np.diff(g.panel_edges)), 0)
        with self.assertRaises(TypeError):
            analytic_fluid_grid(self.geometry, nx=3, ny=5, y_uniform=True)

    def test_blocks_reflection_and_odd_y_against_independent_assembly(self):
        for ny in (1, 5, 6):
            g = analytic_fluid_grid(self.geometry, nx=3, ny=ny)
            fast = Stokes3DAnalytic(self.fluid, g, tolerance=1e-8)
            full = Stokes3DAnalytic(self.fluid, g, tolerance=1e-8, use_symmetry=False)
            a, b = fast.assemble_matrix(self.omega), full.assemble_matrix(self.omega)
            np.testing.assert_allclose(a, b, rtol=2e-10, atol=1e-17)
            self.assertEqual(a.dtype, np.complex128)
            self.assertFalse(a.flags.writeable)
            report = fast.integration_report
            self.assertEqual(report.evaluated_rows, (ny + 1) // 2)
            self.assertEqual(report.evaluated_panels, (ny + 1) // 2 * 3 * ny)
            self.assertLessEqual(report.max_error_ratio, 1)
            self.assertEqual(fast.assemble_blocks(self.omega).shape, (3, ny, ny))

    def test_asymmetric_nonuniform_y_and_shifted_x(self):
        g = analytic_fluid_grid(self.geometry, nx=3, ny=3)
        edges = np.array([-25e-6, -19e-6, 2e-6, 25e-6])
        y = np.array([-22e-6, -3e-6, 14e-6])
        shift = 0.002
        points = np.column_stack((np.repeat(g.x + shift, 3), np.tile(y, 3)))
        custom = FluidGrid(
            points, g.weights, edges, 3, 3, x_panel_edges=g.x_panel_edges + shift
        )
        fast = Stokes3DAnalytic(self.fluid, custom, tolerance=1e-9)
        full = Stokes3DAnalytic(self.fluid, custom, tolerance=1e-9, use_symmetry=False)
        np.testing.assert_allclose(
            fast.assemble_matrix(self.omega),
            full.assemble_matrix(self.omega),
            rtol=1e-10,
        )
        self.assertFalse(fast.integration_report.reflected_y)
        self.assertEqual(fast.integration_report.evaluated_rows, 3)

    def test_block_application_pressure_solve_and_frequency_cache(self):
        g = analytic_fluid_grid(self.geometry, nx=3, ny=5)
        progress = []
        h = Stokes3DAnalytic(
            self.fluid, g, progress=lambda i, n: progress.append((i, n))
        )
        rng = np.random.default_rng(71)
        pressure = rng.normal(size=(15, 2)) + 1j * rng.normal(size=(15, 2))
        velocity = h.apply_mobility(self.omega, pressure)
        self.assertIsNone(h._matrix)
        matrix = h.assemble_matrix(self.omega)
        np.testing.assert_allclose(matrix @ pressure, velocity, atol=1e-17)
        np.testing.assert_allclose(
            h.apply_mobility(self.omega, pressure[:, 0]), velocity[:, 0], atol=1e-17
        )
        np.testing.assert_allclose(
            h.pressure_from_velocity(self.omega, velocity), pressure, atol=1e-12
        )
        self.assertIs(h.assemble_matrix(self.omega), matrix)
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
        rigid = h.pressure_from_velocity(self.omega, np.ones(15))
        self.assertGreater(np.dot(g.weights, rigid).real, 0)
        h.assemble_blocks(2 * self.omega)
        self.assertIsNone(h._matrix)
        self.assertIsNone(h._factorized_omega)
        np.testing.assert_allclose(
            h.assemble_matrix(2 * self.omega)
            @ h.pressure_from_velocity(2 * self.omega, velocity),
            velocity,
            atol=1e-15,
        )
        h.clear_cache()
        self.assertIsNone(h.integration_report)
        self.assertIsNone(h._blocks)

    def test_reject_nonuniform_x_and_invalid_fields(self):
        g = FluidGrid.cantilever(self.geometry, nx=3, ny=3)
        with self.assertRaisesRegex(ValueError, "uniform midpoint"):
            Stokes3DAnalytic(self.fluid, g)
        h = Stokes3DAnalytic(self.fluid, analytic_fluid_grid(self.geometry, nx=1, ny=1))
        for omega in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                h.assemble_matrix(omega)
        for pressure in ([np.nan], [1.0, 2.0], np.ones((1, 1, 1))):
            with self.assertRaises(ValueError):
                h.apply_mobility(self.omega, pressure)
        self.assertTrue(np.isfinite(h.assemble_matrix(0)).all())


if __name__ == "__main__":
    unittest.main()
