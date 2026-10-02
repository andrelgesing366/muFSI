"""Physical and independent integral checks of the F2D field reconstruction."""

import unittest
from itertools import pairwise

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import kv

from mufsi import (
    Fluid,
    FluidGrid,
    PlateGeometry,
    Stokes2D,
    reconstruct_flow,
    reconstruct_section,
)


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.fluid = Fluid(997, 890e-6)
        self.geometry = PlateGeometry(800e-6, 100e-6, 5e-6)
        self.grid = FluidGrid.chebyshev_gauss(self.geometry, nx=3, ny=32)
        self.hydro = Stokes2D(self.fluid, self.grid)
        self.omega = 2 * np.pi * 30e3
        rng = np.random.default_rng(41)
        self.v = 1e-3 * (
            rng.normal(size=len(self.grid.points))
            + 1j * rng.normal(size=len(self.grid.points))
        )

    def test_surface_no_slip_sign_and_complex_phase(self):
        points = np.column_stack((self.grid.y, np.zeros(self.grid.ny)))
        for omega in (1e-7, 2 * np.pi * 1e3, self.omega, 2 * np.pi * 600e3):
            pressure = self.hydro.pressure_from_velocity(omega, self.v)
            for section in (0, -1):
                field = reconstruct_flow(
                    omega, points, pressure, self.hydro, section_index=section
                )
                np.testing.assert_allclose(field.velocity[:, 0], 0, atol=1e-15)
                np.testing.assert_allclose(
                    field.velocity[:, 1],
                    self.v.reshape(self.grid.nx, self.grid.ny)[section],
                    rtol=2e-7,
                    atol=1e-11,
                )

    def test_independent_regular_boundary_integral(self):
        # Integrate the analytic derivative directly, rather than endpoints.
        points = np.array([[13e-6, 9e-6], [80e-6, -4e-6]])
        p = self.hydro.pressure_from_velocity(self.omega, self.v)[-self.grid.ny :]
        field = reconstruct_flow(self.omega, points, p, self.hydro)
        alpha = np.sqrt(1j * self.omega / self.fluid.kinematic_viscosity)
        for i, (y, z) in enumerate(points):

            def kernel(source, y=y, z=z):
                dy = y - source
                r = np.hypot(dy, z)
                h = (alpha * kv(1, alpha * r) - 1 / r) / (2 * np.pi * alpha**2 * r)
                d = (
                    2 / r**2
                    - alpha**2 * kv(0, alpha * r)
                    - 2 * alpha * kv(1, alpha * r) / r
                ) / (2 * np.pi * alpha**2)
                return np.array([h * dy, d * dy * z / r**2, -h - d * dy**2 / r**2])

            expected = (
                sum(
                    p[j] * quad_vec(kernel, a, b, epsabs=1e-16, epsrel=1e-10)[0]
                    for j, (a, b) in enumerate(
                        zip(self.grid.panel_edges[:-1], self.grid.panel_edges[1:])
                    )
                )
                / self.fluid.dynamic_viscosity
            )
            np.testing.assert_allclose(
                np.r_[field.streamfunction[i], field.velocity[i]],
                expected,
                rtol=2e-9,
                atol=1e-15,
            )

    def test_curl_strain_and_incompressibility_by_finite_difference(self):
        p = self.hydro.pressure_from_velocity(self.omega, self.v)
        point = np.array([[12e-6, 7e-6]])
        field = reconstruct_flow(self.omega, point, p, self.hydro)
        step = 1e-9
        grad = np.empty((2, 2), dtype=complex)
        dpsi = np.empty(2, dtype=complex)
        for axis in range(2):
            offset = np.zeros((1, 2))
            offset[0, axis] = step
            plus = reconstruct_flow(self.omega, point + offset, p, self.hydro)
            minus = reconstruct_flow(self.omega, point - offset, p, self.hydro)
            grad[:, axis] = (plus.velocity[0] - minus.velocity[0]) / (2 * step)
            dpsi[axis] = (plus.streamfunction[0] - minus.streamfunction[0]) / (2 * step)
        np.testing.assert_allclose(field.velocity[0], [dpsi[1], -dpsi[0]], rtol=1e-6)
        np.testing.assert_allclose(field.strain_rate[0], (grad + grad.T) / 2, rtol=1e-6)
        self.assertLess(abs(np.trace(grad)), np.linalg.norm(grad) * 1e-6)

    def test_dissipation_time_average_phase_and_amplitude_scaling(self):
        p = self.hydro.pressure_from_velocity(self.omega, self.v)
        points = np.array([[0, 8e-6], [20e-6, -11e-6]])
        field = reconstruct_flow(self.omega, points, p, self.hydro)
        phases = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        eps = np.real(
            field.strain_rate[None] * np.exp(1j * phases[:, None, None, None])
        )
        power = 2 * self.fluid.dynamic_viscosity * np.sum(eps**2, axis=(-1, -2))
        np.testing.assert_allclose(
            power.mean(axis=0), field.mean_dissipation, rtol=1e-13
        )
        np.testing.assert_allclose(
            field.energy_dissipation, power.mean(axis=0) * 2 * np.pi / self.omega
        )
        scaled = reconstruct_flow(self.omega, points, 2 * np.exp(0.7j) * p, self.hydro)
        np.testing.assert_allclose(scaled.velocity, 2 * np.exp(0.7j) * field.velocity)
        np.testing.assert_allclose(
            scaled.energy_dissipation, 4 * field.energy_dissipation
        )
        self.assertTrue(np.all(field.energy_dissipation >= 0))

    def test_reflection_and_zero_flow(self):
        p = self.hydro.pressure_from_velocity(
            self.omega, np.ones(len(self.grid.points))
        )
        points = np.array([[10e-6, 4e-6], [-10e-6, 4e-6], [10e-6, -4e-6]])
        field = reconstruct_flow(self.omega, points, p, self.hydro)
        np.testing.assert_allclose(
            field.velocity[1], field.velocity[0] * [-1, 1], rtol=1e-10
        )
        np.testing.assert_allclose(
            field.velocity[2], field.velocity[0] * [-1, 1], rtol=1e-10
        )
        np.testing.assert_allclose(
            field.energy_dissipation, field.energy_dissipation[0], rtol=1e-10
        )
        zero = reconstruct_flow(self.omega, points, np.zeros_like(p), self.hydro)
        np.testing.assert_array_equal(zero.velocity, 0)
        np.testing.assert_array_equal(zero.energy_dissipation, 0)

    def test_x_interpolation_and_batching(self):
        p = self.hydro.pressure_from_velocity(self.omega, self.v).reshape(
            self.grid.nx, self.grid.ny
        )
        x = (self.grid.x[0] + self.grid.x[1]) / 2
        points = np.array([[x, 10e-6, 5e-6], [self.grid.x[-1], -10e-6, 7e-6]])
        field = reconstruct_flow(self.omega, points, p, self.hydro, batch_size=1)
        for i, pressure in enumerate(((p[0] + p[1]) / 2, p[-1])):
            expected = reconstruct_flow(
                self.omega, points[i : i + 1, 1:], pressure, self.hydro
            )
            np.testing.assert_allclose(field.velocity[i], expected.velocity[0])
            np.testing.assert_allclose(field.strain_rate[i], expected.strain_rate[0])
        with self.assertRaises(ValueError):
            reconstruct_flow(self.omega, [[-1, 0, 1e-6]], p, self.hydro)

    def test_singular_points_and_validation(self):
        p = np.ones(len(self.grid.points))
        points = [[self.grid.panel_edges[0], 0], [0, 1e-6]]
        field = reconstruct_flow(self.omega, points, p, self.hydro)
        np.testing.assert_array_equal(field.singular_points, [True, False])
        self.assertTrue(np.isnan(field.velocity[0]).all())
        self.assertTrue(np.isfinite(field.velocity[1]).all())
        with self.assertRaises(ValueError):
            reconstruct_flow(self.omega, points, p, self.hydro, singular="raise")
        for options in (
            {"omega": 0},
            {"pressure": p[:-1]},
            {"section_index": 20},
            {"batch_size": 0},
        ):
            args = {
                "omega": self.omega,
                "points": points,
                "pressure": p,
                "hydrodynamics": self.hydro,
            }
            args.update(options)
            with self.assertRaises(ValueError):
                reconstruct_flow(**args)
        with self.assertRaises(NotImplementedError):
            reconstruct_flow(self.omega, points, p, object())

    def test_section_plot(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("Optional Matplotlib is unavailable.")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from mufsi import plot_flow

        p = self.hydro.pressure_from_velocity(self.omega, self.v)
        for pressure in (p, np.zeros_like(p)):
            field = reconstruct_section(
                self.omega,
                np.linspace(-75e-6, 75e-6, 25),
                np.linspace(-20e-6, 20e-6, 20),
                pressure,
                self.hydro,
            )
            fig, axes = plot_flow(field, phase=0.3)
            self.assertEqual(len(axes), 3)
            fig.canvas.draw()
            plt.close(fig)

    def test_volume_dissipation_balances_surface_work(self):
        # A uniform pressure on one panel has only the two physical edge
        # singularities. Integrate both fluid half-spaces to a large radius.
        grid = FluidGrid.midpoint(self.geometry, nx=1, ny=1)
        hydro = Stokes2D(self.fluid, grid)
        omega = 2 * np.pi * 1e4
        width = self.geometry.width
        delta = np.sqrt(2 * self.fluid.kinematic_viscosity / omega)
        pressure = np.array([1 + 0.3j])

        def rule(edges, order=40):
            nodes, weights = np.polynomial.legendre.leggauss(order)
            edges = np.unique(edges)
            nodes = np.concatenate(
                [(a + b) / 2 + nodes * (b - a) / 2 for a, b in pairwise(edges)]
            )
            weights = np.concatenate(
                [weights * (b - a) / 2 for a, b in pairwise(edges)]
            )
            return nodes, weights

        y, wy = rule(
            [
                -8 * width,
                -width,
                -width / 2 - delta,
                -width / 2,
                -width / 2 + delta,
                0,
                width / 2 - delta,
                width / 2,
                width / 2 + delta,
                width,
                8 * width,
            ]
        )
        z, wz = rule([0, delta / 20, delta / 4, delta, 4 * delta, width, 8 * width])
        field = reconstruct_section(omega, y, z, pressure, hydro)
        volume_power = 2 * np.sum(field.mean_dissipation * wz[:, None] * wy[None, :])
        # Direct Bessel primitive at surface quadrature points, independent
        # of the recovery routine and the pressure collocation matrix.
        ys, ws = rule([-width / 2, width / 2], order=200)
        alpha = np.sqrt(1j * omega / self.fluid.kinematic_viscosity)
        offsets = ys[:, None] - grid.panel_edges[None]
        r = abs(offsets)
        derivative = (
            (alpha * kv(1, alpha * r) - 1 / r) * offsets / (2 * np.pi * alpha**2 * r)
        )
        v = (
            np.diff(derivative, axis=1).ravel()
            * pressure[0]
            / self.fluid.dynamic_viscosity
        )
        surface_power = 0.5 * np.sum(ws * np.real(pressure[0] * v.conj()))
        self.assertGreater(surface_power, 0)
        self.assertLess(abs(volume_power / surface_power - 1), 5e-3)


if __name__ == "__main__":
    unittest.main()
