"""Independent numerical checks for the isolated pressure fitting experiment.

python -m unittest discover -s research -p test_pressure_polynomial.py -v
"""

import unittest
from itertools import pairwise

import numpy as np
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.legendre import leggauss
from pressure_polynomial import (
    Fluid,
    PlateGeometry,
    analytic_fluid_grid,
    areas,
    corner_mask,
    design,
    first_mode,
    fit_pressure,
    forces,
    overlap_average,
    panel_basis,
    point_fit,
    solve_field,
)


class PressurePolynomialChecks(unittest.TestCase):
    def test_basis_panel_means_against_independent_quadrature(self):
        nodes, weights = leggauss(100)
        edges = np.array([-1, -0.99, -0.3, 0.4, 0.98, 1])
        for weighted in (False, True):
            actual = panel_basis(edges, 8, weighted)
            for i, (a, b) in enumerate(pairwise(edges)):
                if weighted:
                    lo, hi = np.arccos(b), np.arccos(a)
                    z = (lo + hi) / 2 + (hi - lo) / 2 * nodes
                    points = np.cos(z)
                    measure = (hi - lo) / (2 * (b - a))
                else:
                    points = (a + b) / 2 + (b - a) / 2 * nodes
                    measure = 0.5
                for n in range(9):
                    coefficients = np.zeros(n + 1)
                    coefficients[n] = 1
                    expected = measure * np.dot(weights, chebval(points, coefficients))
                    self.assertAlmostEqual(actual[i, n], expected, delta=2e-12)
            integrals = np.diff(edges) @ actual
            if weighted:
                np.testing.assert_allclose(integrals, [np.pi] + [0] * 8, atol=2e-15)

    def test_complex_weighted_density_recovery_with_held_out_panels(self):
        geometry = PlateGeometry(250e-6, 25e-6, 1e-6)
        grid = analytic_fluid_grid(geometry, nx=24, ny=24)
        known = np.array([[1 + 2j, 0.1 - 0.2j], [-0.3 + 0.4j, 0.03j], [0.2, 0.1j]])
        # Independent tensor integration of the manufactured continuous density.
        nodes, weights = leggauss(40)
        xangles = np.arccos(2 * grid.x_panel_edges / geometry.length - 1)
        yangles = np.arccos(2 * grid.panel_edges / geometry.width)
        pressure = np.empty((grid.nx, grid.ny), complex)
        for ix in range(grid.nx):
            a, b = xangles[ix + 1], xangles[ix]
            tx = (a + b) / 2 + (b - a) / 2 * nodes
            for iy in range(grid.ny):
                c, d = yangles[iy + 1], yangles[iy]
                ty = (c + d) / 2 + (d - c) / 2 * nodes
                value = np.zeros((40, 40), complex)
                for m in range(3):
                    for k in range(2):
                        value += (
                            known[m, k]
                            * np.cos(m * tx[:, None])
                            * np.cos(2 * k * ty[None, :])
                        )
                dx = 2 * np.diff(grid.x_panel_edges)[ix] / geometry.length
                dy = 2 * np.diff(grid.panel_edges)[iy] / geometry.width
                pressure[ix, iy] = (
                    weights @ value @ weights * (b - a) * (d - c) / (4 * dx * dy)
                )
        for scope in ("all", "omit_corners"):
            fit = fit_pressure(
                grid, geometry, pressure.ravel(), "both", 2, 2, scope, 0.25
            )
            np.testing.assert_allclose(
                fit["coefficients"], known, rtol=1e-12, atol=1e-12
            )
            self.assertLess(fit["weighted_error_holdout"], 1e-12)
            np.testing.assert_allclose(fit["prediction"], pressure.ravel(), rtol=1e-12)
        self.assertTrue(corner_mask(grid, geometry, 0.25).any())

    def test_overlap_restriction_preserves_complex_area_integral(self):
        xe, ye = np.array([0, 0.1, 0.7, 1]), np.array([-1, -0.8, 0.4, 1])
        target_x, target_y = np.linspace(0, 1, 8), np.linspace(-1, 1, 10)
        field = np.arange(9).reshape(3, 3) + 1j * np.arange(9, 18).reshape(3, 3)
        rx, ry = overlap_average(target_x, xe), overlap_average(target_y, ye)
        np.testing.assert_allclose(rx.sum(axis=1), 1)
        np.testing.assert_allclose(ry.sum(axis=1), 1)
        projected = rx @ field @ ry.T
        original_force = np.diff(xe) @ field @ np.diff(ye)
        projected_force = np.diff(target_x) @ projected @ np.diff(target_y)
        self.assertAlmostEqual(abs(projected_force - original_force), 0, delta=1e-13)

    def test_first_mode_and_existing_fluid_solve(self):
        geometry = PlateGeometry(500e-6, 25e-6, 1e-6)
        np.testing.assert_allclose(
            first_mode([0, geometry.length], geometry.length), [0, 1], atol=1e-15
        )
        # Independently check the clamped slope and free moment/shear analytically.
        beta = 1.875104068711961
        self.assertAlmostEqual(np.cosh(beta) * np.cos(beta), -1, delta=1e-14)
        sigma = (np.cosh(beta) + np.cos(beta)) / (np.sinh(beta) + np.sin(beta))
        moment = np.cosh(beta) + np.cos(beta) - sigma * (np.sinh(beta) + np.sin(beta))
        shear = np.sinh(beta) - np.sin(beta) - sigma * (np.cosh(beta) + np.cos(beta))
        self.assertLess(abs(moment), 1e-14)
        self.assertLess(abs(shear), 1e-14)
        h = geometry.length * 1e-5
        self.assertLess(abs(first_mode(h, geometry.length)), 1e-9)
        fluid = Fluid(997, 890e-6)
        omega = fluid.kinematic_viscosity / (geometry.width / 2) ** 2
        field = solve_field(geometry, fluid, 6, 8, omega, 1e-6)
        self.assertLess(field["summary"]["reference_no_slip"], 1e-12)
        self.assertLess(field["summary"]["pressure_y_symmetry"], 1e-12)
        self.assertGreater(field["modal"].real, 0)
        self.assertAlmostEqual(
            np.sum(areas(field["grid"])), geometry.length * geometry.width
        )
        _, force, _ = forces(field["grid"], field["pressure"], geometry)
        np.testing.assert_allclose(force, areas(field["grid"]) @ field["pressure"])
        field["model"].clear_cache()

    def test_point_reconstruction_and_panel_projection_are_distinct(self):
        geometry = PlateGeometry(1, 0.1, 0.001)
        grid = analytic_fluid_grid(geometry, nx=12, ny=12)
        fit = {"family": "transverse", "m": 0, "n": 0, "coefficients": np.ones((1, 1))}
        point_values = point_fit(grid.points, geometry, fit)
        panel_values = design(grid, geometry, "transverse", 0, 0).ravel()
        self.assertGreater(np.max(abs(point_values - panel_values)), 0.1)
        # The singular continuous density has the exact finite integrated force pi*b*L.
        np.testing.assert_allclose(
            areas(grid) @ panel_values, np.pi * geometry.width / 2 * geometry.length
        )


if __name__ == "__main__":
    unittest.main()
