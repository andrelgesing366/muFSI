"""Kernel regression and physical F2D checks without a FEM backend."""

import unittest

import numpy as np
from scipy import sparse
from scipy.special import keip, kerp, kv

from mufsi import Fluid, FluidGrid, Material, PlateGeometry, SaderMethod, Stokes2D
from mufsi.hydrodynamics.sader import gamma_function
from mufsi.solvers.linear import SciPyLUSolver


class FluidTests(unittest.TestCase):
    def setUp(self):
        self.geometry = PlateGeometry(500e-6, 50e-6, 5e-6)
        self.fluid = Fluid(997, 890e-6)

    def test_quadrature_order_and_convergence(self):
        g = FluidGrid.chebyshev_gauss(self.geometry, nx=8, ny=128)
        self.assertEqual(g.nx, 9)
        self.assertLess(abs(g.weights.sum() / (500e-6 * 50e-6) - 1), 3e-5)
        np.testing.assert_allclose(g.points[:128, 0], 0)
        midpoint = FluidGrid.midpoint(self.geometry, nx=3, ny=5)
        self.assertAlmostEqual(midpoint.weights.sum(), 500e-6 * 50e-6)
        with self.assertRaises(ValueError):
            FluidGrid.chebyshev_gauss(self.geometry, nx=2, ny=8)
        with self.assertRaises(ValueError):
            FluidGrid.midpoint(self.geometry, nx=3, ny=0)

    def test_legacy_kernel_and_blocked_rhs(self):
        g = FluidGrid.chebyshev_gauss(self.geometry, nx=5, ny=24)
        model = Stokes2D(self.fluid, g)
        omega = 2 * np.pi * 1e4
        s = np.sqrt(omega / self.fluid.kinematic_viscosity)
        a = s * (g.panel_edges[None, 1:] - g.y[:, None])
        b = s * (g.panel_edges[None, :-1] - g.y[:, None])

        def primitive(x):
            z = abs(x)
            return np.sign(x) * (1 / z + kerp(z) + 1j * keip(z))

        legacy = (primitive(a) - primitive(b)) / (2j * np.pi * s)
        np.testing.assert_allclose(
            model.section_mobility(omega), legacy / self.fluid.dynamic_viscosity,
            rtol=1e-11, atol=1e-14,
        )
        rng = np.random.default_rng(2)
        v = (
            rng.standard_normal((len(g.points), 3))
            + 1j * rng.standard_normal((len(g.points), 3))
        )
        p = model.pressure_from_velocity(omega, v)
        np.testing.assert_allclose(
            model.assemble_matrix(omega) @ p, v, rtol=1e-10, atol=1e-10,
        )
        np.testing.assert_allclose(
            model.pressure_from_velocity(omega, v[:, 0]), p[:, 0],
        )
        with self.assertRaises(ValueError):
            model.pressure_from_velocity(0, v)

    def test_rigid_section_sader_and_positive_dissipation(self):
        g = FluidGrid.chebyshev_gauss(self.geometry, nx=3, ny=128)
        model = Stokes2D(self.fluid, g)
        for hz in (1e3, 1e4, 1e5):
            omega = 2 * np.pi * hz
            p = model.pressure_from_velocity(omega, np.ones(len(g.points)))
            force_per_length = np.dot(g.weights, p) / self.geometry.length
            re = omega * self.geometry.width**2 / (4 * self.fluid.kinematic_viscosity)
            reference = (
                1j * omega * np.pi * self.fluid.density * self.geometry.width**2 / 4
                * gamma_function(re).conjugate()
            )
            self.assertGreater(force_per_length.real, 0)
            self.assertLess(abs(force_per_length / reference - 1), .01)

    def test_small_argument_is_finite_and_dissipative(self):
        g = FluidGrid.chebyshev_gauss(self.geometry, nx=3, ny=32)
        h = Stokes2D(self.fluid, g)
        p = h.pressure_from_velocity(1e-7, np.ones(len(g.points)))
        self.assertTrue(np.isfinite(p).all())
        self.assertGreater(np.dot(g.weights, p).real, 0)

    def test_dense_and_sparse_lu_complex_rhs(self):
        A = np.array([[3., 1.], [1., 2.]])
        rhs = np.array([1j, 2 + 3j])
        solver = SciPyLUSolver()
        with self.assertRaises(RuntimeError):
            solver.solve(rhs)
        for a in (A, sparse.csr_matrix(A)):
            solver.factorize(a)
            np.testing.assert_allclose(A @ solver.solve(rhs), rhs)


class SaderTests(unittest.TestCase):
    def setUp(self):
        self.sader = SaderMethod(
            PlateGeometry(500e-6, 50e-6, 5e-6), Material(169e9, 2330, .3),
            Fluid(997, 890e-6),
        )

    def test_gamma_re_one_against_paper_coefficients(self):
        root = np.sqrt(1j)
        circular = 1 + 4j * kv(1, -1j * root) / (root * kv(0, -1j * root))
        np.testing.assert_allclose(gamma_function(1), (.91324 - .024134j) * circular)
        root10 = np.sqrt(10j)
        circular10 = 1 + 4j * kv(1, -1j * root10) / (root10 * kv(0, -1j * root10))
        # Eq. (21) at log10(Re)=1: exact decimal sums, including -0.000044510.
        correction10 = .81129415 / .82507965 - 1j * .037185543 / .8388531
        np.testing.assert_allclose(gamma_function(10), correction10 * circular10)
        self.assertTrue(np.isfinite(gamma_function(np.array([1e-6, 1, 1e8]))).all())
        with self.assertRaises(ValueError):
            gamma_function(0)

    def test_static_uniform_pressure_limit(self):
        s = self.sader
        x = np.linspace(0, s.geometry.length, 10)
        expected = (
            s.geometry.width * x**2
            * (6 * s.geometry.length**2 - 4 * s.geometry.length * x + x**2)
            / (24 * s.flexural_rigidity)
        )
        result = s.displacement_per_pressure([1e-6], x)[0]
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-20)
        self.assertEqual(result[0], 0)

    def test_appendix_b4_uniform_response(self):
        s = self.sader
        f = np.array([1000, 10000, 100000])
        x = np.array([.25, .7, 1.0])
        omega = 2 * np.pi * f
        g = s.geometry
        b = (
            omega**2 * g.length**4 / s.flexural_rigidity
            * (s.material.density * g.width * g.thickness
               + np.pi * s.fluid.density * g.width**2 / 4 * s.hydrodynamic_function(f))
        )**.25
        b = b[:, None]
        r = x[None, :]
        numerator = (
            -2 - 2 * np.cos(b) * np.cosh(b) + np.cos(b*r) + np.cosh(b*r)
            + np.cos(b*(1-r))*np.cosh(b) + np.cos(b)*np.cosh(b*(1-r))
            - np.sin(b*(1-r))*np.sinh(b) + np.sin(b)*np.sinh(b*(1-r))
        )
        reference = (
            numerator / (2 * b**4 * (1 + np.cos(b)*np.cosh(b)))
            * g.length**4 / s.flexural_rigidity
        )
        np.testing.assert_allclose(
            s.displacement_per_line_force(f, x*g.length), reference, rtol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
