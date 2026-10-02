"""Padded assembly against independently integrated full pressure panels."""

import importlib.util
import unittest

import numpy as np
from scipy import sparse

from mufsi import Fluid, FluidGrid, PlateGeometry
from mufsi.coupling.operator import CouplingOperator
from mufsi.hydrodynamics.legacy.stokes_3d import Stokes3D
from mufsi.hydrodynamics.legacy.stokes_3d_analytic import Stokes3DAnalytic
from mufsi.hydrodynamics.legacy.stokeslet_multigrid import (
    Stokes3DMultigrid,
    multigrid_fluid_grid,
)


class MultigridTests(unittest.TestCase):
    def setUp(self):
        self.geometry = PlateGeometry(500e-6, 50e-6, 5e-6)
        self.fluid = Fluid(997, 890e-6)

    def test_partition_geometry_order_and_force_weights(self):
        grid = multigrid_fluid_grid(
            self.geometry, x_partitions=(7, 3, 3), y_partitions=(3, 3, 3)
        )
        self.assertEqual((grid.nx, grid.ny), (11, 11))
        np.testing.assert_allclose(
            np.diff(grid.x_panel_edges) / (self.geometry.length / 63),
            [9, 9, 9, 9, 9, 9, 3, 3, 1, 1, 1],
        )
        np.testing.assert_allclose(
            np.diff(grid.panel_edges) / (self.geometry.width / 27),
            [1, 1, 1, 3, 3, 9, 3, 3, 1, 1, 1],
        )
        areas = np.prod(
            grid.panel_bounds[:, [1, 3]] - grid.panel_bounds[:, [0, 2]], axis=1
        )
        np.testing.assert_allclose(grid.weights, areas, rtol=1e-14)
        self.assertAlmostEqual(grid.weights.sum() / (500e-6 * 50e-6), 1)
        E = sparse.csr_matrix(np.ones((len(grid.points), 1)))
        coupling = CouplingOperator(E, grid.weights)
        np.testing.assert_allclose(
            coupling.to_structure(np.full(len(grid.points), 2)),
            [2 * self.geometry.length * self.geometry.width],
        )
        p = np.arange(len(grid.points)) + 1j
        u = np.array([1.2 + 0.7j])
        self.assertAlmostEqual(
            np.vdot(u, coupling.to_structure(p)),
            np.vdot(coupling.to_fluid(u), grid.weights * p),
        )

    def test_uniform_and_refined_against_direct_panel_integration(self):
        for xs, ys in (((3,), (5,)), ((3, 3), (3, 3))):
            g = multigrid_fluid_grid(self.geometry, x_partitions=xs, y_partitions=ys)
            m = Stokes3DMultigrid(
                self.fluid, g, quadrature_backend="gauss", tolerance=1e-7
            )
            q = Stokes3D(
                self.fluid,
                g,
                quadrature_backend="gauss",
                tolerance=1e-7,
                use_symmetry=False,
                max_refinements=20,
            )
            velocities = np.column_stack(
                (
                    np.ones(len(g.points)),
                    (g.points[:, 0] / self.geometry.length) ** 2 * (1 + 0.5j),
                )
            )
            for hz in (1e3, 1e5):
                with self.subTest(xs=xs, ys=ys, hz=hz):
                    omega = 2 * np.pi * hz
                    bm, bq = m.assemble_matrix(omega), q.assemble_matrix(omega)
                    self.assertLess(np.linalg.norm(bm - bq) / np.linalg.norm(bq), 2e-7)
                    pm, pq = (
                        m.pressure_from_velocity(omega, velocities),
                        q.pressure_from_velocity(omega, velocities),
                    )
                    self.assertLess(np.linalg.norm(pm - pq) / np.linalg.norm(pq), 2e-7)
                    np.testing.assert_allclose(bm @ pm, velocities, atol=1e-12)
                    self.assertIs(bm, m.assemble_matrix(omega))
                    self.assertFalse(bm.flags.writeable)
                    report = m.integration_report
                    self.assertEqual(
                        report.evaluated_unit_cells, np.prod(report.fine_shape)
                    )
                    self.assertLessEqual(report.max_error_ratio, 1)
            m.clear_cache()
            self.assertIsNone(m.integration_report)
            self.assertIsNone(m._matrix)

    def test_single_panel_and_asymmetric_panel_sizes(self):
        single = FluidGrid.midpoint(self.geometry, nx=1, ny=1)
        m = Stokes3DMultigrid(
            self.fluid, single, quadrature_backend="gauss", tolerance=1e-9
        )
        a = Stokes3DAnalytic(self.fluid, single, tolerance=1e-9)
        np.testing.assert_allclose(
            m.assemble_matrix(2 * np.pi * 1e3),
            a.assemble_matrix(2 * np.pi * 1e3),
            rtol=1e-9,
        )
        xe = np.array([0, 3, 4, 5]) * self.geometry.length / 5
        ye = np.array([0, 1, 4, 5]) * self.geometry.width / 5 - self.geometry.width / 2
        x, y = (xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2
        g = FluidGrid(
            np.column_stack((np.repeat(x, 3), np.tile(y, 3))),
            np.outer(np.diff(xe), np.diff(ye)).ravel(),
            ye,
            3,
            3,
            x_panel_edges=xe,
        )
        m = Stokes3DMultigrid(self.fluid, g, quadrature_backend="gauss", tolerance=1e-8)
        q = Stokes3D(
            self.fluid,
            g,
            quadrature_backend="gauss",
            tolerance=1e-8,
            use_symmetry=False,
            max_refinements=20,
        )
        np.testing.assert_allclose(
            m.assemble_matrix(2 * np.pi * 1e3),
            q.assemble_matrix(2 * np.pi * 1e3),
            rtol=1e-7,
            atol=1e-14,
        )

    @unittest.skipUnless(importlib.util.find_spec("quadpy"), "Quadpy unavailable")
    def test_actual_quadpy_and_analytic_shared_grid(self):
        g = multigrid_fluid_grid(self.geometry, x_partitions=(3,), y_partitions=(3, 3))
        m = Stokes3DMultigrid(self.fluid, g, tolerance=1e-7)
        for reference in (
            Stokes3D(self.fluid, g, tolerance=1e-7, use_symmetry=False),
            Stokes3DAnalytic(self.fluid, g, tolerance=1e-7),
        ):
            omega = 2 * np.pi * 1e3
            np.testing.assert_allclose(
                m.assemble_matrix(omega),
                reference.assemble_matrix(omega),
                rtol=2e-7,
                atol=1e-14,
            )

    def test_invalid_geometry_weights_lattice_and_inputs(self):
        for partitions in ((), (2,), (1,), (-3,), (3.0,), (True,), (3, 4)):
            with self.assertRaises(ValueError):
                multigrid_fluid_grid(self.geometry, x_partitions=partitions)
        g = multigrid_fluid_grid(
            self.geometry, x_partitions=(3, 3), y_partitions=(3, 3)
        )
        bad_weights = FluidGrid(
            g.points,
            2 * g.weights,
            g.panel_edges,
            g.nx,
            g.ny,
            x_panel_edges=g.x_panel_edges,
        )
        with self.assertRaisesRegex(ValueError, "panel areas"):
            Stokes3DMultigrid(self.fluid, bad_weights, quadrature_backend="gauss")
        with self.assertRaisesRegex(ValueError, "max_unit_cells"):
            Stokes3DMultigrid(
                self.fluid, g, max_unit_cells=80, quadrature_backend="gauss"
            )
        with self.assertRaises(ValueError):
            Stokes3DMultigrid(
                self.fluid,
                FluidGrid.cantilever(self.geometry, nx=3, ny=3),
                quadrature_backend="gauss",
            )
        m = Stokes3DMultigrid(self.fluid, g, quadrature_backend="gauss")
        for omega in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                m.assemble_matrix(omega)
        with self.assertRaises(ValueError):
            m.pressure_from_velocity(1, np.ones(3))


if __name__ == "__main__":
    unittest.main()
