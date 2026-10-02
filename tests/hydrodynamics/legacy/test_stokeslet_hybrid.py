"""Validate the research algorithm against independent cubature and joint solves."""

import unittest

import numpy as np
from scipy import sparse

from mufsi.coupling.operator import CouplingOperator
from mufsi.hydrodynamics.legacy.panel_quadrature import PanelIntegrator
from mufsi.hydrodynamics.legacy.stokes_3d import Stokes3D
from mufsi.hydrodynamics.legacy.stokeslet_multigrid import Stokes3DMultigrid
from mufsi.models.fluid import Fluid
from mufsi.models.geometry import PlateGeometry
from mufsi.solvers.frequency_response import FrequencyResponseSolver
from mufsi.solvers.problem import CoupledProblem
from research.stokeslet_hybrid import (
    HybridPanelIntegrator,
    Stokes3DHybrid,
    Stokes3DHybridMultigrid,
    edge_clustered_grid,
    hierarchical_grid,
    lattice_edge_grid,
)
from tests.helpers import TinyStructure


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.geometry = PlateGeometry(500e-6, 50e-6, 5e-6)
        self.fluid = Fluid(997, 890e-6)

    def test_edge_grid_weights_and_virtual_work(self):
        rng = np.random.default_rng(4)
        for clustering in ("both", "tip", "uniform"):
            grid = edge_clustered_grid(
                self.geometry, nx=8, ny=9, x_clustering=clustering
            )
            areas = np.outer(
                np.diff(grid.x_panel_edges), np.diff(grid.panel_edges)
            ).ravel()
            np.testing.assert_allclose(grid.weights, areas, rtol=1e-15)
            self.assertAlmostEqual(grid.weights.sum() / (500e-6 * 50e-6), 1)
            np.testing.assert_allclose(grid.y, -grid.y[::-1], atol=1e-19)
            E, u, p = rng.normal(size=(72, 4)), rng.normal(size=4), rng.normal(size=72)
            self.assertAlmostEqual(
                u @ (E.T @ (grid.weights * p)), (E @ u) @ (grid.weights * p)
            )

    def test_near_and_fallback_paths_against_tensor_gauss(self):
        # Include long panels, a near edge, and a far panel at high frequency.
        bounds = (
            np.array(
                [[1, 60, -0.5, 0.5], [0.01, 1, 1, 3], [100, 105, 50, 55], [1, 3, 1, 2]]
            )
            * 1e-6
        )
        point = np.zeros(2)
        for hz in (1, 1e3, 400e3):
            lam = np.sqrt(2j * np.pi * hz / self.fluid.kinematic_viscosity)
            reference, _, _, _ = PanelIntegrator(
                "gauss", 1e-9, 1e-18, 24, 65536
            ).regular(point, bounds, lam)
            for ratio in (0, 0.5, 1):
                integrator = HybridPanelIntegrator(
                    "quadpy", 1e-7, 1e-18, near_ratio=ratio
                )
                values, errors, _, _ = integrator.regular(point, bounds, lam)
                np.testing.assert_allclose(values, reference, rtol=1e-6, atol=2e-18)
                self.assertTrue(np.all(errors <= integrator._target(values)))
                if ratio == 0:
                    self.assertGreater(integrator.counts["fallback_radial"], 0)

    def test_shared_edge_grid_pressure_and_symmetry(self):
        grid = edge_clustered_grid(self.geometry, nx=4, ny=5)
        velocity = np.column_stack((np.ones(20), (grid.points[:, 0] / 500e-6) ** 2))
        for hz in (1, 1e3, 400e3):
            omega = 2 * np.pi * hz
            hybrid = Stokes3DHybrid(self.fluid, grid, tolerance=1e-7)
            independent = Stokes3D(
                self.fluid,
                grid,
                tolerance=1e-8,
                quadrature_backend="gauss",
                max_refinements=24,
                max_subpanels=65536,
            )
            reference = independent.assemble_matrix(omega)
            np.testing.assert_allclose(
                hybrid.assemble_matrix(omega), reference, rtol=2e-6, atol=3e-12
            )
            np.testing.assert_allclose(
                hybrid.pressure_from_velocity(omega, velocity),
                independent.pressure_from_velocity(omega, velocity),
                rtol=3e-6,
                atol=1e-5,
            )
            plain = Stokes3DHybrid(self.fluid, grid, tolerance=1e-7, use_symmetry=False)
            np.testing.assert_allclose(
                plain.assemble_matrix(omega), hybrid._matrix, rtol=1e-12, atol=1e-14
            )
            self.assertFalse(hybrid._matrix.flags.writeable)
            self.assertIs(hybrid.assemble_matrix(omega), hybrid._matrix)

    def test_signed_prefix_reuse_matches_direct_and_original_multigrid(self):
        for both in (True, False):
            grid = hierarchical_grid(
                self.geometry,
                x_partitions=(3, 3),
                y_partitions=(3, 3),
                both_x_edges=both,
            )
            for hz in (1e3, 400e3):
                omega = 2 * np.pi * hz
                reusable = Stokes3DHybridMultigrid(self.fluid, grid, tolerance=1e-7)
                direct = Stokes3DHybrid(
                    self.fluid,
                    grid,
                    tolerance=1e-8,
                    analytic_only=True,
                    quadrature_backend="gauss",
                )
                original = Stokes3DMultigrid(
                    self.fluid,
                    grid,
                    tolerance=1e-7,
                    max_refinements=24,
                    max_subpanels=65536,
                )
                np.testing.assert_allclose(
                    reusable.assemble_matrix(omega),
                    direct.assemble_matrix(omega),
                    rtol=2e-6,
                    atol=3e-12,
                )
                np.testing.assert_allclose(
                    reusable._matrix,
                    original.assemble_matrix(omega),
                    rtol=2e-6,
                    atol=3e-12,
                )
                self.assertLessEqual(reusable.hybrid_report["max_error_ratio"], 1)

    def test_full_coupling_against_independent_joint_system(self):
        grid = hierarchical_grid(self.geometry, x_partitions=(3,), y_partitions=(3,))
        structure = TinyStructure()
        E = sparse.csr_matrix(np.random.default_rng(7).uniform(0.1, 1, size=(9, 3)))
        for cls in (Stokes3DHybrid, Stokes3DHybridMultigrid):
            hydro = cls(self.fluid, grid, tolerance=1e-8)
            coupling = CouplingOperator(E, grid.weights)
            result = FrequencyResponseSolver(
                CoupledProblem(structure, hydro, coupling)
            ).solve([10, 100], None)
            free = np.array([1, 2])
            e, g = E[:, free].toarray(), E[:, free].T.toarray() * grid.weights
            for i, hz in enumerate(result.frequencies):
                omega = 2 * np.pi * hz
                D = (structure.K - omega**2 * structure.M)[np.ix_(free, free)]
                joint = np.block(
                    [[D, g], [-1j * omega * e, hydro.assemble_matrix(omega)]]
                )
                reference = np.linalg.solve(
                    joint, np.r_[structure.force[free], np.zeros(9)]
                )
                np.testing.assert_allclose(
                    result.displacement[i, free], reference[:2], rtol=1e-11, atol=1e-14
                )
                np.testing.assert_allclose(
                    result.pressure[i], reference[2:], rtol=1e-11, atol=1e-10
                )

    def test_lattice_cosine_grid_and_prefix_precision_fallback(self):
        grid = lattice_edge_grid(self.geometry, nx=8, ny=12)
        self.assertEqual((grid.nx, grid.ny), (8, 12))
        np.testing.assert_allclose(grid.y, -grid.y[::-1], atol=1e-19)
        omega = 2 * np.pi * 400e3
        reuse = Stokes3DHybridMultigrid(
            self.fluid, grid, tolerance=1e-6, absolute_tolerance=0
        )
        radial = Stokes3DHybrid(
            self.fluid,
            grid,
            tolerance=1e-7,
            absolute_tolerance=0,
            analytic_only=True,
            quadrature_backend="gauss",
        )
        np.testing.assert_allclose(
            reuse.assemble_matrix(omega),
            radial.assemble_matrix(omega),
            rtol=2e-6,
            atol=2e-14,
        )
        self.assertGreater(reuse.hybrid_report["prefix_recomputed_entries"], 0)
        with self.assertRaises(ValueError):
            lattice_edge_grid(self.geometry, nx=24, ny=36, max_unit_cells=100)

    def test_invalid_controls_and_lattice(self):
        for n in (0, True, 1.5):
            with self.assertRaises(ValueError):
                edge_clustered_grid(self.geometry, nx=n, ny=3)
        with self.assertRaises(ValueError):
            HybridPanelIntegrator(near_ratio=-1)
        with self.assertRaises(ValueError):
            Stokes3DHybridMultigrid(
                self.fluid, edge_clustered_grid(self.geometry, nx=4, ny=5)
            )
        grid = hierarchical_grid(self.geometry)
        model = Stokes3DHybridMultigrid(self.fluid, grid)
        for omega in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                model.assemble_matrix(omega)

    def test_peak_measurement_against_exact_sho(self):
        from benchmarks.legacy.benchmark_stokeslet_hybrid import peak_metrics

        f0, q = 12345.0, 4.0
        f = np.linspace(0.4 * f0, 1.5 * f0, 501)
        response = f0**2 / (f0**2 - f**2 + 1j * f * f0 / q)
        measured = peak_metrics(f, response, [f[0], f[-1]])
        peak_square = 1 - 1 / (2 * q**2)
        minimum = 1 / q**2 - 1 / (4 * q**4)
        exact_peak = f0 * np.sqrt(peak_square)
        exact_width = f0 * (
            np.sqrt(peak_square + np.sqrt(minimum))
            - np.sqrt(peak_square - np.sqrt(minimum))
        )
        self.assertLess(abs(measured["frequency"] / exact_peak - 1), 1e-5)
        self.assertLess(abs(measured["linewidth"] / exact_width - 1), 1e-5)
        self.assertLess(abs(measured["amplitude"] * np.sqrt(minimum) - 1), 1e-5)


if __name__ == "__main__":
    unittest.main()
