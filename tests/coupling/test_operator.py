"""Basis transfer and coupled-response validation."""

import importlib.util
import unittest

import numpy as np
from scipy import sparse

from mufsi import (
    CoupledProblem, CouplingOperator, DistributedLoad, Fluid, FluidGrid,
    KirchhoffPlate, Material, PlateGeometry, Stokes2D,
)
from mufsi.coupling.basis_evaluation import build_evaluation_matrix


class TransferTests(unittest.TestCase):
    def test_virtual_work_identity(self):
        rng = np.random.default_rng(4)
        E = sparse.csr_matrix(rng.standard_normal((9, 5)))
        weights = rng.uniform(.1, 1, 9)
        c = CouplingOperator(E, weights)
        u = rng.standard_normal(5) + 1j*rng.standard_normal(5)
        p = rng.standard_normal(9) + 1j*rng.standard_normal(9)
        np.testing.assert_allclose(
            np.vdot(u, c.to_structure(p)), np.vdot(c.to_fluid(u), weights*p),
        )


@unittest.skipUnless(importlib.util.find_spec("dolfinx"), "DOLFINx unavailable")
class CoupledTests(unittest.TestCase):
    def setUp(self):
        from mpi4py import MPI
        if MPI.COMM_WORLD.size != 1:
            self.skipTest("Coupled fluid solve currently supports one rank.")
        self.plate = KirchhoffPlate(
            PlateGeometry(500e-6, 50e-6, 5e-6), Material(169e9, 2330, .3),
            mesh_resolution=(8, 2),
        )
        self.grid = FluidGrid.chebyshev_gauss(self.plate.geometry, nx=5, ny=16)
        self.hydro = Stokes2D(Fluid(997, 890e-6), self.grid)

    def test_basis_reproduces_quadratic_including_boundary(self):
        from dolfinx import fem
        V = self.plate.function_space
        coords = V.tabulate_dof_coordinates()
        values = 1 + 2e3*coords[:, 0] + 1e9*coords[:, 1]**2
        E = build_evaluation_matrix(V, self.grid.points)
        expected = 1 + 2e3*self.grid.points[:, 0] + 1e9*self.grid.points[:, 1]**2
        np.testing.assert_allclose(E @ values, expected, rtol=1e-13)
        u = fem.Function(V)
        u.x.array[:] = values
        from dolfinx import geometry
        xyz = np.zeros((len(self.grid.points), 3))
        xyz[:, :2] = self.grid.points
        tree = geometry.bb_tree(V.mesh, V.mesh.topology.dim)
        cells = geometry.compute_colliding_cells(
            V.mesh, geometry.compute_collisions_points(tree, xyz), xyz,
        )
        owners = np.array([cells.links(i)[0] for i in range(len(xyz))], dtype=np.int32)
        np.testing.assert_allclose(E @ values, u.eval(xyz, owners).ravel(), rtol=1e-13)
        with self.assertRaises(ValueError):
            build_evaluation_matrix(V, [[1., 0.]])

    def test_block_solution_matches_eliminated_dense_reference(self):
        p, h = self.plate, self.hydro
        coupling = CouplingOperator.from_structure(p, self.grid)
        problem = CoupledProblem(p, h, coupling)
        frequencies = np.array([1e3, 1e4])
        load = DistributedLoad(lambda x: np.ones(x.shape[1]))
        result = problem.frequency_response(frequencies, load)
        K, M, F = p.stiffness_matrix(), p.mass_matrix(), p.force_vector(load)
        try:
            k, m = K.convert("dense"), M.convert("dense")
            try:
                kd, md = k.getDenseArray().copy(), m.getDenseArray().copy()
            finally:
                k.destroy()
                m.destroy()
            rhs = F.getArray().copy()
        finally:
            K.destroy()
            M.destroy()
            F.destroy()
        free = np.setdiff1d(np.arange(len(rhs)), p.constrained_dofs)
        E = coupling.evaluation_matrix.toarray()[:, free]
        for i, hz in enumerate(frequencies):
            omega = 2*np.pi*hz
            P = E.T @ (self.grid.weights[:, None] * h.pressure_from_velocity(omega, E))
            D = (kd - omega**2*md)[np.ix_(free, free)] + 1j*omega*P
            expected = np.linalg.solve(D, rhs[free])
            np.testing.assert_allclose(
                result.displacement[i, free], expected, rtol=1e-7, atol=1e-17,
            )
            np.testing.assert_allclose(result.displacement[i, p.constrained_dofs], 0)
            np.testing.assert_allclose(
                result.pressure[i], h.pressure_from_velocity(
                    omega, 1j*omega*coupling.to_fluid(result.displacement[i]),
                ), rtol=1e-8, atol=1e-12,
            )
        self.assertLess(result.relative_errors.max(), 1e-6)
        self.assertLess(result.fluid_errors.max(), 1e-8)

    def test_action_only_model_matches_damped_oscillator(self):
        from types import SimpleNamespace
        from petsc4py import PETSc
        from mufsi.hydrodynamics.base import HydrodynamicModel

        def matrix(diagonal):
            A = PETSc.Mat().createAIJ(
                size=(2, 2), csr=(
                    np.array([0, 1, 2], dtype=PETSc.IntType),
                    np.array([0, 1], dtype=PETSc.IntType),
                    np.array(diagonal, dtype=PETSc.ScalarType),
                ), comm=PETSc.COMM_SELF,
            )
            A.assemble()
            return A

        def force(_):
            v = PETSc.Vec().createSeq(2, comm=PETSc.COMM_SELF)
            v.setValues([0, 1], [0, 1])
            v.assemble()
            return v

        class Drag(HydrodynamicModel):
            def pressure_from_velocity(self, omega, velocity):
                return 3 * velocity

        grid = FluidGrid.midpoint(PlateGeometry(1, 1, 1), nx=1, ny=1)
        hydro = Drag()
        hydro.grid, hydro.fluid = grid, Fluid(1, 1)
        structure = SimpleNamespace(
            mesh=SimpleNamespace(comm=SimpleNamespace(size=1)),
            constrained_dofs=np.array([0]),
            stiffness_matrix=lambda: matrix([1, 10]),
            mass_matrix=lambda: matrix([1, 2]), force_vector=force,
        )
        coupling = CouplingOperator(sparse.csr_matrix([[0., 1.]]), grid.weights)
        result = CoupledProblem(structure, hydro, coupling).frequency_response([.2], None)
        omega = 2*np.pi*.2
        expected = 1 / (10 - 2*omega**2 + 3j*omega)
        np.testing.assert_allclose(result.displacement[0, 1], expected, rtol=1e-13)
        self.assertLess(result.relative_errors[0], 1e-13)
        self.assertTrue(np.isnan(result.fluid_errors[0]))


if __name__ == "__main__":
    unittest.main()
