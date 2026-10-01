"""Tiny analytic coupled systems, independent of the FEM runtime."""

import unittest
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from mufsi.coupling.operator import CouplingOperator
from mufsi.hydrodynamics.stokes_3d_analytic import (
    Stokes3DAnalytic,
    analytic_fluid_grid,
)
from mufsi.models.fluid import Fluid
from mufsi.models.geometry import PlateGeometry
from mufsi.solvers.frequency_response_analytic import AnalyticFrequencyResponseSolver
from mufsi.solvers.problem import CoupledProblem


class MatrixHandle:
    """Only the serial PETSc matrix interface used by the response solver."""

    def __init__(self, matrix):
        self.matrix = sparse.csr_matrix(matrix)
        self.destroyed = False

    def getValuesCSR(self):
        return self.matrix.indptr, self.matrix.indices, self.matrix.data

    def getSize(self):
        return self.matrix.shape

    def destroy(self):
        self.destroyed = True


class VectorHandle:
    def __init__(self, values):
        self.values, self.destroyed = np.asarray(values), False

    def getArray(self, readonly=True):
        return self.values

    def destroy(self):
        self.destroyed = True


class TinyStructure:
    def __init__(self):
        self.K = np.array([[9.0, 1.0, 0.5], [1.0, 4.0, 0.25], [0.5, 0.25, 6.0]])
        self.M = np.diag([0.05, 0.1, 0.2])
        self.force = np.array([7.0, 1.0 + 0.2j, 0.4])
        self.constrained_dofs = np.array([0])
        self.mesh = SimpleNamespace(comm=SimpleNamespace(size=1))
        self.handles = []

    def stiffness_matrix(self):
        handle = MatrixHandle(self.K)
        self.handles.append(handle)
        return handle

    def mass_matrix(self):
        handle = MatrixHandle(self.M)
        self.handles.append(handle)
        return handle

    def force_vector(self, load):
        handle = VectorHandle(self.force)
        self.handles.append(handle)
        return handle


class AnalyticResponseTests(unittest.TestCase):
    def setUp(self):
        g = analytic_fluid_grid(PlateGeometry(2.0, 0.5, 0.02), nx=2, ny=3)
        self.hydro = Stokes3DAnalytic(Fluid(1.0, 0.8), g, tolerance=1e-9)
        self.structure = TinyStructure()
        self.E = sparse.csr_matrix(
            np.array(
                [
                    [0.0, 0.1, 0.2],
                    [0.2, 0.4, 0.1],
                    [0.1, 0.5, 0.3],
                    [0.3, 0.6, 0.4],
                    [0.2, 0.9, 0.7],
                    [0.1, 0.8, 1.0],
                ]
            )
        )
        self.problem = CoupledProblem(
            self.structure,
            self.hydro,
            CouplingOperator(self.E, g.weights),
        )

    def check_against_joint_system(self, frequencies):
        progress = []
        result = AnalyticFrequencyResponseSolver(
            self.problem, structural_batch_size=1
        ).solve(
            frequencies,
            None,
            progress=lambda i, n: progress.append((i, n)),
        )
        free = np.setdiff1d(np.arange(3), self.structure.constrained_dofs)
        E = self.E[:, free].toarray()
        G = E.T * self.hydro.grid.weights
        for i, f in enumerate(frequencies):
            omega = 2 * np.pi * f
            D = (self.structure.K - omega**2 * self.structure.M)[np.ix_(free, free)]
            B = self.hydro.assemble_matrix(omega)
            joint = np.block([[D, G], [-1j * omega * E, B]])
            reference = np.linalg.solve(
                joint, np.r_[self.structure.force[free], np.zeros(6)]
            )
            np.testing.assert_allclose(
                result.displacement[i, free],
                reference[: len(free)],
                rtol=1e-12,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                result.pressure[i], reference[len(free) :], rtol=1e-12, atol=1e-14
            )
        np.testing.assert_array_equal(
            result.displacement[:, self.structure.constrained_dofs], 0
        )
        self.assertLess(np.max(result.relative_errors), 1e-12)
        self.assertLess(np.max(result.fluid_errors), 1e-12)
        self.assertEqual(
            progress, [(i + 1, len(frequencies)) for i in range(len(frequencies))]
        )
        self.assertTrue(all(handle.destroyed for handle in self.structure.handles))

    def test_full_displacement_and_pressure_against_joint_solve(self):
        self.check_against_joint_system([0.05, 0.2])

    def test_singular_dry_stiffness_joint_fallback(self):
        self.structure.K = np.diag([1.0, 0.0, 2.0])
        self.structure.M = np.zeros((3, 3))
        self.check_against_joint_system([0.1])

    def test_zero_load(self):
        self.structure.force[:] = 0
        result = AnalyticFrequencyResponseSolver(self.problem).solve([0.1], None)
        np.testing.assert_array_equal(result.displacement, 0)
        np.testing.assert_array_equal(result.pressure, 0)
        np.testing.assert_array_equal(result.relative_errors, 0)
        np.testing.assert_array_equal(result.fluid_errors, 0)

    def test_invalid_inputs(self):
        solver = AnalyticFrequencyResponseSolver(self.problem)
        for f in ([], [0.0], [np.nan], [[1.0]]):
            with self.assertRaises(ValueError):
                solver.solve(f, None)
        solver.structural_batch_size = 0
        with self.assertRaises(ValueError):
            solver.solve([0.1], None)
        solver.structural_batch_size = 1
        self.structure.mesh.comm.size = 2
        with self.assertRaises(NotImplementedError):
            solver.solve([0.1], None)
        self.structure.mesh.comm.size = 1
        self.problem.coupling = CouplingOperator(self.E, 2 * self.hydro.grid.weights)
        with self.assertRaises(ValueError):
            solver.solve([0.1], None)


if __name__ == "__main__":
    unittest.main()
