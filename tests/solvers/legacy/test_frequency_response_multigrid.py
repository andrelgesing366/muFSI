"""Legacy multigrid response and dry-pole fallback with panel-area coupling."""

import unittest

import numpy as np
from scipy import sparse

from mufsi import (
    CoupledProblem,
    CouplingOperator,
    Fluid,
    FrequencyResponseSolver,
    PlateGeometry,
)
from mufsi.hydrodynamics.legacy.stokeslet_multigrid import (
    Stokes3DMultigrid,
    multigrid_fluid_grid,
)
from tests.helpers import TinyStructure


class MultigridResponseTests(unittest.TestCase):
    def test_full_response_against_independent_joint_system(self):
        geometry = PlateGeometry(2, 0.5, 0.02)
        grid = multigrid_fluid_grid(geometry, x_partitions=(3, 3), y_partitions=(3, 3))
        hydro = Stokes3DMultigrid(
            Fluid(1, 0.8), grid, quadrature_backend="gauss", tolerance=1e-9
        )
        x, y = grid.points.T
        E = sparse.csr_matrix(np.column_stack((np.ones(len(x)), x**2, x * y)))
        structure = TinyStructure()
        problem = CoupledProblem(structure, hydro, CouplingOperator(E, grid.weights))
        for pole in (False, True):
            if pole:
                structure.K = np.diag([1.0, 0.0, 2.0])
                structure.M = np.zeros((3, 3))
            frequencies = [0.05, 0.2]
            result = FrequencyResponseSolver(problem).solve(frequencies, None)
            free = np.array([1, 2])
            ef = E[:, free].toarray()
            gf = ef.T * grid.weights
            for i, hz in enumerate(frequencies):
                omega = 2 * np.pi * hz
                D = (structure.K - omega**2 * structure.M)[np.ix_(free, free)]
                B = hydro.assemble_matrix(omega)
                expected = np.linalg.solve(
                    np.block([[D, gf], [-1j * omega * ef, B]]),
                    np.r_[structure.force[free], np.zeros(len(x))],
                )
                np.testing.assert_allclose(
                    result.displacement[i, free], expected[:2], rtol=1e-12
                )
                np.testing.assert_allclose(
                    result.pressure[i], expected[2:], rtol=1e-11, atol=1e-13
                )
            np.testing.assert_array_equal(result.displacement[:, 0], 0)
            self.assertLess(result.relative_errors.max(), 1e-12)
            self.assertLess(result.fluid_errors.max(), 1e-12)
        self.assertTrue(all(handle.destroyed for handle in structure.handles))


if __name__ == "__main__":
    unittest.main()
