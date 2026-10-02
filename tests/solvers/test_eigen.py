"""Eigenproblem constraints, modal mass, and convergence diagnostics."""

import importlib.util
import unittest

from mufsi import EigenSolver, KirchhoffPlate, Material, PlateGeometry

HAS_FEM = all(importlib.util.find_spec(name) is not None for name in (
    "dolfinx", "petsc4py", "slepc4py", "mpi4py",
))


class EigenInputTests(unittest.TestCase):
    def test_invalid_solver_controls(self):
        for controls in ({"tolerance": 0}, {"tolerance": float("nan")}, {"max_iterations": 0}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                EigenSolver(None, **controls)

    def test_invalid_mode_counts_are_rejected_before_backend_loading(self):
        for n_modes in (0, -1, 1.5, True):
            with self.subTest(n_modes=n_modes), self.assertRaises(ValueError):
                EigenSolver(None).solve(n_modes)


@unittest.skipUnless(HAS_FEM, "Requires DOLFINx/PETSc/SLEPc.")
class EigenFEMTests(unittest.TestCase):
    def test_constraints_normalization_and_orthogonality(self):
        import numpy as np

        plate = KirchhoffPlate(
            PlateGeometry(1.0, 0.7, 0.01), Material(1e7, 1500, 0.3),
            mesh_resolution=(8, 6), boundary_condition="clamped",
        )
        result = EigenSolver(plate).solve(4)
        self.assertEqual(result.modes.shape[1], 4)
        self.assertTrue(np.all(np.diff(result.frequencies) >= 0))
        np.testing.assert_array_equal(result.modes[plate.constrained_dofs, :], 0)
        self.assertLess(float(np.max(result.relative_errors)), 1e-7)
        M = plate.mass_matrix()
        vectors = [M.createVecRight() for _ in range(4)]
        weighted = M.createVecLeft()
        try:
            n_owned = plate.function_space.dofmap.index_map.size_local
            for i, vector in enumerate(vectors):
                vector.array[:] = result.modes[:n_owned, i]
            gram = np.zeros((4, 4), dtype=complex)
            for j, vector in enumerate(vectors):
                M.mult(vector, weighted)
                for i, left in enumerate(vectors):
                    gram[i, j] = left.dot(weighted)
            np.testing.assert_allclose(gram, np.eye(4), atol=1e-8)
        finally:
            weighted.destroy()
            for vector in vectors:
                vector.destroy()
            M.destroy()

    def test_too_many_modes(self):
        plate = KirchhoffPlate(
            PlateGeometry(1.0, 0.7, 0.01), Material(1e7, 1500, 0.3),
            mesh_resolution=(2, 2),
        )
        with self.assertRaises(ValueError):
            EigenSolver(plate).solve(1000)
