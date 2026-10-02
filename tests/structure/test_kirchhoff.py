"""Input checks plus independent plate/beam and assembly checks."""

import importlib.util
import math
import unittest
from dataclasses import FrozenInstanceError, replace

from mufsi import DistributedLoad, EigenSolver, KirchhoffPlate, Material, PlateGeometry

HAS_FEM = all(importlib.util.find_spec(name) is not None for name in (
    "dolfinx", "petsc4py", "slepc4py", "mpi4py",
))


def make_plate(**kwargs):
    parameters = {
        "geometry": PlateGeometry(1.0, 0.7, 0.01),
        "material": Material(1e7, 1500.0, 0.3),
        "mesh_resolution": (8, 6),
    }
    parameters.update(kwargs)
    return KirchhoffPlate(**parameters)


class PlateInputTests(unittest.TestCase):
    def test_invalid_physical_inputs(self):
        plate = make_plate()
        invalid = (
            {"geometry": replace(plate.geometry, length=0)},
            {"geometry": replace(plate.geometry, thickness=-1)},
            {"material": replace(plate.material, density=0)},
            {"material": replace(plate.material, young_modulus=math.inf)},
            {"material": replace(plate.material, poisson_ratio=0.5)},
            {"material": replace(plate.material, poisson_ratio=-1)},
            {"penalty": 0},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(plate, **changes)

    def test_invalid_discretization(self):
        for kwargs in (
            {"mesh_resolution": (0, 6)},
            {"mesh_resolution": (8.5, 6)},
            {"element_degree": 1},
            {"element_degree": True},
            {"boundary_condition": "unknown"},
            {"diagonal": "unknown"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_plate(**kwargs)

    def test_configuration_cannot_invalidate_cached_operators(self):
        with self.assertRaises(FrozenInstanceError):
            make_plate().penalty = 32.0


@unittest.skipUnless(HAS_FEM, "Requires DOLFINx/PETSc/SLEPc.")
class PlateFEMTests(unittest.TestCase):
    def test_simply_supported_analytical_spectrum_and_convergence(self):
        import numpy as np

        coarse = make_plate(boundary_condition="simply_supported")
        fine = replace(coarse, mesh_resolution=(24, 18))
        # Exact homogeneous rectangular plate frequencies: w=sin(mx)sin(ny).
        exact = np.sort([
            math.pi**2 * math.sqrt(fine.bending_rigidity / fine.surface_density)
            * ((m / fine.geometry.length)**2 + (n / fine.geometry.width)**2)
            for m in range(1, 6) for n in range(1, 6)
        ])[:6]
        coarse_result = EigenSolver(coarse).solve(6)
        fine_result = EigenSolver(fine).solve(6)
        coarse_error = np.linalg.norm(coarse_result.angular_frequencies / exact - 1)
        fine_error = np.linalg.norm(fine_result.angular_frequencies / exact - 1)
        self.assertLess(fine_error, coarse_error)
        np.testing.assert_allclose(fine_result.angular_frequencies, exact, rtol=0.03)
        self.assertLess(float(np.max(fine_result.relative_errors)), 1e-7)

    def test_cantilever_and_bridge_against_beam_limit(self):
        # At nu=0 the width-uniform longitudinal mode satisfies the free sides.
        plate = KirchhoffPlate(
            PlateGeometry(1.0, 0.1, 0.01), Material(1e7, 1500, 0),
            mesh_resolution=(32, 4),
        )
        for boundary, beta in (("cantilever", 1.8751040687), ("bridge", 4.7300407449)):
            with self.subTest(boundary=boundary):
                current = replace(plate, boundary_condition=boundary)
                omega = EigenSolver(current).solve(1).angular_frequencies[0]
                exact = beta**2 / plate.geometry.length**2 * math.sqrt(
                    plate.material.young_modulus * plate.geometry.thickness**2
                    / (12 * plate.material.density)
                )
                self.assertLess(abs(omega / exact - 1), 0.02)

    def test_thickness_and_density_scaling(self):
        import numpy as np

        plate = make_plate(boundary_condition="bridge")
        base = EigenSolver(plate).solve(3).angular_frequencies
        thick = replace(plate, geometry=replace(plate.geometry, thickness=0.02))
        dense = replace(plate, material=replace(plate.material, density=6000))
        scale = 500e-6
        small = replace(plate, geometry=PlateGeometry(
            plate.geometry.length * scale,
            plate.geometry.width * scale,
            plate.geometry.thickness * scale,
        ))
        np.testing.assert_allclose(EigenSolver(thick).solve(3).angular_frequencies, 2 * base, rtol=1e-8)
        np.testing.assert_allclose(EigenSolver(dense).solve(3).angular_frequencies, base / 2, rtol=1e-8)
        small_result = EigenSolver(small).solve(3)
        np.testing.assert_allclose(
            small_result.angular_frequencies, base / scale, rtol=1e-8
        )
        self.assertLess(float(np.max(small_result.relative_errors)), 1e-7)

    def test_symmetric_matrices_and_distributed_pressure(self):
        import numpy as np

        plate = make_plate()
        K, M = plate.stiffness_matrix(), plate.mass_matrix()
        ones, expected = M.createVecRight(), M.createVecLeft()
        force = None
        try:
            self.assertTrue(K.isSymmetric(tol=1e-8))
            self.assertTrue(M.isSymmetric(tol=1e-12))
            pressure = 2.0
            force = plate.force_vector(DistributedLoad(lambda x: np.full(x.shape[1], pressure)))
            ones.set(1)
            M.mult(ones, expected)
            expected.scale(pressure / plate.surface_density)
            n_owned = plate.function_space.dofmap.index_map.size_local
            constraints = plate.constrained_dofs
            expected.array[constraints[constraints < n_owned]] = 0
            np.testing.assert_allclose(force.array, expected.array, rtol=1e-12, atol=1e-12)
        finally:
            if force is not None:
                force.destroy()
            expected.destroy()
            ones.destroy()
            M.destroy()
            K.destroy()
