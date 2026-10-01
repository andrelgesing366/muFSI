"""Independent beam spectra, load units, static compliance and modal mass."""

import importlib.util
import unittest
from dataclasses import FrozenInstanceError, replace

import numpy as np

from mufsi import (
    BeamFrequencyResponseSolver,
    BeamGeometry,
    DistributedLoad,
    EigenSolver,
    EulerBernoulliBeam,
    Material,
    PointLoad,
)
from mufsi.coupling.basis_evaluation import build_evaluation_matrix

HAS_FEM = all(
    importlib.util.find_spec(name) for name in ("dolfinx", "petsc4py", "slepc4py")
)


def beam(**kwargs):
    settings = {
        "geometry": BeamGeometry(2e-3, 100e-6, 10e-6),
        "material": Material(200e9, 2650, 0.3),
        "mesh_resolution": 32,
        "element_degree": 3,
    }
    settings.update(kwargs)
    return EulerBernoulliBeam(**settings)


class BeamInputTests(unittest.TestCase):
    def test_validation_and_immutable_configuration(self):
        b = beam()
        for update in (
            {"mesh_resolution": 0},
            {"mesh_resolution": True},
            {"element_degree": 1},
            {"penalty": 0},
            {"boundary_condition": "free"},
            {"geometry": replace(b.geometry, width=-1)},
            {"material": replace(b.material, density=float("nan"))},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                replace(b, **update)
        with self.assertRaises(FrozenInstanceError):
            b.mesh_resolution = 100


@unittest.skipUnless(HAS_FEM, "Requires DOLFINx/PETSc/SLEPc")
class BeamFEMTests(unittest.TestCase):
    def test_cantilever_spectrum_constraints_and_modal_mass(self):
        b = beam()
        r = EigenSolver(b).solve(6)
        beta = np.array(
            [
                1.875104068711961,
                4.694091132974175,
                7.854757438237612,
                10.99554073487547,
                14.13716839104647,
                17.27875953208824,
            ]
        )
        exact = (
            beta**2
            / (2 * np.pi * b.geometry.length**2)
            * np.sqrt(b.flexural_rigidity / b.line_density)
        )
        np.testing.assert_allclose(r.frequencies, exact, rtol=6e-5)
        np.testing.assert_array_equal(r.modes[b.constrained_dofs], 0)
        self.assertLess(r.relative_errors.max(), 2e-6)
        M = b.mass_matrix()
        vectors = [M.createVecRight() for _ in range(6)]
        weighted = M.createVecLeft()
        try:
            n_owned = b.function_space.dofmap.index_map.size_local
            for i, vector in enumerate(vectors):
                vector.array[:] = r.modes[:n_owned, i]
            gram = np.zeros((6, 6), dtype=complex)
            for j, vector in enumerate(vectors):
                M.mult(vector, weighted)
                for i, left in enumerate(vectors):
                    gram[i, j] = left.dot(weighted)
            np.testing.assert_allclose(gram, np.eye(6), atol=1e-8)
        finally:
            weighted.destroy()
            for vector in vectors:
                vector.destroy()
            M.destroy()

    def test_bridge_and_simply_supported_spectra(self):
        for boundary, beta in (
            (
                "bridge",
                np.array([4.730040744862704, 7.853204624095838, 10.99560783800167]),
            ),
            ("simply_supported", np.arange(1, 4) * np.pi),
        ):
            b = beam(boundary_condition=boundary)
            exact = (
                beta**2
                / (2 * np.pi * b.geometry.length**2)
                * np.sqrt(b.flexural_rigidity / b.line_density)
            )
            r = EigenSolver(b).solve(3)
            np.testing.assert_allclose(r.frequencies, exact, rtol=2e-5)

    def test_quadratic_mesh_convergence_and_physical_scaling(self):
        b = beam(mesh_resolution=12, element_degree=2)
        fine = replace(b, mesh_resolution=48)
        exact = np.array([1.875104068711961, 4.694091132974175, 7.854757438237612]) ** 2
        exact *= np.sqrt(b.flexural_rigidity / b.line_density) / (
            2 * np.pi * b.geometry.length**2
        )
        coarse_f = EigenSolver(b).solve(3).frequencies
        fine_f = EigenSolver(fine).solve(3).frequencies
        self.assertLess(
            np.linalg.norm(fine_f / exact - 1), np.linalg.norm(coarse_f / exact - 1) / 8
        )
        np.testing.assert_allclose(fine_f, exact, rtol=0.004)
        for scaled, factor in (
            (
                replace(
                    b, geometry=replace(b.geometry, thickness=2 * b.geometry.thickness)
                ),
                2,
            ),
            (replace(b, geometry=replace(b.geometry, width=2 * b.geometry.width)), 1),
            (
                replace(
                    b, material=replace(b.material, density=4 * b.material.density)
                ),
                0.5,
            ),
            (replace(b, material=replace(b.material, poisson_ratio=0)), 1),
        ):
            np.testing.assert_allclose(
                EigenSolver(scaled).solve(3).frequencies, factor * coarse_f, rtol=2e-7
            )

    def test_symmetric_operators_and_line_load_units(self):
        b = beam()
        K, M = b.stiffness_matrix(), b.mass_matrix()
        ones, expected = M.createVecRight(), M.createVecLeft()
        force = None
        try:
            self.assertTrue(K.isSymmetric(tol=1e-8))
            self.assertTrue(M.isSymmetric(tol=1e-14))
            q = 0.002
            force = b.force_vector(DistributedLoad(lambda x: np.full(x.shape[1], q)))
            ones.set(1)
            M.mult(ones, expected)
            expected.scale(q / b.line_density)
            n_owned = b.function_space.dofmap.index_map.size_local
            fixed = b.constrained_dofs
            expected.array[fixed[fixed < n_owned]] = 0
            np.testing.assert_allclose(
                force.array, expected.array, rtol=1e-12, atol=1e-20
            )
        finally:
            if force is not None:
                force.destroy()
            ones.destroy()
            expected.destroy()
            K.destroy()
            M.destroy()

    def test_static_distributed_and_tip_point_compliance(self):
        b = beam(mesh_resolution=8, element_degree=4)
        if b.mesh.comm.size != 1:
            self.skipTest("Beam response and arbitrary point loads are serial.")
        x = np.linspace(0, b.geometry.length, 21)
        E = build_evaluation_matrix(b.function_space, x[:, None])
        q, P, L, EI = 0.001, 1e-6, b.geometry.length, b.flexural_rigidity
        for load, exact in (
            (
                DistributedLoad(lambda x: np.full(x.shape[1], q)),
                q * x**2 * (x**2 - 4 * L * x + 6 * L**2) / (24 * EI),
            ),
            (PointLoad((L,), P), P * x**2 * (3 * L - x) / (6 * EI)),
        ):
            r = BeamFrequencyResponseSolver(b).solve([0], load)
            np.testing.assert_allclose(
                (E @ r.displacement.T).ravel(), exact, rtol=1e-7, atol=1e-16
            )
            self.assertLess(r.relative_errors.max(), 1e-7)
            np.testing.assert_array_equal(r.line_force, 0)
        # Interior point force preserves virtual work for representable fields.
        position = 0.413 * L
        force = b.force_vector(PointLoad((position,), P))
        try:
            coordinates = b.function_space.tabulate_dof_coordinates()[:, 0]
            field = coordinates**2
            self.assertAlmostEqual(
                np.dot(field, force.array) / (P * position**2), 1, places=12
            )
        finally:
            force.destroy()


if __name__ == "__main__":
    unittest.main()
