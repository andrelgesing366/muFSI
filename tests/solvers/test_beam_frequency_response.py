"""Full beam response against the independent uniform-load analytical solution."""

import importlib.util
import unittest
from dataclasses import replace

import numpy as np

from mufsi import (
    BeamFrequencyResponseSolver,
    BeamGeometry,
    DistributedLoad,
    EulerBernoulliBeam,
    Fluid,
    Material,
    SaderMethod,
    SectionForce2D,
)
from mufsi.coupling.basis_evaluation import build_evaluation_matrix


@unittest.skipUnless(importlib.util.find_spec("dolfinx"), "Requires DOLFINx")
class BeamResponseTests(unittest.TestCase):
    def test_fem_sader_matches_full_analytical_field(self):
        b = EulerBernoulliBeam(
            BeamGeometry(800e-6, 50e-6, 10e-6),
            Material(169e9, 2330, 0.3),
            mesh_resolution=24,
            element_degree=3,
        )
        if b.mesh.comm.size != 1:
            self.skipTest("Beam response is serial.")
        fluid = Fluid(997, 890e-6)
        h = SectionForce2D(b.geometry, fluid, method="sader")
        x = np.linspace(0, b.geometry.length, 17)
        E = build_evaluation_matrix(b.function_space, x[:, None])
        frequencies = [1e3, 1e4, 5e4]
        q = 0.001
        r = BeamFrequencyResponseSolver(b, h).solve(
            frequencies,
            DistributedLoad(lambda x: np.full(x.shape[1], q)),
        )
        expected = (
            SaderMethod(b.geometry, b.material, fluid).displacement_per_line_force(
                frequencies,
                x,
            )
            * q
        )
        np.testing.assert_allclose(
            (E @ r.displacement.T).T, expected, rtol=2e-4, atol=1e-14
        )
        np.testing.assert_array_equal(r.displacement[:, b.constrained_dofs], 0)
        self.assertLess(r.relative_errors.max(), 2e-6)
        for i, hz in enumerate(frequencies):
            v = 2j * np.pi * hz * r.displacement[i]
            np.testing.assert_allclose(
                r.line_force[i], h.resisting_force_from_velocity(2 * np.pi * hz, v)
            )
        with self.assertRaises(ValueError):
            BeamFrequencyResponseSolver(
                b,
                SectionForce2D(replace(b.geometry, width=2 * b.geometry.width), fluid),
            )


if __name__ == "__main__":
    unittest.main()
