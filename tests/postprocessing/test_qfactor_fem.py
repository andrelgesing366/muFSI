"""Corner-load virtual work, parity and both Q estimators with 2D flow."""

import importlib.util
import unittest

import numpy as np

from mufsi import (
    BeamFrequencyResponseSolver,
    BeamGeometry,
    CoupledProblem,
    EulerBernoulliBeam,
    Fluid,
    FluidGrid,
    FrequencyResponseSolver,
    KirchhoffPlate,
    Material,
    PlateGeometry,
    PointLoad,
    SaderMethod,
    SectionForce2D,
    Stokes2D,
    analyze_q_factor,
    corner_displacement,
    corner_loads,
    energy_from_response,
)
from mufsi.coupling.basis_evaluation import build_evaluation_matrix


@unittest.skipUnless(importlib.util.find_spec("dolfinx"), "Requires DOLFINx")
class QFactorFEMTests(unittest.TestCase):
    def setUp(self):
        from mpi4py import MPI

        if MPI.COMM_WORLD.size != 1:
            self.skipTest("Corner-driven coupled responses are serial.")
        self.geometry = BeamGeometry(800e-6, 50e-6, 5e-6)
        self.material, self.fluid = Material(169e9, 2330, 0.3), Fluid(997, 890e-6)
        self.beam = EulerBernoulliBeam(
            self.geometry,
            self.material,
            mesh_resolution=24,
            element_degree=3,
        )
        self.plate = KirchhoffPlate(
            PlateGeometry(800e-6, 50e-6, 5e-6),
            self.material,
            mesh_resolution=(16, 4),
        )
        self.grid = FluidGrid.chebyshev_gauss(self.plate.geometry, nx=17, ny=24)
        self.problem = CoupledProblem(self.plate, Stokes2D(self.fluid, self.grid))
        self.plate_solver = FrequencyResponseSolver(self.problem)

    def test_plate_point_pair_virtual_work_and_parity(self):
        p = self.plate
        x = p.function_space.tabulate_dof_coordinates()
        probes = np.array(
            [
                [0.45 * p.geometry.length, 0.23 * p.geometry.width],
                [0.45 * p.geometry.length, -0.23 * p.geometry.width],
            ]
        )
        E = build_evaluation_matrix(p.function_space, probes)
        for symmetry, sign in (("symmetric", 1), ("antisymmetric", -1)):
            load = corner_loads(p, symmetry=symmetry)
            vector = p.force_vector(load)
            try:
                field = x[:, 0] ** 2 * x[:, 1]
                actual = np.dot(field, vector.getArray())
                expected = sum(
                    item.amplitude * item.position[0] ** 2 * item.position[1]
                    for item in load.loads
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-35)
                np.testing.assert_array_equal(vector.getArray()[p.constrained_dofs], 0)
            finally:
                vector.destroy()
            result = self.plate_solver.solve([2e3], load)
            values = (E @ result.displacement.T).ravel()
            np.testing.assert_allclose(
                values[1], sign * values[0], rtol=2e-6, atol=1e-20
            )
            energy = energy_from_response(
                p, result, load, coupling=self.problem.coupling
            )
            self.assertGreater(energy.q_factor[0], 0)
            self.assertLess(energy.work_balance_errors[0], 1e-6)
            opposite = "antisymmetric" if symmetry == "symmetric" else "symmetric"
            unwanted = corner_displacement(p, result.displacement, symmetry=opposite)
            wanted = corner_displacement(p, result.displacement, symmetry=symmetry)
            self.assertLess(abs(unwanted[0]) / abs(wanted[0]), 1e-6)

    def test_beam_pair_equals_total_force_and_sader_energy(self):
        beam = self.beam
        model = SectionForce2D(beam.geometry, self.fluid, method="sader")
        reference = SaderMethod(beam.geometry, beam.material, self.fluid)
        vacuum = 1.875104068711961**2 / (2 * np.pi * beam.geometry.length**2)
        vacuum *= np.sqrt(beam.flexural_rigidity / beam.line_density)
        loaded, q = reference.resonance_and_q(vacuum)
        load = corner_loads(beam)
        solver = BeamFrequencyResponseSolver(beam, model)
        pair = solver.solve([float(loaded)], load)
        single = solver.solve([float(loaded)], PointLoad(load.loads[0].position, 2e-9))
        np.testing.assert_allclose(pair.displacement, single.displacement, rtol=1e-13)
        energy = energy_from_response(beam, pair, load)
        self.assertLess(energy.work_balance_errors[0], 1e-7)
        self.assertAlmostEqual(energy.q_factor[0] / float(q), 1, delta=0.01)

    def test_complete_sho_energy_workflow_plate_and_beam_tuck(self):
        reference = SaderMethod(self.geometry, self.material, self.fluid)
        vacuum = 1.875104068711961**2 / (2 * np.pi * self.geometry.length**2)
        vacuum *= np.sqrt(self.beam.flexural_rigidity / self.beam.line_density)
        center, _ = reference.resonance_and_q(vacuum)
        f = np.linspace(float(center) * 0.4, float(center) * 1.6, 61)
        beam_solver = BeamFrequencyResponseSolver(
            self.beam,
            SectionForce2D(self.geometry, self.fluid, method="tuck", ny=24),
        )
        for solver in (beam_solver, self.plate_solver):
            result = analyze_q_factor(solver, f)
            self.assertGreater(result.sho.q_factor, 0)
            self.assertGreater(result.energy.q_factor[0], 0)
            self.assertLess(result.sho.relative_error, 0.1)
            self.assertLess(result.energy.work_balance_errors[0], 1e-6)
            self.assertEqual(
                result.resonance_response.frequencies[0], result.sho.resonance_frequency
            )

    def test_antisymmetric_plate_sho_and_energy(self):
        # Locate a width-antisymmetric resonance by a driven coarse sweep;
        # no eigenmode-based forcing or modal truncation is used.
        load = corner_loads(self.plate, symmetry="antisymmetric")
        frequencies = np.geomspace(1e4, 6e5, 81)
        coarse = self.plate_solver.solve(frequencies, load)
        observable = corner_displacement(
            self.plate, coarse.displacement, symmetry="antisymmetric"
        )
        peak = int(np.argmax(abs(observable)))
        self.assertNotIn(peak, (0, len(frequencies) - 1))
        center = frequencies[peak]
        result = analyze_q_factor(
            self.plate_solver,
            np.linspace(center * 0.75, center * 1.25, 81),
            symmetry="antisymmetric",
            fit_background=True,
        )
        self.assertGreater(result.sho.q_factor, 0)
        self.assertGreater(result.energy.q_factor[0], 0)
        self.assertLess(result.sho.relative_error, 0.05)
        self.assertLess(result.energy.work_balance_errors[0], 1e-6)


if __name__ == "__main__":
    unittest.main()
