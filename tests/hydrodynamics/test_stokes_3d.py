"""Coefficient mobility, full FE coupling, virtual work and 2D field recovery."""

import importlib.util
import unittest

import numpy as np
from scipy import sparse

from mufsi import (
    BeamGeometry,
    CoupledProblem,
    Fluid,
    FluidGrid,
    FrequencyResponseSolver,
    Material,
    PlateGeometry,
    Stokes3D,
    WeightedCouplingOperator,
    corner_loads,
    energy_from_response,
    reconstruct_flow_from_response,
)
from tests.helpers import TinyStructure


def _check_continuous_coefficient_recovery_and_degrees():
    geometry, fluid = PlateGeometry(2, 1, 0.01), Fluid(1, 1)
    for formulation, degrees in (("EB", [0, 2]), ("KL", [0, 1, 2, 3])):
        hydro = Stokes3D(
            fluid, geometry, formulation=formulation, x_degree=2, y_degree=3
        )
        np.testing.assert_array_equal(hydro.basis.transverse_degrees, degrees)
        H = hydro.assemble_matrix(1.2)
        a = np.arange(1, hydro.coefficient_count + 1) * (1 + 0.3j)
        recovered = hydro.coefficients_from_velocity(1.2, H @ a)
        np.testing.assert_allclose(recovered, a, rtol=1e-11)
        assert hydro.assemble_matrix(1.2) is H
        with unittest.TestCase().assertRaises(ValueError):
            hydro.pressure_from_coefficients(a, [[0, 0]])


def _check_coefficient_schur_against_independent_joint_solve_at_dry_pole():
    hydro = Stokes3D(
        Fluid(1, 1), PlateGeometry(2, 1, 0.01), formulation="EB", x_degree=1, y_degree=0
    )
    x = hydro.collocation_points[:, 0]
    E = sparse.csr_matrix(np.column_stack((np.ones(len(x)), x, x**2)))
    C = np.array([[0, 0], [1.1, -0.2], [0.3, 0.7]])
    structure = TinyStructure()
    coupling = WeightedCouplingOperator(E, C)
    problem = CoupledProblem(structure, hydro, coupling)
    for pole in (False, True):
        if pole:
            structure.K = np.diag([1.0, 0.0, 2.0])
            structure.M = np.zeros((3, 3))
        result = FrequencyResponseSolver(problem).solve([0.2], None)
        omega = 2 * np.pi * 0.2
        H = hydro.assemble_matrix(omega)
        L = np.linalg.lstsq(H, E[:, 1:].toarray(), rcond=None)[0]
        D = (structure.K - omega**2 * structure.M)[1:, 1:]
        joint = np.block([[D, C[1:]], [-1j * omega * L, np.eye(2)]])
        expected = np.linalg.solve(joint, np.r_[structure.force[1:], [0, 0]])
        np.testing.assert_allclose(result.displacement[0, 1:], expected[:2], rtol=1e-11)
        np.testing.assert_allclose(
            result.pressure_coefficients[0], expected[2:], rtol=1e-11
        )
        np.testing.assert_allclose(result.fluid_force[0], C @ expected[2:], rtol=1e-11)
        assert result.relative_errors.max() < 1e-12


def _check_fem_work_balance_and_surface_field(formulation, symmetry):
    from mufsi import EulerBernoulliBeam, KirchhoffPlate
    from mufsi.coupling.weighted import surface_evaluation

    material, fluid = Material(169e9, 2330, 0.3), Fluid(997, 890e-6)
    geometry = PlateGeometry(800e-6, 100e-6, 5e-6)
    structure = (
        EulerBernoulliBeam(
            BeamGeometry(geometry.length, geometry.width, geometry.thickness),
            material,
            mesh_resolution=6,
        )
        if formulation == "EB"
        else KirchhoffPlate(geometry, material, mesh_resolution=(4, 2))
    )
    hydro = Stokes3D(
        fluid, geometry, formulation=formulation, x_degree=4, y_degree=2, tolerance=1e-4
    )
    problem = CoupledProblem(structure, hydro)
    load = corner_loads(structure, symmetry=symmetry)
    response = FrequencyResponseSolver(problem).solve([8000.0], load)
    # Partition of unity and a linear coordinate give independently known
    # weighted Chebyshev force moments, including pressure normalization.
    C = problem.coupling.force_projection
    normalization = geometry.length * geometry.width * np.pi**2 / 4
    expected = np.zeros(hydro.coefficient_count)
    expected[0] = 1
    np.testing.assert_allclose(C.sum(axis=0) / normalization, expected, atol=2e-4)
    coordinates = structure.function_space.tabulate_dof_coordinates()
    expected[0] = geometry.length / 2
    expected[hydro.basis.K + 1] = geometry.length / 4
    np.testing.assert_allclose(
        coordinates[:, 0] @ C / normalization, expected, atol=2e-4 * geometry.length
    )
    assert response.relative_errors.max() < 1e-8
    energy = energy_from_response(structure, response, load, coupling=problem.coupling)
    assert energy.q_factor[0] > 0
    assert energy.work_balance_errors[0] < 1e-8
    if symmetry == "antisymmetric":
        even = np.tile(hydro.basis.transverse_degrees % 2 == 0, hydro.basis.M + 1)
        assert np.linalg.norm(
            response.pressure_coefficients[0, even]
        ) < 1e-6 * np.linalg.norm(response.pressure_coefficients[0])
    grid = FluidGrid.chebyshev_gauss(geometry, nx=5, ny=8)
    points = np.column_stack((grid.y, np.zeros(grid.ny)))
    field = reconstruct_flow_from_response(
        structure, response, fluid, grid, points, section_index=2
    )
    velocity = (
        2j
        * np.pi
        * response.frequencies[0]
        * (
            surface_evaluation(structure, grid.points) @ response.displacement[0]
        ).reshape(grid.nx, grid.ny)[2]
    )
    np.testing.assert_allclose(field.velocity[:, 1], velocity, rtol=1e-10, atol=1e-14)


class WeightedStokesTests(unittest.TestCase):
    def test_coefficients(self):
        _check_continuous_coefficient_recovery_and_degrees()

    def test_joint_algebra(self):
        _check_coefficient_schur_against_independent_joint_solve_at_dry_pole()

    @unittest.skipUnless(importlib.util.find_spec("dolfinx"), "Requires DOLFINx")
    def test_eb_work_and_flow(self):
        _check_fem_work_balance_and_surface_field("EB", "symmetric")

    @unittest.skipUnless(importlib.util.find_spec("dolfinx"), "Requires DOLFINx")
    def test_kl_work_and_flow(self):
        _check_fem_work_balance_and_surface_field("KL", "symmetric")

    @unittest.skipUnless(importlib.util.find_spec("dolfinx"), "Requires DOLFINx")
    def test_kl_odd_work_and_flow(self):
        _check_fem_work_balance_and_surface_field("KL", "antisymmetric")


if __name__ == "__main__":
    unittest.main()
