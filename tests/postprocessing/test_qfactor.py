"""Independent oscillator, harmonic work and numerical energy checks."""

import unittest

import numpy as np
from scipy import sparse
from scipy.integrate import trapezoid

from mufsi import (
    BeamGeometry,
    EulerBernoulliBeam,
    Fluid,
    KirchhoffPlate,
    Material,
    PlateGeometry,
    SaderMethod,
    corner_loads,
    energy_q_factor,
    fit_sho,
    q_factor,
    resonance_frequency,
)


class QFactorTests(unittest.TestCase):
    def test_independent_oscillator_fit_and_energy(self):
        # Generate response from m*u'' + c*u' + k*u = F, not the fit function.
        mass, f0, q = 2e-9, 12345.0, 17.3
        omega0 = 2 * np.pi * f0
        stiffness, damping = mass * omega0**2, mass * omega0 / q
        f = np.linspace(0.8 * f0, 1.2 * f0, 161)
        omega = 2 * np.pi * f
        force = 2e-9 * np.exp(0.7j)
        u = force / (stiffness - mass * omega**2 + 1j * damping * omega)
        fit = fit_sho(f, u)
        self.assertAlmostEqual(fit.resonance_frequency / f0, 1, places=9)
        self.assertAlmostEqual(fit.q_factor / q, 1, places=9)
        self.assertLess(fit.relative_error, 1e-10)
        self.assertAlmostEqual(q_factor(f, u), q, places=8)
        peak = f0 * np.sqrt(1 - 1 / (2 * q**2))
        self.assertAlmostEqual(resonance_frequency(f, u) / peak, 1, places=5)
        at_resonance = force / (1j * damping * omega0)
        energy = energy_q_factor(
            [f0],
            [[at_resonance]],
            [[stiffness]],
            sparse.diags([mass]),
            [force],
        )
        self.assertAlmostEqual(energy.q_factor[0] / q, 1, places=12)
        expected_work = np.pi * damping * omega0 * abs(at_resonance) ** 2
        self.assertAlmostEqual(
            energy.dissipated_energy[0] / expected_work, 1, places=12
        )

    def test_coherent_background_and_scale_invariance(self):
        f0, q = 12000, 35
        f = np.linspace(0.9 * f0, 1.1 * f0, 181)
        omega, omega0 = 2 * np.pi * f, 2 * np.pi * f0
        response = 1 / (omega0**2 - omega**2 + 1j * omega0 * omega / q)
        response += (0.07 + 0.1j) / omega0**2
        for scale in (1e-12, 1, 1e12):
            fit = fit_sho(f, scale * response, fit_background=True)
            self.assertAlmostEqual(fit.resonance_frequency / f0, 1, places=8)
            self.assertAlmostEqual(fit.q_factor / q, 1, places=8)
            self.assertLess(fit.relative_error, 1e-9)

    def test_energy_maximum_matches_time_integral_with_spatial_phase(self):
        K = np.array([[8.0, -1.0], [-1.0, 5.0]])
        M = np.diag([0.1, 0.3])
        u = np.array([[0.2 + 0.8j, -0.3 + 0.1j]])
        f = 0.7
        omega = 2 * np.pi * f
        damping = np.diag([0.4, 0.7])
        F = (K - omega**2 * M + 1j * omega * damping) @ u[0]
        result = energy_q_factor([f], u, K, M, F)
        theta = np.linspace(0, 2 * np.pi, 100001)
        actual_u = np.real(u[0, :, None] * np.exp(1j * theta))
        actual_v = np.real(1j * omega * u[0, :, None] * np.exp(1j * theta))
        energies = np.sum(actual_u * (K @ actual_u), axis=0) / 2
        energies += np.sum(actual_v * (M @ actual_v), axis=0) / 2
        force_t = np.real(F[:, None] * np.exp(1j * theta))
        work = trapezoid(np.sum(force_t * actual_v, axis=0), theta / omega)
        self.assertAlmostEqual(result.stored_energy[0] / max(energies), 1, places=8)
        self.assertAlmostEqual(result.dissipated_energy[0] / work, 1, places=12)
        rotated = energy_q_factor([f], u * np.exp(1.3j), K, M, F * np.exp(1.3j))
        np.testing.assert_allclose(rotated.q_factor, result.q_factor, rtol=1e-13)

    def test_input_and_resolution_errors(self):
        f = np.linspace(90, 110, 21)
        for frequencies, amplitudes in (
            (f[::-1], np.ones(21)),
            (f, np.zeros(21)),
            (f, np.arange(21)),
            (f, np.ones((21, 1))),
        ):
            with self.assertRaises(ValueError):
                fit_sho(frequencies, amplitudes)
        u = 1 / (100**2 - f**2 + 1j * f * 100 / 10000)
        with self.assertRaisesRegex(ValueError, "five samples"):
            fit_sho(f, u)
        with self.assertRaises(ValueError):
            energy_q_factor([1], [[1]], [[1]], [[1]], [1])
        with self.assertRaises(ValueError):
            energy_q_factor([1], [[1, 1j]], [[1, 2], [0, 1]], np.eye(2), [1, 1])

    def test_corner_patterns_and_1d_restriction(self):
        material = Material(169e9, 2330, 0.3)
        plate = KirchhoffPlate(PlateGeometry(800e-6, 50e-6, 5e-6), material)
        for symmetry, sign in (("symmetric", 1), ("antisymmetric", -1)):
            load = corner_loads(plate, symmetry=symmetry, amplitude=2e-9)
            self.assertEqual(load.loads[0].position, (0.999 * 800e-6, 25e-6))
            self.assertEqual(load.loads[1].position, (0.999 * 800e-6, -25e-6))
            self.assertEqual(load.loads[1].amplitude, sign * 2e-9)
        beam = EulerBernoulliBeam(BeamGeometry(800e-6, 50e-6, 5e-6), material)
        load = corner_loads(beam)
        self.assertEqual(load.loads[0].position, load.loads[1].position)
        self.assertEqual(sum(p.amplitude for p in load.loads), 2e-9)
        with self.assertRaisesRegex(ValueError, "no antisymmetric"):
            corner_loads(beam, symmetry="antisymmetric")

    def test_sader_frequency_equilibrium_and_positive_q(self):
        reference = SaderMethod(
            BeamGeometry(800e-6, 50e-6, 5e-6),
            Material(169e9, 2330, 0.3),
            Fluid(997, 890e-6),
        )
        vacuum = np.array([1e4, 6e4])
        loaded, q = reference.resonance_and_q(vacuum)
        self.assertTrue(np.all((loaded > 0) & (loaded < vacuum)))
        self.assertTrue(np.all(q > 0))
        # The real dynamic stiffness of each loaded mode vanishes at f0.
        from mufsi.hydrodynamics.section_force import SectionForce2D

        section = SectionForce2D(reference.geometry, reference.fluid, method="sader")
        line_mass = (
            reference.material.density
            * reference.geometry.width
            * reference.geometry.thickness
        )
        for fv, fl, expected_q in zip(vacuum, loaded, q):
            omega = 2 * np.pi * fl
            k = line_mass * (2 * np.pi * fv) ** 2
            fluid_term = section.dynamic_stiffness(omega)
            self.assertLess(
                abs((k - line_mass * omega**2 + fluid_term).real) / k, 1e-10
            )
            self.assertAlmostEqual(k / fluid_term.imag / expected_q, 1, places=10)


if __name__ == "__main__":
    unittest.main()
