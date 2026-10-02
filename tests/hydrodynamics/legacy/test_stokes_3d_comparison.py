"""Cross-method checks with the actual Quadpy backend, on tiny shared grids."""

import importlib.util
import unittest

import numpy as np

from mufsi.hydrodynamics.legacy.stokes_3d import Stokes3D
from mufsi.hydrodynamics.legacy.stokes_3d_analytic import (
    Stokes3DAnalytic,
    analytic_fluid_grid,
)
from mufsi.models.fluid import Fluid
from mufsi.models.geometry import PlateGeometry


@unittest.skipUnless(importlib.util.find_spec("quadpy"), "Quadpy unavailable")
class CrossMethodTests(unittest.TestCase):
    def test_mobility_pressure_and_generalized_force(self):
        fluid = Fluid(997.0, 890e-6)
        for width in (50e-6, 250e-6):
            g = analytic_fluid_grid(PlateGeometry(500e-6, width, 5e-6), nx=3, ny=5)
            analytic = Stokes3DAnalytic(fluid, g, tolerance=2e-5)
            quadpy = Stokes3D(fluid, g, tolerance=2e-5, quadrature_backend="quadpy")
            x, y = g.points.T
            velocity = np.column_stack(
                (
                    np.ones(len(x)),
                    (x / 500e-6) ** 2,
                    (1 + 0.5j) * (x / 500e-6) ** 2 * 2 * y / width,
                )
            )
            for frequency in (1e3, 1e5):
                with self.subTest(width=width, frequency=frequency):
                    omega = 2 * np.pi * frequency
                    ba, bq = (
                        analytic.assemble_matrix(omega),
                        quadpy.assemble_matrix(omega),
                    )
                    np.testing.assert_allclose(ba, bq, rtol=1e-4, atol=1e-15)
                    pa = analytic.pressure_from_velocity(omega, velocity)
                    pq = quadpy.pressure_from_velocity(omega, velocity)
                    error = np.linalg.norm(pa - pq, axis=0) / np.linalg.norm(pq, axis=0)
                    self.assertLess(np.max(error), 1e-4)
                    ra = velocity.conj().T @ (g.weights[:, None] * pa)
                    rq = velocity.conj().T @ (g.weights[:, None] * pq)
                    self.assertLess(np.linalg.norm(ra - rq) / np.linalg.norm(rq), 1e-4)
                    np.testing.assert_allclose(ba @ pa, velocity, atol=1e-12)
                    np.testing.assert_allclose(bq @ pq, velocity, atol=1e-12)
                    self.assertEqual(
                        analytic.integration_report.evaluated_rows,
                        quadpy.integration_report.evaluated_rows,
                    )


if __name__ == "__main__":
    unittest.main()
