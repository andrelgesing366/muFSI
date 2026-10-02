"""Coupled algebra at a singular in-vacuo dynamic stiffness."""

import unittest

import numpy as np
from scipy import sparse

from mufsi.solvers.frequency_response import _fluid_schur_solve


class CoupledAlgebraTests(unittest.TestCase):
    def test_vacuum_pole_falls_back_to_joint_solve(self):
        D = sparse.diags([0.0, 2.0], format="csr")
        E = G = sparse.eye(2, format="csr")
        B = np.array([[2 - 1j, 0.1], [0.1, 3 - 2j]])
        force = np.array([1 + 1j, 2.0])
        omega = 1.3
        u, p = _fluid_schur_solve(D, G, E, B, force, omega, k_scale=2)
        joint = np.block([[D.toarray(), G.toarray()], [-1j * omega * E.toarray(), B]])
        expected = np.linalg.solve(joint, np.r_[force, np.zeros(2)])
        np.testing.assert_allclose(np.r_[u, p], expected, rtol=1e-13)
        np.testing.assert_allclose(D @ u + G @ p, force, atol=1e-13)
        np.testing.assert_allclose(B @ p, 1j * omega * (E @ u), atol=1e-13)


if __name__ == "__main__":
    unittest.main()
