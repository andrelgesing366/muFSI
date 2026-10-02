"""Full even/odd pressure basis and KL FEM modes, confined to research."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from mufsi.hydrodynamics.weighted_pressure import PlateMobility, PlatePressureBasis


class PlateModes:
    """Existing KL eigenmodes, real and peak-DOF normalized, in serial only."""

    def __init__(self, plate, count=3):
        from mufsi import EigenSolver

        if plate.mesh.comm.size != 1:
            raise NotImplementedError("The research fluid coupling is serial.")
        self.plate, self.geometry, self.count = plate, plate.geometry, count
        result = EigenSolver(plate).solve(count)
        if np.max(abs(np.imag(result.modes))) > 1e-8 * np.max(abs(result.modes)):
            raise ValueError("Expected real dry plate modes.")
        self.normalization = np.max(abs(result.modes), axis=0)
        self.dof_modes = result.modes.real / self.normalization
        self.mass = 1 / self.normalization**2
        self.dry_omega = result.angular_frequencies
        self.stiffness = self.mass * self.dry_omega**2
        self.eigen_residuals = result.relative_errors
        # Fix arbitrary eigenvector signs deterministically.
        index = np.argmax(abs(self.dof_modes), axis=0)
        self.dof_modes *= np.sign(self.dof_modes[index, np.arange(count)])

    def values(self, points):
        from mufsi.coupling.basis_evaluation import build_evaluation_matrix

        return (
            build_evaluation_matrix(self.plate.function_space, points) @ self.dof_modes
        )

    def symmetry(self):
        """Measured even/odd reflection errors; no parity is imposed on FEM modes."""
        g = self.geometry
        x = np.linspace(0.03, 0.97, 17) * g.length
        y = np.linspace(0.02, 0.48, 13) * g.width
        points = np.column_stack((np.repeat(x, len(y)), np.tile(y, len(x))))
        reflected = points * [1, -1]
        v, r = self.values(points), self.values(reflected)
        denominator = np.maximum(np.linalg.norm(v, axis=0), 1e-300)
        even = np.linalg.norm(v - r, axis=0) / denominator
        odd = np.linalg.norm(v + r, axis=0) / denominator
        return [
            {
                "parity": "even" if e < o else "odd",
                "even_error": float(e),
                "odd_error": float(o),
            }
            for e, o in zip(even, odd)
        ]


def plate_modal_projection(basis, shape, order=64):
    """C_ij = integral phi_i(x,y) psi_j(x,y) dA; no beam k=0 shortcut.

    Cosine coordinates cancel the singular weight. FEM shapes remain continuous,
    piecewise polynomials, so global quadrature convergence must be checked.
    """
    nodes, weights = leggauss(order)
    angles, weights = np.pi / 2 * (nodes + 1), np.pi / 2 * weights
    g = basis.geometry
    x = g.length / 2 * (1 + np.cos(angles))
    y = g.width / 2 * np.cos(angles)
    points = np.column_stack((np.repeat(x, order), np.tile(y, order)))
    values = np.asarray(shape(points))
    if values.ndim == 1:
        values = values[:, None]
    bx = np.cos(angles[:, None] * np.arange(basis.M + 1))
    by = np.cos(angles[:, None] * basis.transverse_degrees)
    tensor = (bx[:, None, :, None] * by[None, :, None, :]).reshape(
        order**2, basis.count
    )
    return (
        g.length
        * g.width
        / 4
        * values.T
        @ ((weights[:, None] * weights[None, :]).ravel()[:, None] * tensor)
    )


def converged_projection(basis, shape, tolerance=2e-4, orders=(40, 64, 96, 144)):
    previous = plate_modal_projection(basis, shape, orders[0])
    history = []
    for order in orders[1:]:
        current = plate_modal_projection(basis, shape, order)
        # Per-mode normalization avoids hiding a weak odd mode behind an even one.
        errors = np.linalg.norm(current - previous, axis=1) / np.maximum(
            np.linalg.norm(current, axis=1), 1e-300
        )
        history.append({"order": order, "mode_relative_differences": errors.tolist()})
        if np.max(errors) < tolerance:
            return current, {"tolerance": tolerance, "history": history}
        previous = current
    raise RuntimeError(f"Plate force projection did not converge: {history[-1]}")


__all__ = [
    "PlateMobility",
    "PlateModes",
    "PlatePressureBasis",
    "converged_projection",
    "plate_modal_projection",
]
