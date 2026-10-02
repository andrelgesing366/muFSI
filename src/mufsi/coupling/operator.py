"""Evaluate structural fields and project resisting fluid traction."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from mufsi.coupling.basis_evaluation import build_evaluation_matrix


@dataclass(frozen=True)
class CouplingOperator:
    """Sparse E and positive weights for work-consistent fluid coupling.

    to_structure returns the positive projection E.T Q p. A coupled equation
    adds this resisting force to its left-hand side. Actual fluid force is
    its negative. E is real so ordinary transpose is also the adjoint.
    """

    evaluation_matrix: sparse.csr_matrix
    weights: np.ndarray

    def __post_init__(self):
        raw = sparse.csr_matrix(self.evaluation_matrix)
        if np.iscomplexobj(raw.data) and np.any(raw.data.imag != 0):
            raise ValueError("The evaluation matrix must be real.")
        E = raw.real.astype(float).copy()
        w = np.array(self.weights, dtype=float, copy=True)
        if w.shape != (E.shape[0],) or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("weights must contain one finite positive area per row.")
        if not np.isfinite(E.data).all():
            raise ValueError("The evaluation matrix must be finite.")
        w.setflags(write=False)
        object.__setattr__(self, "evaluation_matrix", E)
        object.__setattr__(self, "weights", w)

    @classmethod
    def from_structure(cls, structure, grid):
        points = grid.points
        if structure.mesh.geometry.dim == 1:
            points = points[:, :1]
        return cls(
            build_evaluation_matrix(structure.function_space, points),
            grid.weights,
        )

    def to_fluid(self, structural_values):
        """Evaluate displacement or velocity, with optional multiple columns."""
        return self.evaluation_matrix @ structural_values

    def to_structure(self, pressure):
        """Project resisting pressure to structural forces, in N."""
        p = np.asarray(pressure)
        if p.ndim not in (1, 2) or p.shape[0] != len(self.weights):
            raise ValueError("pressure must have shape (npoints[, nrhs]).")
        return self.evaluation_matrix.T @ (
            self.weights * p if p.ndim == 1 else self.weights[:, None] * p
        )
