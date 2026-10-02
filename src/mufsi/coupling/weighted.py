"""FE evaluation and area-integrated weighted pressure force projection."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from mufsi.coupling.basis_evaluation import build_evaluation_matrix


def surface_evaluation(structure, points):
    """EB fields depend only on x; KL fields depend on both x and y."""
    points = np.asarray(points)
    if structure.mesh.geometry.dim == 1:
        points = points[:, :1]
    return build_evaluation_matrix(structure.function_space, points)


def _angle_rule(vertices, start, length, order):
    coordinates = np.clip(2 * (vertices - start) / length - 1, -1, 1)
    edges = np.unique(np.r_[0, np.arccos(coordinates), np.pi])
    nodes, weights = leggauss(order)
    half = np.diff(edges) / 2
    angles = ((edges[:-1] + edges[1:])[:, None] / 2 + half[:, None] * nodes).ravel()
    return angles, (half[:, None] * weights).ravel()


def _force_projection(structure, basis, order):
    """Cosine-space quadrature on beam intervals or individual plate triangles."""
    g = basis.geometry
    vertices = structure.mesh.geometry.x
    alpha, wa = _angle_rule(vertices[:, 0], 0, g.length, order)
    x = g.length / 2 * (1 + np.cos(alpha))
    if structure.mesh.geometry.dim == 1:
        E = surface_evaluation(structure, np.column_stack((x, np.zeros(len(x)))))
        bx = np.cos(alpha[:, None] * np.arange(basis.M + 1))
        longitudinal = E.T @ (wa[:, None] * bx)
        result = np.zeros((E.shape[1], basis.count))
        # Exactly integrate transverse Chebyshev terms with the endpoint weight.
        result[:, :: basis.K + 1] = np.pi * g.length * g.width / 4 * longitudinal
        return result
    # Integrate each triangle separately, so quadrature never crosses a
    # derivative discontinuity in an FE basis. Vertical triangle slices become
    # cosine-coordinate intervals; endpoint grading smooths square-root limits.
    V, mesh = structure.function_space, structure.mesh
    nodes, weights = leggauss(order)
    t, wt = (nodes + 1) / 2, weights / 2
    result = np.zeros((V.dofmap.index_map.size_local, basis.count))
    for cell in range(mesh.topology.index_map(mesh.topology.dim).size_local):
        cell_geometry = mesh.geometry.x[mesh.geometry.dofmap[cell]]
        xy = cell_geometry[:, :2]
        cuts = np.unique(xy[:, 0])
        for left, right in pairwise(cuts):
            middle = (left + right) / 2
            edges = []
            for i, j in ((0, 1), (1, 2), (2, 0)):
                v0, v1 = xy[i], xy[j]
                if min(v0[0], v1[0]) < middle < max(v0[0], v1[0]):
                    slope = (v1[1] - v0[1]) / (v1[0] - v0[0])
                    edges.append((slope, v0[1] - slope * v0[0]))
            if len(edges) != 2:
                raise ValueError(
                    "Weighted KL projection requires linear triangular geometry."
                )
            lo, hi = np.arccos(
                np.clip(2 * np.array([right, left]) / g.length - 1, -1, 1)
            )
            alpha = lo + (hi - lo) * np.sin(np.pi * t / 2) ** 2
            aw = (hi - lo) * np.pi / 2 * np.sin(np.pi * t) * wt
            xcell = g.length / 2 * (1 + np.cos(alpha))
            limits = np.array([slope * xcell + intercept for slope, intercept in edges])
            ylo, yhi = limits.min(axis=0), limits.max(axis=0)
            blo = np.arccos(np.clip(2 * yhi / g.width, -1, 1))
            bhi = np.arccos(np.clip(2 * ylo / g.width, -1, 1))
            beta = blo[:, None] + (bhi - blo)[:, None] * t
            points = np.column_stack(
                (np.repeat(xcell, order), (g.width / 2 * np.cos(beta)).ravel())
            )
            X = mesh.geometry.cmap.pull_back(points, cell_geometry)
            phi = V.element.basix_element.tabulate(0, X)[0, :, :, 0]
            pressure_values = basis.values(points, remainder=True)
            jac_weights = g.length * g.width / 4 * (aw * (bhi - blo))[:, None] * wt
            contribution = phi.T @ (jac_weights.ravel()[:, None] * pressure_values)
            result[V.dofmap.cell_dofs(cell)] += contribution
    return result


@dataclass(frozen=True)
class WeightedCouplingOperator:
    """E maps DOFs to velocities; C integrates basis traction to FE forces.

    Collocation and force integration are independent. In particular,
    sampled pressure times collocation panel areas does not define C.
    """

    evaluation_matrix: sparse.csr_matrix
    force_projection: np.ndarray
    projection_error: float = 0.0

    def __post_init__(self):
        E = sparse.csr_matrix(self.evaluation_matrix, dtype=float)
        C = np.array(self.force_projection, dtype=float, copy=True)
        if C.ndim != 2 or C.shape[0] != E.shape[1]:
            raise ValueError("Force projection must have one row per structural DOF.")
        if not np.isfinite(E.data).all() or not np.isfinite(C).all():
            raise ValueError("Coupling matrices must be finite.")
        C.setflags(write=False)
        object.__setattr__(self, "evaluation_matrix", E)
        object.__setattr__(self, "force_projection", C)

    @classmethod
    def from_structure(
        cls, structure, hydrodynamics, *, tolerance=1e-3, orders=(6, 10, 16, 24, 36, 52)
    ):
        if not np.isfinite(tolerance) or tolerance <= 0 or len(orders) < 2:
            raise ValueError(
                "Use a positive tolerance and at least two projection orders."
            )
        if any(not isinstance(n, (int, np.integer)) or n < 2 for n in orders):
            raise ValueError("Projection orders must be integers >= 2.")
        if any(a >= b for a, b in pairwise(orders)):
            raise ValueError("Projection orders must increase strictly.")
        a, b = structure.geometry, hydrodynamics.geometry
        if not np.allclose(
            [a.length, a.width, a.thickness],
            [b.length, b.width, b.thickness],
            rtol=1e-12,
            atol=0,
        ):
            raise ValueError("Structure and fluid geometry must match.")
        expected = "EB" if structure.mesh.geometry.dim == 1 else "KL"
        if hydrodynamics.formulation != expected:
            raise ValueError(f"Use formulation='{expected}' for this structure.")
        E = surface_evaluation(structure, hydrodynamics.collocation_points)
        previous = _force_projection(structure, hydrodynamics.basis, orders[0])
        for order in orders[1:]:
            current = _force_projection(structure, hydrodynamics.basis, order)
            error = np.linalg.norm(current - previous) / max(
                np.linalg.norm(current), np.finfo(float).tiny
            )
            if error <= tolerance:
                return cls(E, current, float(error))
            previous = current
        raise RuntimeError(
            f"Weighted force projection did not converge: {error:g}. "
            "Increase projection orders or tolerance explicitly."
        )

    def to_fluid(self, structural_values):
        return self.evaluation_matrix @ structural_values

    def to_structure(self, coefficients):
        return self.force_projection @ coefficients
