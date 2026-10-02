"""Sparse scalar Lagrange basis evaluation at physical points."""

import numpy as np
from scipy import sparse


def point_force_values(function_space, load):
    """Project one or several point forces by virtual work: F = E.T @ forces.

    Returns full serial DOF values. Scalar conversion and displacement boundary
    conditions remain the responsibility of the structural assembler.
    """
    from mufsi.structure.loads import PointLoad, PointLoads

    if not isinstance(load, (PointLoad, PointLoads)):
        raise TypeError("Use PointLoad or PointLoads.")
    loads = (load,) if isinstance(load, PointLoad) else load.loads
    positions = np.asarray([item.position for item in loads], dtype=float)
    amplitudes = np.asarray([item.amplitude for item in loads], dtype=complex)
    dim = function_space.mesh.geometry.dim
    if positions.shape != (len(loads), dim) or not np.isfinite(positions).all():
        raise ValueError(f"Point positions must contain {dim} finite coordinates.")
    if amplitudes.shape != (len(loads),) or not np.isfinite(amplitudes).all():
        raise ValueError("Point amplitudes must be finite scalars.")
    evaluation = build_evaluation_matrix(function_space, positions)
    return np.asarray(evaluation.T @ amplitudes).ravel()


def build_evaluation_matrix(function_space, points):
    """Build CSR E with E[i,j] = phi_j(points[i]) on a serial DOLFINx mesh.

    Supports continuous scalar Lagrange elements with identity DOF transforms.
    Points on cell edges are evaluated in either adjacent cell. Outside points
    raise ValueError. MPI coupling and non-scalar/transformed elements are not
    implemented; the structural eigen solver remains MPI capable.
    """
    from dolfinx import geometry

    V = function_space
    mesh = V.mesh
    if mesh.comm.size != 1:
        raise NotImplementedError("Fluid coupling currently requires one MPI rank.")
    if V.dofmap.bs != 1 or tuple(V.element.value_shape) != ():
        raise ValueError("Basis evaluation requires a scalar function space.")
    element = V.element.basix_element
    if element.discontinuous or not element.dof_transformations_are_identity:
        raise NotImplementedError(
            "Use scalar Lagrange elements with identity DOF transformations (e.g. P2)."
        )
    p = np.asarray(points, dtype=mesh.geometry.x.dtype)
    if p.ndim != 2 or p.shape[1] not in (mesh.geometry.dim, 3):
        raise ValueError("points must have shape (n, geometric_dimension) or (n, 3).")
    if not np.isfinite(p).all():
        raise ValueError("points must be finite.")
    xyz = np.zeros((len(p), 3), dtype=p.dtype)
    xyz[:, : p.shape[1]] = p
    diameter = np.ptp(mesh.geometry.x, axis=0).max()
    tree = geometry.bb_tree(mesh, mesh.topology.dim, padding=1e-12 * diameter)
    candidates = geometry.compute_collisions_points(tree, xyz)
    collisions = geometry.compute_colliding_cells(mesh, candidates, xyz)
    rows, cols, values = [], [], []
    for i, point in enumerate(xyz):
        cells = collisions.links(i)
        if len(cells) == 0:
            raise ValueError(f"Point {i} lies outside the structural mesh.")
        cell = int(cells[0])
        cell_geometry = mesh.geometry.x[mesh.geometry.dofmap[cell]]
        X = mesh.geometry.cmap.pull_back(
            point[None, : mesh.geometry.dim], cell_geometry
        )
        phi = element.tabulate(0, X)[0, 0, :, 0]
        dofs = V.dofmap.cell_dofs(cell)
        rows.extend([i] * len(dofs))
        cols.extend(dofs)
        values.extend(phi)
    return sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(len(p), V.dofmap.index_map.size_local),
    )
