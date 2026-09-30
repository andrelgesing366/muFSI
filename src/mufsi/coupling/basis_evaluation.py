"""Sparse structural basis evaluation at arbitrary physical points.

TODO: locate containing cells, pull coordinates back to reference cells,
tabulate the basis with Basix, and assemble E in global structural DOF order.
Specify edge-point tolerances, constrained DOFs, and MPI ownership explicitly.
"""

from typing import Any


def build_evaluation_matrix(function_space: Any, points: Any) -> Any:
    """Build sparse E with E[i, j] = phi_j(points[i]).

    DOLFINx/Basix interactions belong here. No auxiliary fluid FEM mesh is needed.
    """
    raise NotImplementedError("Sparse structural basis evaluation is pending.")
