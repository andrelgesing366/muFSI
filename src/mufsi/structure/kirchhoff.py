"""Isotropic Kirchhoff–Love plates using DOLFINx C0 interior penalty.

Displacement is constrained strongly; clamped-edge normal slope is imposed
weakly. Unmarked edges retain natural free moment/shear conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from math import isfinite
from numbers import Integral
from typing import Any

from mufsi.models.geometry import PlateGeometry
from mufsi.models.material import Material
from mufsi.structure.base import StructuralModel
from mufsi.structure.loads import DistributedLoad, Load


def _backend() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Load the matched scientific environment only when FEM is requested."""
    try:
        import numpy as np
        import ufl
        from dolfinx import fem, mesh
        from mpi4py import MPI
        from petsc4py import PETSc
    except ImportError as error:
        raise ImportError(
            "KirchhoffPlate requires FEniCSx/DOLFINx, UFL, mpi4py, and petsc4py "
            "from a compatible environment. See docs/plate_eigenproblem.md."
        ) from error
    return np, ufl, fem, mesh, MPI, PETSc


@dataclass(frozen=True)
class KirchhoffPlate(StructuralModel):
    """Homogeneous isotropic rectangular plate in physical SI coordinates.

    The domain is [0, length] x [-width/2, width/2], matching the legacy model.
    cantilever clamps x=0; bridge clamps both x ends; clamped clamps all edges;
    simply_supported constrains displacement on all edges, leaving slope free.

    Setup is lazy and cached. Configuration is immutable so changing parameters
    requires a new plate. Matrix calls return new caller-owned PETSc matrices
    in the full DOF space, before displacement elimination.
    """

    geometry: PlateGeometry
    material: Material
    mesh_resolution: tuple[int, int] = (64, 32)
    boundary_condition: str = "cantilever"
    element_degree: int = 2
    penalty: float = 16.0
    diagonal: str = "crossed"
    comm: Any = field(default=None, repr=False, compare=False, kw_only=True)

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.geometry.length),
            ("width", self.geometry.width),
            ("thickness", self.geometry.thickness),
            ("young_modulus", self.material.young_modulus),
            ("density", self.material.density),
            ("penalty", self.penalty),
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        nu = self.material.poisson_ratio
        if not isfinite(nu) or not -1.0 < nu < 0.5:
            raise ValueError("poisson_ratio must lie strictly between -1 and 0.5.")
        if len(self.mesh_resolution) != 2 or any(
            isinstance(n, bool) or not isinstance(n, Integral) or n < 1
            for n in self.mesh_resolution
        ):
            raise ValueError("mesh_resolution must contain two positive integers.")
        if (
            isinstance(self.element_degree, bool)
            or not isinstance(self.element_degree, Integral)
            or self.element_degree < 2
        ):
            raise ValueError("C0 interior penalty requires element_degree >= 2.")
        if self.boundary_condition not in {
            "cantilever", "bridge", "clamped", "simply_supported"
        }:
            raise ValueError("Unknown plate boundary_condition.")
        if self.diagonal not in {
            "crossed", "left", "right", "left_right", "right_left"
        }:
            raise ValueError("Unknown triangular mesh diagonal pattern.")

    @property
    def bending_rigidity(self) -> float:
        """Return D = E t^3 / (12 (1 - nu^2)), in N m."""
        return (
            self.material.young_modulus * self.geometry.thickness**3
            / (12.0 * (1.0 - self.material.poisson_ratio**2))
        )

    @property
    def surface_density(self) -> float:
        """Return rho * t, in kg/m^2."""
        return self.material.density * self.geometry.thickness

    @cached_property
    def mesh(self) -> Any:
        """Create a triangular rectangle with facet ghosts for interior terms."""
        np, _, _, mesh, MPI, _ = _backend()
        g = self.geometry
        return mesh.create_rectangle(
            MPI.COMM_WORLD if self.comm is None else self.comm,
            np.array([[0.0, -g.width / 2], [g.length, g.width / 2]]),
            tuple(int(n) for n in self.mesh_resolution),
            cell_type=mesh.CellType.triangle,
            ghost_mode=mesh.GhostMode.shared_facet,
            diagonal=getattr(mesh.DiagonalType, self.diagonal),
        )

    @cached_property
    def function_space(self) -> Any:
        """Continuous scalar Lagrange space of degree at least two."""
        _, _, fem, _, _, _ = _backend()
        return fem.functionspace(self.mesh, ("Lagrange", int(self.element_degree)))

    @cached_property
    def boundary_tags(self) -> Any:
        """Facet tag 1 marks edges with a zero-displacement constraint."""
        np, _, _, mesh, _, _ = _backend()
        g = self.geometry
        atol = 1e-12 * max(g.length, g.width)

        def supported(x: Any) -> Any:
            left = np.isclose(x[0], 0.0, atol=atol, rtol=0.0)
            right = np.isclose(x[0], g.length, atol=atol, rtol=0.0)
            if self.boundary_condition == "cantilever":
                return left
            if self.boundary_condition == "bridge":
                return left | right
            return (
                left | right
                | np.isclose(x[1], -g.width / 2, atol=atol, rtol=0.0)
                | np.isclose(x[1], g.width / 2, atol=atol, rtol=0.0)
            )

        fdim = self.mesh.topology.dim - 1
        self.mesh.topology.create_connectivity(fdim, self.mesh.topology.dim)
        facets = np.sort(mesh.locate_entities_boundary(self.mesh, fdim, supported))
        return mesh.meshtags(
            self.mesh, fdim, facets, np.ones(facets.size, dtype=np.int32)
        )

    @cached_property
    def constrained_dofs(self) -> Any:
        """Local supported displacement indices, including ghost DOFs."""
        _, _, fem, _, _, _ = _backend()
        return fem.locate_dofs_topological(
            self.function_space, self.mesh.topology.dim - 1,
            self.boundary_tags.find(1),
        )

    @cached_property
    def boundary_conditions(self) -> tuple[Any, ...]:
        """Homogeneous displacement Dirichlet condition; slope is in the form."""
        _, _, fem, _, _, PETSc = _backend()
        return (fem.dirichletbc(
            PETSc.ScalarType(0), self.constrained_dofs, self.function_space
        ),)

    @cached_property
    def stiffness_form(self) -> Any:
        """Compile bending, symmetric consistency, and slope-penalty terms."""
        _, ufl, fem, _, _, _ = _backend()
        u = ufl.TrialFunction(self.function_space)
        v = ufl.TestFunction(self.function_space)
        n = ufl.FacetNormal(self.mesh)
        h = ufl.CellDiameter(self.mesh)
        dx = ufl.Measure("dx", domain=self.mesh)
        dS = ufl.Measure("dS", domain=self.mesh)
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.boundary_tags)
        D, nu = self.bending_rigidity, self.material.poisson_ratio

        def moment(w: Any) -> Any:
            H = ufl.grad(ufl.grad(w))
            return D * ((1 - nu) * H + nu * ufl.tr(H) * ufl.Identity(2))

        def moment_nn(w: Any) -> Any:
            return ufl.dot(n, ufl.dot(moment(w), n))

        jump_u = ufl.jump(ufl.grad(u), n)
        jump_v = ufl.jump(ufl.grad(v), n)
        a = (
            ufl.inner(moment(u), ufl.grad(ufl.grad(v))) * dx
            - ufl.inner(ufl.avg(moment_nn(u)), jump_v) * dS
            - ufl.inner(jump_u, ufl.avg(moment_nn(v))) * dS
            + self.penalty * D / ufl.avg(h) * ufl.inner(jump_u, jump_v) * dS
        )
        if self.boundary_condition != "simply_supported":
            slope_u = ufl.dot(ufl.grad(u), n)
            slope_v = ufl.dot(ufl.grad(v), n)
            a += (
                -ufl.inner(moment_nn(u), slope_v)
                - ufl.inner(slope_u, moment_nn(v))
                + self.penalty * D / h * ufl.inner(slope_u, slope_v)
            ) * ds(1)
        return fem.form(a)

    @cached_property
    def mass_form(self) -> Any:
        """Compile the consistent translational mass form rho t u v."""
        _, ufl, fem, _, _, _ = _backend()
        u = ufl.TrialFunction(self.function_space)
        v = ufl.TestFunction(self.function_space)
        return fem.form(
            self.surface_density * ufl.inner(u, v)
            * ufl.Measure("dx", domain=self.mesh)
        )

    def stiffness_matrix(self) -> Any:
        """Return a new assembled PETSc stiffness matrix before elimination."""
        _backend()
        from dolfinx.fem.petsc import assemble_matrix

        matrix = assemble_matrix(self.stiffness_form)
        matrix.assemble()
        return matrix

    def mass_matrix(self) -> Any:
        """Return a new assembled PETSc consistent mass before elimination."""
        _backend()
        from dolfinx.fem.petsc import assemble_matrix

        matrix = assemble_matrix(self.mass_form)
        matrix.assemble()
        return matrix

    def force_vector(self, load: Load) -> Any:
        """Assemble distributed pressure loading and zero supported entries.

        The callable is interpolated into the FEM space, with values in N/m^2.
        Point loads await the shared arbitrary-point basis evaluator.
        The returned ghosted PETSc vector belongs to the caller.
        """
        if not isinstance(load, DistributedLoad):
            raise NotImplementedError(
                "Point loading requires the planned basis-evaluation component."
            )
        np, ufl, fem, _, _, PETSc = _backend()
        from dolfinx.fem.petsc import assemble_vector, set_bc

        pressure = fem.Function(self.function_space)

        def values(x: Any) -> Any:
            data = np.asarray(load.values(x))
            is_real = not np.issubdtype(PETSc.ScalarType, np.complexfloating)
            if np.iscomplexobj(data) and is_real:
                if np.any(data.imag != 0):
                    raise ValueError("Complex loading requires a complex PETSc build.")
                data = data.real
            return np.asarray(data, dtype=PETSc.ScalarType)

        pressure.interpolate(values)
        pressure.x.scatter_forward()
        v = ufl.TestFunction(self.function_space)
        form = fem.form(
            ufl.inner(pressure, v) * ufl.Measure("dx", domain=self.mesh)
        )
        vector = assemble_vector(form)
        vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )
        set_bc(vector, self.boundary_conditions)
        vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )
        return vector
