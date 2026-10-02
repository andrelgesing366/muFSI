"""DOLFINx Euler-Bernoulli beam with symmetric C0 interior penalty."""

from dataclasses import dataclass, field
from functools import cached_property
from math import isfinite
from numbers import Integral
from typing import Any

from mufsi.models.geometry import BeamGeometry
from mufsi.models.material import Material
from mufsi.structure.base import StructuralModel
from mufsi.structure.loads import DistributedLoad, Load, PointLoad, PointLoads


def _backend():
    try:
        import numpy as np
        import ufl
        from dolfinx import fem, mesh
        from mpi4py import MPI
        from petsc4py import PETSc
    except ImportError as error:
        raise ImportError(
            "EulerBernoulliBeam requires a matched DOLFINx/UFL/MPI/PETSc "
            "environment. See docs/beam_cantilever.md."
        ) from error
    return np, ufl, fem, mesh, MPI, PETSc


@dataclass(frozen=True)
class EulerBernoulliBeam(StructuralModel):
    """Homogeneous rectangular beam on [0,L], in physical SI coordinates.

    Cantilever clamps x=0, bridge/clamped clamps both ends, and
    simply_supported fixes displacement at both ends with free slope.
    Slope continuity and clamped slopes are enforced weakly. No Poisson
    correction is applied to EI. The full consistent mass is rho*A.

    Setup is lazy and immutable. Matrix calls return caller-owned PETSc
    matrices before displacement constraints are eliminated by the solver.
    """

    geometry: BeamGeometry
    material: Material
    mesh_resolution: int = 64
    boundary_condition: str = "cantilever"
    element_degree: int = 2
    penalty: float = 16.0
    comm: Any = field(default=None, repr=False, compare=False, kw_only=True)

    def __post_init__(self):
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
        if (
            not isfinite(self.material.poisson_ratio)
            or not -1 < self.material.poisson_ratio < 0.5
        ):
            raise ValueError("poisson_ratio must lie strictly between -1 and 0.5.")
        for name, value, minimum in (
            ("mesh_resolution", self.mesh_resolution, 1),
            ("element_degree", self.element_degree, 2),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.boundary_condition not in {
            "cantilever",
            "bridge",
            "clamped",
            "simply_supported",
        }:
            raise ValueError("Unknown beam boundary_condition.")

    @property
    def cross_section_area(self):
        """Area W*t in m^2."""
        return self.geometry.width * self.geometry.thickness

    @property
    def second_moment_of_area(self):
        """Out-of-plane second moment W*t^3/12 in m^4."""
        return self.geometry.width * self.geometry.thickness**3 / 12

    @property
    def flexural_rigidity(self):
        """EI in N m^2."""
        return self.material.young_modulus * self.second_moment_of_area

    @property
    def line_density(self):
        """rho*A in kg/m."""
        return self.material.density * self.cross_section_area

    @cached_property
    def mesh(self):
        _, _, _, mesh, MPI, _ = _backend()
        return mesh.create_interval(
            MPI.COMM_WORLD if self.comm is None else self.comm,
            int(self.mesh_resolution),
            [0.0, self.geometry.length],
            ghost_mode=mesh.GhostMode.shared_facet,
        )

    @cached_property
    def function_space(self):
        _, _, fem, _, _, _ = _backend()
        return fem.functionspace(self.mesh, ("Lagrange", int(self.element_degree)))

    @cached_property
    def boundary_tags(self):
        """Endpoint tags 1=left and 2=right, using the physical length."""
        np, _, _, mesh, _, _ = _backend()
        length = self.geometry.length
        self.mesh.topology.create_connectivity(0, 1)
        left = mesh.locate_entities_boundary(
            self.mesh,
            0,
            lambda x: np.isclose(x[0], 0, rtol=0, atol=1e-12 * length),
        )
        right = mesh.locate_entities_boundary(
            self.mesh,
            0,
            lambda x: np.isclose(x[0], length, rtol=0, atol=1e-12 * length),
        )
        indices = np.r_[left, right]
        values = np.r_[
            np.ones(len(left), dtype=np.int32), np.full(len(right), 2, dtype=np.int32)
        ]
        order = np.argsort(indices)
        return mesh.meshtags(self.mesh, 0, indices[order], values[order])

    @cached_property
    def constrained_dofs(self):
        np, _, fem, _, _, _ = _backend()
        endpoints = self.boundary_tags.find(1)
        if self.boundary_condition != "cantilever":
            endpoints = np.r_[endpoints, self.boundary_tags.find(2)]
        return fem.locate_dofs_topological(self.function_space, 0, endpoints)

    @cached_property
    def boundary_conditions(self):
        _, _, fem, _, _, PETSc = _backend()
        return (
            fem.dirichletbc(
                PETSc.ScalarType(0),
                self.constrained_dofs,
                self.function_space,
            ),
        )

    @cached_property
    def stiffness_form(self):
        """EI bending plus symmetric slope consistency and stabilization."""
        _, ufl, fem, _, _, _ = _backend()
        u, v = (
            ufl.TrialFunction(self.function_space),
            ufl.TestFunction(self.function_space),
        )
        n, h = ufl.FacetNormal(self.mesh), ufl.CellDiameter(self.mesh)
        dx, dS = (
            ufl.Measure("dx", domain=self.mesh),
            ufl.Measure("dS", domain=self.mesh),
        )
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.boundary_tags)
        curvature_u, curvature_v = u.dx(0).dx(0), v.dx(0).dx(0)
        jump_u, jump_v = ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n)
        EI = self.flexural_rigidity
        alpha = self.penalty * (self.element_degree / 2) ** 2 * EI
        a = (
            EI * ufl.inner(curvature_u, curvature_v) * dx
            - EI * ufl.inner(ufl.avg(curvature_u), jump_v) * dS
            - EI * ufl.inner(jump_u, ufl.avg(curvature_v)) * dS
            + alpha / ufl.avg(h) * ufl.inner(jump_u, jump_v) * dS
        )
        if self.boundary_condition != "simply_supported":
            slope_u, slope_v = ufl.dot(ufl.grad(u), n), ufl.dot(ufl.grad(v), n)
            endpoint_term = (
                -EI * ufl.inner(curvature_u, slope_v)
                - EI * ufl.inner(slope_u, curvature_v)
                + 2 * alpha / h * ufl.inner(slope_u, slope_v)
            )
            a += endpoint_term * ds(1)
            if self.boundary_condition != "cantilever":
                a += endpoint_term * ds(2)
        return fem.form(a)

    @cached_property
    def mass_form(self):
        _, ufl, fem, _, _, _ = _backend()
        u, v = (
            ufl.TrialFunction(self.function_space),
            ufl.TestFunction(self.function_space),
        )
        return fem.form(
            self.line_density * ufl.inner(u, v) * ufl.Measure("dx", domain=self.mesh)
        )

    def stiffness_matrix(self):
        _backend()
        from dolfinx.fem.petsc import assemble_matrix

        matrix = assemble_matrix(self.stiffness_form)
        matrix.assemble()
        return matrix

    def mass_matrix(self):
        _backend()
        from dolfinx.fem.petsc import assemble_matrix

        matrix = assemble_matrix(self.mass_form)
        matrix.assemble()
        return matrix

    def force_vector(self, load: Load):
        """Assemble line loading in N/m or a serial point force in N.

        The distributed callable receives coordinates (gdim,npoints).
        Complex values require a complex PETSc build. Returned vectors are
        caller-owned, ghost-synchronized, with supported entries zero.
        """
        np, ufl, fem, _, _, PETSc = _backend()
        from dolfinx.fem.petsc import assemble_vector, set_bc

        def scalar_values(data):
            data = np.asarray(data)
            if not np.isfinite(data).all():
                raise ValueError("Load values must be finite.")
            if np.iscomplexobj(data) and not np.issubdtype(
                PETSc.ScalarType, np.complexfloating
            ):
                if np.any(data.imag != 0):
                    raise ValueError("Complex loading requires a complex PETSc build.")
                data = data.real
            return np.asarray(data, dtype=PETSc.ScalarType)

        v = ufl.TestFunction(self.function_space)
        if isinstance(load, DistributedLoad):
            line_load = fem.Function(self.function_space)
            line_load.interpolate(
                lambda x: np.broadcast_to(
                    scalar_values(load.values(x)),
                    (x.shape[1],),
                )
            )
            line_load.x.scatter_forward()
            form = fem.form(
                ufl.inner(line_load, v) * ufl.Measure("dx", domain=self.mesh)
            )
            vector = assemble_vector(form)
        elif isinstance(load, (PointLoad, PointLoads)):
            if self.mesh.comm.size != 1:
                raise NotImplementedError(
                    "Arbitrary beam point loads currently require one MPI rank."
                )
            from mufsi.coupling.basis_evaluation import point_force_values

            values = scalar_values(point_force_values(self.function_space, load))
            zero = fem.Constant(self.mesh, PETSc.ScalarType(0))
            vector = assemble_vector(
                fem.form(ufl.inner(zero, v) * ufl.Measure("dx", domain=self.mesh))
            )
            vector.getArray()[:] = values
        else:
            raise TypeError("Use DistributedLoad, PointLoad or PointLoads.")
        vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )
        set_bc(vector, self.boundary_conditions)
        vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )
        return vector
