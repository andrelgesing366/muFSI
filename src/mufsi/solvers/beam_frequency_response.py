"""Full beam response with a local hydrodynamic force, using complex SciPy LU."""

from dataclasses import dataclass

import numpy as np

from mufsi.hydrodynamics.section_force import SectionForce2D
from mufsi.solvers.frequency_response import _csr
from mufsi.solvers.linear import SciPyLUSolver
from mufsi.structure.euler_bernoulli import EulerBernoulliBeam


@dataclass(frozen=True)
class BeamFrequencyResponseResult:
    """Frequency-first displacement [m] and resisting line force [N/m].

    Both arrays use full structural DOF order. line_force contains nodal values
    of the local distributed force, rather than a structural force vector.
    relative_errors measures equilibrium on free DOFs / applied force norm.
    """

    frequencies: np.ndarray
    displacement: np.ndarray
    line_force: np.ndarray
    relative_errors: np.ndarray


@dataclass(frozen=True)
class BeamFrequencyResponseSolver:
    """Solve (K - omega^2 M + i*omega*Z*C) u = F, C = integral phi_i phi_j dx.

    C = M/(rho*A) for a homogeneous beam. This consistent FEM projection does
    not introduce a separate longitudinal fluid grid or truncate modes.
    Currently serial. With no force model, omega=0 gives static response.
    Zero frequency also means zero hydrodynamic force for static displacement.
    """

    structure: EulerBernoulliBeam
    force_model: SectionForce2D | None = None

    def __post_init__(self):
        if not isinstance(self.structure, EulerBernoulliBeam):
            raise TypeError("BeamFrequencyResponseSolver requires EulerBernoulliBeam.")
        if self.force_model is not None:
            if not isinstance(self.force_model, SectionForce2D):
                raise TypeError("force_model must be SectionForce2D or None.")
            a, b = self.structure.geometry, self.force_model.geometry
            if not np.allclose(
                [a.length, a.width, a.thickness],
                [b.length, b.width, b.thickness],
                rtol=1e-12,
                atol=0,
            ):
                raise ValueError("Beam and fluid-force geometry must match.")

    def solve(self, frequencies, load, *, progress=None):
        """Frequencies in Hz; DistributedLoad in N/m or PointLoad in N."""
        f = np.atleast_1d(np.array(frequencies, dtype=float, copy=True))
        if f.ndim != 1 or not len(f) or not np.isfinite(f).all() or np.any(f < 0):
            raise ValueError(
                "frequencies must be a nonempty finite nonnegative 1D array."
            )
        beam = self.structure
        if beam.mesh.comm.size != 1:
            raise NotImplementedError(
                "Beam driven response currently requires one MPI rank."
            )
        K, M = _csr(beam.stiffness_matrix()), _csr(beam.mass_matrix())
        vector = beam.force_vector(load)
        try:
            F = vector.getArray(readonly=True).copy().astype(complex)
        finally:
            vector.destroy()
        free = np.setdiff1d(np.arange(len(F)), beam.constrained_dofs)
        k, m, rhs = K[free][:, free], M[free][:, free], F[free]
        c = m / beam.line_density
        scale = max(np.max(np.abs(k.diagonal())), np.finfo(float).tiny)
        displacement = np.zeros((len(f), len(F)), dtype=complex)
        line_force = np.zeros_like(displacement)
        errors = np.zeros(len(f))
        for i, hz in enumerate(f):
            omega = 2 * np.pi * hz
            fluid_term = (
                0j
                if self.force_model is None or hz == 0
                else self.force_model.dynamic_stiffness(omega)
            )
            A = k - omega**2 * m + fluid_term * c
            solver = SciPyLUSolver()
            solver.factorize(A / scale)
            u = solver.solve(rhs / scale)
            for _ in range(2):
                u += solver.solve((rhs - A @ u) / scale)
            if not np.isfinite(u).all():
                raise RuntimeError(f"Non-finite beam response at {hz:g} Hz.")
            displacement[i, free] = u
            line_force[i] = fluid_term * displacement[i]
            errors[i] = np.linalg.norm(A @ u - rhs) / max(
                np.linalg.norm(rhs), np.finfo(float).tiny
            )
            if progress is not None:
                progress(i + 1, len(f))
        return BeamFrequencyResponseResult(f, displacement, line_force, errors)
