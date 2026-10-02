"""SHO fits and energy Q for forced displacement, using exp(+i omega t).

Frequencies are Hz, displacement m, force N and energy J. The energy
definition is the maximum *structural* bending + kinetic energy / work per
cycle, matching the legacy energy_qfac approach. No explicit fluid stored
energy is added. Frequency-dependent fluid loading need not be an exact SHO.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from mufsi.structure.loads import PointLoad, PointLoads


@dataclass(frozen=True)
class SHOFitResult:
    """f0 is the fitted undamped frequency, distinct from the amplitude peak.

    amplitude and background have the same units as the input observable.
    relative_error is the L2 amplitude-fit residual / L2 data norm.
    """

    resonance_frequency: float
    q_factor: float
    amplitude: float
    background: complex
    relative_error: float
    fitted_amplitudes: np.ndarray


@dataclass(frozen=True)
class EnergyQResult:
    """Frequency-first structural energies and full-cycle work.

    Fluid work and its relative discrepancy from input work are populated by
    energy_from_response when a pressure coupling or beam line force is given.
    """

    frequencies: np.ndarray
    q_factor: np.ndarray
    stored_energy: np.ndarray
    mean_stored_energy: np.ndarray
    dissipated_energy: np.ndarray
    fluid_dissipated_energy: np.ndarray | None = None
    work_balance_errors: np.ndarray | None = None


@dataclass(frozen=True)
class QFactorResult:
    """SHO fit plus energy evaluated at the fitted f0 with the same loading."""

    sho: SHOFitResult
    energy: EnergyQResult
    response: object
    resonance_response: object
    observable: np.ndarray
    load: PointLoads
    peak_frequency: float


def sho_amplitude(frequencies, amplitude, f0, q, background=0.0):
    """Constant-force displacement SHO with a coherent complex background.

    |a*f0^2/(f0^2-f^2+i*f*f0/Q) + b|. The numerator is independent of the
    drive frequency, as required by constant-amplitude point forcing.
    """
    f = np.asarray(frequencies, dtype=float)
    return np.abs(amplitude * f0**2 / (f0**2 - f**2 + 1j * f * f0 / q) + background)


def _spectrum(frequencies, amplitudes):
    f = np.asarray(frequencies, dtype=float)
    y = np.abs(np.asarray(amplitudes))
    if (
        f.ndim != 1
        or f.size < 5
        or y.shape != f.shape
        or not np.isfinite(f).all()
        or np.any(f <= 0)
        or np.any(np.diff(f) <= 0)
        or not np.isfinite(y).all()
        or np.max(y) <= 0
    ):
        raise ValueError(
            "Use at least five increasing positive frequencies and finite, "
            "nonzero 1D displacement amplitudes of matching length."
        )
    peak = int(np.argmax(y))
    if peak in (0, len(f) - 1):
        raise ValueError("The selected resonance must lie inside the frequency window.")
    return f, y, peak


def resonance_frequency(frequencies, amplitudes) -> float:
    """Interpolate the strongest amplitude peak with a local quadratic.

    Select an isolated frequency window first. This returns the amplitude
    peak frequency, not the SHO's undamped f0.
    """
    f, y, peak = _spectrum(frequencies, amplitudes)
    scale = max(f[peak] - f[peak - 1], f[peak + 1] - f[peak])
    x = (f[peak - 1 : peak + 2] - f[peak]) / scale
    a, b, _ = np.polyfit(x, y[peak - 1 : peak + 2] / y[peak], 2)
    offset = np.clip(-b / (2 * a), x[0], x[-1]) if a < 0 else 0
    return float(f[peak] + scale * offset)


def fit_sho(frequencies, amplitudes, *, fit_background=False) -> SHOFitResult:
    """Fit one resolved displacement resonance; accept real or complex data.

    Fits magnitudes, not power/PSD. Internally scales frequency and amplitude
    to handle microscopic SI displacements. Q is optimized logarithmically.
    At least five samples must resolve the fitted linewidth f0/Q. Neighbouring
    modes must be excluded by the caller; the fit residual is always returned.
    """
    f, y, peak = _spectrum(frequencies, amplitudes)
    if len(f) < 9:
        raise ValueError("An SHO fit requires at least nine frequency samples.")
    fscale, yscale = f[peak], float(np.max(y))
    x, data = f / fscale, y / yscale
    above = np.flatnonzero(data >= 1 / np.sqrt(2))
    width = f[above[-1]] - f[above[0]] if len(above) > 1 else np.min(np.diff(f))
    q0 = np.clip(fscale / width, 1.0, 1e5)
    f0 = min(f[-1] / fscale, 1 / np.sqrt(1 - 1 / (2 * q0**2)))
    start = [1 / q0, f0, np.log(q0)]
    lower, upper = [0, x[0], np.log(0.71)], [np.inf, x[-1], np.log(1e7)]
    if fit_background:
        start.extend([0.0, 0.0])
        lower.extend([-2.0, -2.0])
        upper.extend([2.0, 2.0])

    def model(parameters):
        a, center, logq = parameters[:3]
        b = parameters[3] + 1j * parameters[4] if fit_background else 0.0
        return sho_amplitude(x, a, center, np.exp(logq), b)

    fit = least_squares(
        lambda p: model(p) - data,
        start,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=4000,
    )
    if not fit.success or np.any(fit.active_mask):
        raise RuntimeError(
            "SHO fit failed or reached a parameter bound; revise the window."
        )
    a, center, logq = fit.x[:3]
    q, center = float(np.exp(logq)), float(center * fscale)
    count = np.count_nonzero(np.abs(f - center) <= center / q / 2)
    if count < 5:
        raise ValueError(
            "Fewer than five samples resolve f0/Q; refine the frequency grid."
        )
    fitted = model(fit.x) * yscale
    return SHOFitResult(
        center,
        q,
        float(a * yscale),
        complex((fit.x[3] + 1j * fit.x[4]) * yscale) if fit_background else 0j,
        float(np.linalg.norm(fitted - y) / np.linalg.norm(y)),
        fitted,
    )


def q_factor(frequencies, amplitudes, *, fit_background=False) -> float:
    """Return the Q of an isolated displacement resonance using fit_sho."""
    return fit_sho(frequencies, amplitudes, fit_background=fit_background).q_factor


def energy_q_factor(frequencies, displacement, stiffness, mass, force) -> EnergyQResult:
    """Maximum structural energy and input work, without time sampling.

    K and M are the real symmetric operators actually used by the solver,
    including C0 interior-penalty terms. u has shape (nfrequencies, ndofs).
    F may be one vector or frequency-first vectors. Constraints must already
    be zero in u. DeltaW = pi/omega * Re(F^H v), for arbitrary force phase.

    The maximum of E(theta) is the largest eigenvalue of its real 2x2
    quadratic form. This preserves spatial phase variations and avoids a
    400-point time scan. The definition includes no explicit fluid energy.
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    u = np.asarray(displacement, dtype=complex)
    if (
        f.ndim != 1
        or not f.size
        or not np.isfinite(f).all()
        or np.any(f <= 0)
        or u.ndim != 2
        or u.shape[0] != len(f)
        or not np.isfinite(u).all()
    ):
        raise ValueError(
            "Use positive finite frequencies and matching 2D displacement."
        )
    operators = []
    for matrix in (stiffness, mass):
        matrix = sparse.csr_matrix(matrix)
        if (
            matrix.shape != (u.shape[1], u.shape[1])
            or not np.isfinite(matrix.data).all()
            or np.any(matrix.data.imag != 0)
        ):
            raise ValueError("K and M must be finite real square matrices matching u.")
        matrix = matrix.real
        if np.linalg.norm((matrix - matrix.T).data) > 1e-10 * np.linalg.norm(
            matrix.data
        ):
            raise ValueError("K and M must be symmetric.")
        operators.append(matrix)
    K, M = operators
    F = np.asarray(force, dtype=complex)
    if F.shape not in {(u.shape[1],), u.shape} or not np.isfinite(F).all():
        raise ValueError("Force must be finite and match the displacement DOFs.")
    omega = 2 * np.pi * f
    v = 1j * omega[:, None] * u

    def quadratic(a, matrix, b):
        return np.sum(a * (matrix @ b.T).T, axis=1)

    a = quadratic(u.real, K, u.real) + quadratic(v.real, M, v.real)
    b = quadratic(u.real, K, u.imag) + quadratic(v.real, M, v.imag)
    c = quadratic(u.imag, K, u.imag) + quadratic(v.imag, M, v.imag)
    mean = (a + c) / 4
    maximum = mean + np.hypot(a - c, 2 * b) / 4
    work = np.pi / omega * np.real(np.sum(F.conj() * v, axis=1))
    if np.any(maximum <= 0) or np.any(work <= 0):
        raise ValueError("Energy Q requires positive stored energy and supplied work.")
    return EnergyQResult(f.copy(), 2 * np.pi * maximum / work, maximum, mean, work)


def corner_loads(structure, *, symmetry="symmetric", amplitude=1e-9, inset=1e-3):
    """Two forces at x=(1-inset)*L, y=+/-W/2; amplitude is N *per corner*.

    Equal signs excite symmetric displacement; opposite signs excite
    antisymmetric displacement. A 1D Euler-Bernoulli beam supports only the
    symmetric case, represented by two coincident forces with total 2F.
    inset is a fractional distance from the free tip, in [0,1).
    """
    from mufsi.structure.euler_bernoulli import EulerBernoulliBeam
    from mufsi.structure.kirchhoff import KirchhoffPlate

    if symmetry not in {"symmetric", "antisymmetric"}:
        raise ValueError("symmetry must be 'symmetric' or 'antisymmetric'.")
    if not np.isscalar(inset) or not np.isfinite(inset) or not 0 <= inset < 1:
        raise ValueError("inset must be a finite fraction in [0,1).")
    if not np.isscalar(amplitude) or not np.isfinite(amplitude) or amplitude == 0:
        raise ValueError("amplitude must be a finite nonzero force per corner.")
    if structure.boundary_condition != "cantilever":
        raise ValueError(
            "Corner excitation requires a cantilever with a free x=L edge."
        )
    x, width = (1 - inset) * structure.geometry.length, structure.geometry.width
    if isinstance(structure, EulerBernoulliBeam):
        if symmetry == "antisymmetric":
            raise ValueError(
                "A 1D Euler-Bernoulli beam has no antisymmetric width motion."
            )
        positions = ((x,), (x,))
    elif isinstance(structure, KirchhoffPlate):
        positions = ((x, width / 2), (x, -width / 2))
    else:
        raise TypeError(
            "Corner excitation supports KirchhoffPlate or EulerBernoulliBeam."
        )
    sign = 1 if symmetry == "symmetric" else -1
    return PointLoads(
        (PointLoad(positions[0], amplitude), PointLoad(positions[1], sign * amplitude))
    )


def corner_displacement(structure, displacement, *, symmetry="symmetric", inset=1e-3):
    """Mean (u_plus+u_minus)/2 or difference (u_plus-u_minus)/2 at the drive."""
    from mufsi.coupling.basis_evaluation import build_evaluation_matrix

    load = corner_loads(structure, symmetry=symmetry, inset=inset)
    E = build_evaluation_matrix(
        structure.function_space,
        np.asarray([p.position for p in load.loads]),
    )
    corners = (E @ np.asarray(displacement).T).T
    sign = 1 if symmetry == "symmetric" else -1
    return (corners[..., 0] + sign * corners[..., 1]) / 2


def energy_from_response(structure, response, load, *, coupling=None) -> EnergyQResult:
    """Apply energy_q_factor to a full-DOF FE result and check fluid work.

    For panel pressure results supply the same CouplingOperator as the solver.
    Weighted 3D results carry their integrated FE fluid_force directly.
    Beam line-force results use the consistent integral C=M/(rho*A). Both
    pressure and line force are *resisting* fluid forces in this library.
    """
    from mufsi.solvers.frequency_response import _csr

    if structure.mesh.comm.size != 1:
        raise NotImplementedError(
            "Energy postprocessing currently requires one MPI rank."
        )
    K, M = _csr(structure.stiffness_matrix()), _csr(structure.mass_matrix())
    force = structure.force_vector(load)
    try:
        F = force.getArray(readonly=True).copy()
    finally:
        force.destroy()
    result = energy_q_factor(response.frequencies, response.displacement, K, M, F)
    omega = 2 * np.pi * result.frequencies
    v = 1j * omega[:, None] * response.displacement
    power = None
    if getattr(response, "fluid_force", None) is not None:
        power = np.real(np.sum(v.conj() * response.fluid_force, axis=1))
    elif coupling is not None and getattr(response, "pressure", None) is not None:
        fluid_v = (coupling.evaluation_matrix @ v.T).T
        power = np.real(
            np.sum(fluid_v.conj() * response.pressure * coupling.weights, axis=1)
        )
    elif hasattr(response, "line_force"):
        projected = (M @ response.line_force.T).T / structure.line_density
        power = np.real(np.sum(v.conj() * projected, axis=1))
    if power is not None:
        work = np.pi / omega * power
        errors = np.abs(work - result.dissipated_energy) / result.dissipated_energy
        if np.any(work <= 0) or np.any(errors > 1e-5):
            raise RuntimeError("Fluid dissipation does not balance the supplied work.")
        result = replace(
            result, fluid_dissipated_energy=work, work_balance_errors=errors
        )
    return result


def analyze_q_factor(
    solver,
    frequencies,
    *,
    symmetry="symmetric",
    amplitude=1e-9,
    inset=1e-3,
    fit_background=False,
) -> QFactorResult:
    """Solve one isolated resonance with corner forcing and both Q estimates.

    Accept FrequencyResponseSolver(EB/KL + Stokes2D/Stokes3D), or
    BeamFrequencyResponseSolver(EulerBernoulliBeam + SectionForce2D). Supply
    a window resolving the resonance. Energy is evaluated with a new solve
    at the fitted f0, rather than at the nearest sampled frequency. Weighted
    3D energy/work uses integrated coefficient forces, not sampled pressures.
    """
    from mufsi.hydrodynamics.stokes_2d import Stokes2D
    from mufsi.hydrodynamics.stokes_3d import Stokes3D
    from mufsi.solvers.beam_frequency_response import BeamFrequencyResponseSolver
    from mufsi.solvers.frequency_response import FrequencyResponseSolver

    if isinstance(solver, BeamFrequencyResponseSolver):
        if solver.force_model is None:
            raise ValueError("A dissipative 2D section-force model is required.")
        structure, problem = solver.structure, None
    elif isinstance(solver, FrequencyResponseSolver):
        problem = solver.problem
        if not isinstance(problem.hydrodynamics, (Stokes2D, Stokes3D)):
            raise NotImplementedError(
                "Use an active Stokes2D or Stokes3D model for the Q workflow."
            )
        structure = problem.structure
    else:
        raise TypeError("Use FrequencyResponseSolver or BeamFrequencyResponseSolver.")
    load = corner_loads(structure, symmetry=symmetry, amplitude=amplitude, inset=inset)
    response = solver.solve(frequencies, load)
    observable = corner_displacement(
        structure,
        response.displacement,
        symmetry=symmetry,
        inset=inset,
    )
    sho = fit_sho(response.frequencies, observable, fit_background=fit_background)
    resonant = solver.solve([sho.resonance_frequency], load)
    energy = energy_from_response(
        structure,
        resonant,
        load,
        coupling=problem.coupling if problem else None,
    )
    return QFactorResult(
        sho,
        energy,
        response,
        resonant,
        observable,
        load,
        resonance_frequency(response.frequencies, observable),
    )
