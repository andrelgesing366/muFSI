"""Panel-exact F2D flow recovery with exp(+i omega t), independent of FEM.

Pressure is the resisting traction used by Stokes2D (fluid force on the
plate is -p). For F = -(log(R) + K0(alpha*R))/(2*pi*alpha**2),
psi_x = -sum_j p_j [F(y-b_j,z) - F(y-a_j,z)] / mu.
The curl convention is u_y = psi_x,z and u_z = -psi_x,y. In particular,
u_z(y,0) = B p at the collocation points: no pressure or phase sign flip.
Velocity and symmetric strain rate are differentiated analytically.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.special import kv

from mufsi.hydrodynamics.stokes_2d import Stokes2D


@dataclass(frozen=True)
class FlowField2D:
    """Complex SI phasors at points (...,2)=(y,z) or (...,3)=(x,y,z).

    streamfunction: m^2/s; velocity: (...,2), (u_y,u_z), m/s;
    strain_rate: (...,2,2), symmetric y/z tensor, 1/s.
    mean_dissipation: cycle-averaged power density, W/m^3;
    energy_dissipation: full-cycle loss density, J/m^3.
    singular_points marks panel endpoints on z=0, where derivatives diverge.
    These densities are local: integrating a section gives loss per length,
    not the total plate loss or a Q-factor.
    """

    omega: float
    points: np.ndarray
    streamfunction: np.ndarray
    velocity: np.ndarray
    strain_rate: np.ndarray
    mean_dissipation: np.ndarray
    energy_dissipation: np.ndarray
    singular_points: np.ndarray
    plate_edges: tuple[float, float]

    def instantaneous_velocity(self, phase=0.0):
        """Real velocity at phase omega*t (radians), preserving spatial phase."""
        if not np.isscalar(phase) or not np.isfinite(phase):
            raise ValueError("phase must be a finite scalar in radians.")
        return np.real(self.velocity * np.exp(1j * phase))


def _radial_kernel(alpha, r):
    """Return F (constant gauge removed), F'/R, and F''-F'/R.

    A small-argument Bessel series avoids cancellation of log(R) with K0,
    and of 1/R with alpha*K1. Four terms suffice for |alpha*R| < 0.05.
    The discarded constant is shared by every endpoint and cancels exactly.
    """
    s = alpha * r
    small = np.abs(s) < 0.05
    f, h, d = (np.empty(r.shape, dtype=complex) for _ in range(3))
    if np.any(small):
        t = s[small]
        log = np.log(t / 2) + np.euler_gamma
        power = np.ones(t.shape, dtype=complex)
        fs, hs, ds = (np.zeros(t.shape, dtype=complex) for _ in range(3))
        harmonic = 0.0
        for k in range(1, 5):
            harmonic += 1 / k
            power *= t**2 / (4 * k**2)
            fs += power * (log - harmonic)
            hs += power * (2 * k * (log - harmonic) + 1)
            ds += power * (4 * k * (k - 1) * (log - harmonic) + 4 * k - 2)
        f[small] = fs / (2 * np.pi * alpha**2)
        h[small] = hs / (2 * np.pi * t**2)
        d[small] = ds / (2 * np.pi * t**2)
    if np.any(~small):
        t = s[~small]
        k0, k1 = kv(0, t), kv(1, t)
        f[~small] = -(np.log(t / 2) + np.euler_gamma + k0) / (2 * np.pi * alpha**2)
        h[~small] = (t * k1 - 1) / (2 * np.pi * t**2)
        d[~small] = (2 - t**2 * k0 - 2 * t * k1) / (2 * np.pi * t**2)
    return f, h, d


def reconstruct_flow(
    omega,
    points,
    pressure,
    hydrodynamics,
    *,
    section_index=-1,
    batch_size=256,
    singular="nan",
) -> FlowField2D:
    """Recover streamfunction, velocity, strain, and dissipation for Stokes2D.

    pressure is one frequency's full grid vector (nx*ny,) or (nx,ny), in
    x-major order. For (...,2) points, select an x section with section_index
    (default: last section). A single (ny,) pressure vector is also accepted.
    For (...,3) physical points, linearly interpolate full-grid pressure in
    x, strictly within grid.x; there is no longitudinal fluid interaction.

    Every pressure panel is integrated exactly using endpoint differences.
    Observation points need not coincide with collocation nodes. Singular
    panel endpoints on z=0 produce NaNs by default, or raise when
    singular='raise'. Other surface points retain the no-slip velocity.
    Bounded batches avoid a full observation-by-pressure-grid tensor.
    """
    if not isinstance(hydrodynamics, Stokes2D):
        raise NotImplementedError("Flow recovery currently supports Stokes2D only.")
    if not np.isscalar(omega) or not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be finite and strictly positive.")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")
    if singular not in {"nan", "raise"}:
        raise ValueError("singular must be 'nan' or 'raise'.")
    points = np.array(points, dtype=float, copy=True)
    if (
        points.ndim < 2
        or points.shape[-1] not in (2, 3)
        or points.size == 0
        or not np.isfinite(points).all()
    ):
        raise ValueError("points must be finite with shape (...,2) or (...,3).")
    grid = hydrodynamics.grid
    p = np.asarray(pressure, dtype=complex)
    full = p.shape in {(grid.nx * grid.ny,), (grid.nx, grid.ny)}
    if not np.isfinite(p).all() or (not full and p.shape != (grid.ny,)):
        raise ValueError(
            "pressure must be finite with shape (nx*ny,), (nx,ny), or (ny,)."
        )
    flat = points.reshape(-1, points.shape[-1])
    if points.shape[-1] == 2:
        if (
            isinstance(section_index, bool)
            or not isinstance(section_index, Integral)
            or not -grid.nx <= section_index < grid.nx
        ):
            raise ValueError("section_index must identify an existing x section.")
        section_p = p.reshape(grid.nx, grid.ny)[section_index] if full else p
    else:
        if not full:
            raise ValueError("Physical (x,y,z) points require full-grid pressure.")
        if np.any(flat[:, 0] < grid.x[0]) or np.any(flat[:, 0] > grid.x[-1]):
            raise ValueError("Observation x coordinates must lie within grid.x.")
        p = p.reshape(grid.nx, grid.ny)

    shape, n = points.shape[:-1], len(flat)
    psi = np.empty(n, dtype=complex)
    velocity = np.empty((n, 2), dtype=complex)
    strain = np.empty((n, 2, 2), dtype=complex)
    mask = np.zeros(n, dtype=bool)
    alpha = np.sqrt(1j * omega / hydrodynamics.fluid.kinematic_viscosity)
    mu = hydrodynamics.fluid.dynamic_viscosity
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        obs = flat[start:stop]
        if points.shape[-1] == 2:
            pb = section_p
        elif grid.nx == 1:
            pb = p[0]
        else:
            right = np.clip(np.searchsorted(grid.x, obs[:, 0]), 1, grid.nx - 1)
            fraction = (obs[:, 0] - grid.x[right - 1]) / (
                grid.x[right] - grid.x[right - 1]
            )
            pb = (1 - fraction[:, None]) * p[right - 1] + fraction[:, None] * p[right]
        dy = obs[:, -2, None] - grid.panel_edges[None, :]
        z = obs[:, -1, None]
        r = np.hypot(dy, z)
        bad = np.any(r == 0, axis=1)
        if singular == "raise" and np.any(bad):
            raise ValueError(
                "Observation point lies on a singular pressure-panel endpoint."
            )
        mask[start:stop] = bad
        # Substitute only for masked points; never regularize the actual fluid kernel.
        safe_r = np.where(r == 0, 1.0, r)
        f, h, d = _radial_kernel(alpha, safe_r)

        def integrate(values, pb=pb):
            return -np.sum(np.diff(values, axis=1) * pb, axis=1) / mu

        psi[start:stop] = integrate(f)
        velocity[start:stop, 0] = integrate(h * z)
        velocity[start:stop, 1] = -integrate(h * dy)
        yy = integrate(h + d * (dy / safe_r) ** 2)
        zz = integrate(h + d * (z / safe_r) ** 2)
        yz = integrate(d * (dy / safe_r) * (z / safe_r))
        strain[start:stop, 0, 0] = yz
        strain[start:stop, 1, 1] = -yz
        strain[start:stop, 0, 1] = strain[start:stop, 1, 0] = (zz - yy) / 2
    psi[mask], velocity[mask], strain[mask] = np.nan, np.nan, np.nan
    # <2 mu eps(t):eps(t)> = mu eps:eps*, for peak-amplitude phasors.
    mean = mu * np.sum(np.abs(strain) ** 2, axis=(1, 2))
    energy = 2 * np.pi / omega * mean
    return FlowField2D(
        float(omega),
        points,
        psi.reshape(shape),
        velocity.reshape((*shape, 2)),
        strain.reshape((*shape, 2, 2)),
        mean.reshape(shape),
        energy.reshape(shape),
        mask.reshape(shape),
        (float(grid.panel_edges[0]), float(grid.panel_edges[-1])),
    )


def reconstruct_section(omega, y, z, pressure, hydrodynamics, **kwargs):
    """Recover a section mesh with shape (len(z),len(y)), coordinates in metres.

    Keyword options go to reconstruct_flow, including section_index.
    """
    y, z = np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    for axis in (y, z):
        if (
            axis.ndim != 1
            or len(axis) < 2
            or not np.isfinite(axis).all()
            or np.any(np.diff(axis) <= 0)
        ):
            raise ValueError(
                "y and z must be increasing finite 1D axes of length >= 2."
            )
    yy, zz = np.meshgrid(y, z)
    return reconstruct_flow(
        omega,
        np.stack((yy, zz), axis=-1),
        pressure,
        hydrodynamics,
        **kwargs,
    )


def reconstruct_flow_from_response(
    structure, response, fluid, grid, points, *, frequency_index=0, **kwargs
):
    """Recover a 2D field approximation from EB/KL displacement of ANY model.

    The supplied grid is a separate 2D fluid grid. Structural velocity drives
    a fresh Stokes2D pressure solve; neither 3D coefficients nor Sader line
    forces are interpreted as 2D pressure. This field's dissipation is the
    2D approximation and is not the energy-Q denominator of a 3D/Sader solve.
    """
    from mufsi.coupling.weighted import surface_evaluation

    if (
        isinstance(frequency_index, bool)
        or not isinstance(frequency_index, Integral)
        or not 0 <= frequency_index < len(response.frequencies)
    ):
        raise ValueError("frequency_index must identify an existing response row.")
    omega = 2 * np.pi * response.frequencies[frequency_index]
    velocity = (
        1j
        * omega
        * (
            surface_evaluation(structure, grid.points)
            @ response.displacement[frequency_index]
        )
    )
    hydro = Stokes2D(fluid, grid)
    pressure = hydro.pressure_from_velocity(omega, velocity)
    return reconstruct_flow(omega, points, pressure, hydro, **kwargs)


def plot_flow(field: FlowField2D, *, phase=0.0, density=1.2):
    """Plot streamfunction/streamlines, velocity, and full-cycle energy loss.

    Requires a uniform rectilinear (z,y) section from reconstruct_section.
    Coordinates display in um, velocity in mm/s, loss density in J/m^3.
    Streamlines and arrows show Re(u exp(i*phase)); they are phase snapshots,
    not particle trajectories. Dissipation is independent of phase.
    Matplotlib is imported only here; install the optional plot extra.
    Returns (figure, axes) without showing or saving the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if field.points.ndim != 3 or field.points.shape[-1] != 2:
        raise ValueError("Plot a single rectilinear (y,z) section.")
    y, z = field.points[0, :, 0], field.points[:, 0, 1]
    yy, zz = np.meshgrid(y, z)
    if not np.array_equal(field.points, np.stack((yy, zz), axis=-1)):
        raise ValueError("Plot points must form a rectilinear section mesh.")
    for axis in (y, z):
        if (
            len(axis) < 2
            or np.any(np.diff(axis) <= 0)
            or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-7, atol=0)
        ):
            raise ValueError(
                "Streamline plotting requires uniform increasing y,z axes."
            )
    velocity = field.instantaneous_velocity(phase)
    psi = np.real(field.streamfunction * np.exp(1j * phase))
    speed = np.linalg.norm(velocity, axis=-1) * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6), layout="constrained")
    if np.any(np.isfinite(psi)) and np.nanmax(psi) > np.nanmin(psi):
        contour = axes[0].contour(y * 1e6, z * 1e6, psi, levels=18, cmap="coolwarm")
        fig.colorbar(
            contour,
            ax=axes[0],
            label="Streamfunction [m²/s]",
            orientation="horizontal",
            pad=0.2,
        )
    axes[0].streamplot(
        y * 1e6,
        z * 1e6,
        np.ma.masked_invalid(velocity[..., 0]),
        np.ma.masked_invalid(velocity[..., 1]),
        color="0.25",
        density=density,
        linewidth=0.7,
        arrowsize=0.8,
    )
    axes[0].set_title(f"Streamlines at phase {phase:.3g} rad")
    mesh = axes[1].pcolormesh(y * 1e6, z * 1e6, speed, shading="auto", cmap="viridis")
    fig.colorbar(
        mesh,
        ax=axes[1],
        label="Instantaneous speed [mm/s]",
        orientation="horizontal",
        pad=0.2,
    )
    step = (
        slice(None, None, max(1, len(z) // 16)),
        slice(None, None, max(1, len(y) // 24)),
    )
    if np.any(np.isfinite(speed) & (speed > 0)):
        axes[1].quiver(
            yy[step] * 1e6,
            zz[step] * 1e6,
            velocity[..., 0][step],
            velocity[..., 1][step],
            color="white",
        )
    axes[1].set_title("Fluid velocity")
    positive = field.energy_dissipation[
        np.isfinite(field.energy_dissipation) & (field.energy_dissipation > 0)
    ]
    norm = None
    if positive.size and positive.max() > positive.min():
        norm = LogNorm(vmin=positive.min(), vmax=positive.max())
    mesh = axes[2].pcolormesh(
        y * 1e6,
        z * 1e6,
        np.ma.masked_invalid(field.energy_dissipation),
        shading="auto",
        cmap="magma",
        norm=norm,
    )
    fig.colorbar(
        mesh,
        ax=axes[2],
        label="Energy dissipated per cycle [J/m³]",
        orientation="horizontal",
        pad=0.2,
    )
    axes[2].set_title("Viscous dissipation")
    for ax in axes:
        ax.plot(np.asarray(field.plate_edges) * 1e6, [0, 0], color="black", lw=3)
        ax.set(xlabel="y [µm]", ylabel="z [µm]", aspect="equal")
    return fig, axes
