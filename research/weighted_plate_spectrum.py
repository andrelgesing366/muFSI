"""Kirchhoff-Love plate spectrum with continuous even/odd weighted pressure.

Research-only serial modal coupling. Reuses production KL FEM and the point
Stokeslet read-only; no panel pressure fitting. exp(+i omega t).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from weighted_beam_spectrum import solve_modal_response
from weighted_plate_mobility import (
    PlateMobility,
    PlateModes,
    PlatePressureBasis,
    converged_projection,
)
from weighted_pressure_mobility import (
    ROOT,
    Fluid,
    PlateGeometry,
    Quadrature,
    solve_coefficients,
)
from weighted_pressure_projection import source_hashes


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=4, help="Maximum longitudinal degree")
    parser.add_argument(
        "--K", type=int, default=6, help="Maximum transverse degree, all 0..K"
    )
    parser.add_argument("--length", type=float, default=500e-6)
    parser.add_argument("--width", type=float, default=500e-6)
    parser.add_argument("--thickness", type=float, default=5e-6)
    parser.add_argument("--young-modulus", type=float, default=169e9)
    parser.add_argument("--solid-density", type=float, default=2330)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument(
        "--boundary-condition",
        choices=("cantilever", "bridge", "clamped", "simply_supported"),
        default="cantilever",
    )
    parser.add_argument("--mesh-x", type=int, default=24)
    parser.add_argument("--mesh-y", type=int, default=24)
    parser.add_argument("--modes", type=int, default=3)
    parser.add_argument("--density", type=float, default=997)
    parser.add_argument("--viscosity", type=float, default=890e-6)
    parser.add_argument("--backend", choices=("quadpy", "gauss"), default="quadpy")
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument(
        "--orders", type=int, nargs="+", default=[12, 20, 32, 48, 72, 104, 144]
    )
    parser.add_argument("--nx", type=int, help="Fluid collocation in x")
    parser.add_argument("--ny", type=int, help="Fluid collocation over the full width")
    parser.add_argument(
        "--force", type=float, default=1e-9, help="Real point force in N"
    )
    parser.add_argument("--force-x", type=float, default=1.0, help="Force/monitor x/L")
    parser.add_argument("--force-y", type=float, default=0.3, help="Force/monitor y/W")
    parser.add_argument(
        "--load",
        choices=("point", "antisymmetric"),
        default="point",
        help="Antisymmetric = +F at y and -F at -y",
    )
    parser.add_argument(
        "--frequencies", type=float, nargs="+", help="Explicit frequencies in Hz"
    )
    parser.add_argument("--start", type=float)
    parser.add_argument("--stop", type=float)
    parser.add_argument("--frequency-count", type=int, default=25)
    parser.add_argument("--projection-rtol", type=float, default=2e-4)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/results/weighted_plate_spectrum"
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if (
        not np.isfinite([args.force, args.force_x, args.force_y]).all()
        or args.force == 0
    ):
        parser.error("Force and coordinates must be finite, with nonzero force.")
    if not 0 <= args.force_x <= 1 or not -0.5 <= args.force_y <= 0.5:
        parser.error("Force coordinates must lie on the plate.")
    if args.load == "antisymmetric" and args.force_y == 0:
        parser.error("An antisymmetric force pair needs nonzero force-y.")
    if not np.isfinite(args.projection_rtol) or args.projection_rtol <= 0:
        parser.error("projection-rtol must be finite and positive.")
    return args, parser


def main():
    args, parser = arguments()
    before = source_hashes()
    from mufsi import KirchhoffPlate, Material

    geometry = PlateGeometry(args.length, args.width, args.thickness)
    plate = KirchhoffPlate(
        geometry,
        Material(args.young_modulus, args.solid_density, args.poisson_ratio),
        mesh_resolution=(args.mesh_x, args.mesh_y),
        boundary_condition=args.boundary_condition,
    )
    modes = PlateModes(plate, args.modes)
    parity = modes.symmetry()
    print(
        f"Dry frequencies (Hz): {modes.dry_omega / (2 * np.pi)}; parities: {[p['parity'] for p in parity]}",
        flush=True,
    )
    basis = PlatePressureBasis(geometry, args.M, args.K)
    fluid = Fluid(args.density, args.viscosity)
    hydro = PlateMobility(
        basis, fluid, Quadrature(args.backend, tuple(args.orders), args.rtol, args.atol)
    )
    points = basis.collocation(args.nx, args.ny)
    E = modes.values(points)
    C, projection_report = converged_projection(
        basis, modes.values, args.projection_rtol
    )
    nx, ny = args.nx or 2 * (args.M + 1), args.ny or 2 * (args.K + 1)
    check_points = basis.collocation(nx + 1, ny + 2)[::3]
    Echeck = modes.values(check_points)
    monitor = np.array(
        [[args.force_x * geometry.length, args.force_y * geometry.width]]
    )
    monitor_shapes = modes.values(monitor)[0]
    force = args.force * monitor_shapes
    if args.load == "antisymmetric":
        force -= args.force * modes.values(monitor * [1, -1])[0]
    if np.linalg.norm(force) < abs(args.force) * 1e-12:
        parser.error(
            "The load has zero projection on the retained modes; change position or mode count."
        )
    first_hz = modes.dry_omega[0] / (2 * np.pi)
    if args.frequencies is None:
        start = 0.01 * first_hz if args.start is None else args.start
        stop = (
            1.1 * modes.dry_omega[min(1, args.modes - 1)] / (2 * np.pi)
            if args.stop is None
            else args.stop
        )
        if (
            not np.isfinite([start, stop]).all()
            or start <= 0
            or stop <= start
            or args.frequency_count < 2
        ):
            parser.error(
                "Use finite 0<start<stop and frequency-count>=2, or --frequencies including 0."
            )
        frequencies = np.geomspace(start, stop, args.frequency_count)
    else:
        frequencies = np.asarray(args.frequencies)
    if (
        not np.isfinite(frequencies).all()
        or np.any(frequencies < 0)
        or np.any(np.diff(frequencies) <= 0)
    ):
        parser.error("Frequencies must be finite, nonnegative, strictly increasing.")
    x, y = (
        np.linspace(0, geometry.length, 61),
        np.linspace(-geometry.width / 2, geometry.width / 2, 41),
    )
    grid = np.column_stack((np.repeat(x, len(y)), np.tile(y, len(x))))
    shapes = modes.values(grid)
    px, py = (
        np.linspace(0.005, 0.995, 61) * geometry.length,
        np.linspace(-0.495, 0.495, 41) * geometry.width,
    )
    pressure_points = np.column_stack((np.repeat(px, len(py)), np.tile(py, len(px))))
    pressure_basis = basis.values(pressure_points)
    modal, dry, coefficients, impedances, diagnostics, saved_H = [], [], [], [], [], []
    started = perf_counter()
    odd_columns = np.tile(basis.transverse_degrees % 2 == 1, basis.M + 1)
    for i, hz in enumerate(frequencies):
        omega = 2 * np.pi * hz
        H, integration = hydro.assemble(omega, points)
        A, fit = solve_coefficients(H, E)
        Z = C @ A
        d, equilibrium = solve_modal_response(modes, omega, Z, force)
        a = 1j * omega * A @ d
        d0, _ = solve_modal_response(modes, omega, np.zeros_like(Z), force)
        Hcheck, _ = hydro.assemble(omega, check_points)
        per_mode = np.linalg.norm(Hcheck @ A - Echeck, axis=0) / np.linalg.norm(
            Echeck, axis=0
        )
        desired, obtained = Echeck @ d, Hcheck @ (A @ d)
        error = float(np.linalg.norm(obtained - desired) / np.linalg.norm(desired))
        resistance = (Z + Z.conj().T) / 2
        row = {
            "frequency_Hz": float(hz),
            "rank": fit["rank"],
            "scaled_condition": fit["scaled_condition"],
            "modal_collocation_errors": fit["relative_residual"],
            "modal_heldout_errors": per_mode.tolist(),
            "response_heldout_error": error,
            "equilibrium_error": equilibrium,
            "reciprocity_error": float(np.linalg.norm(Z - Z.T) / np.linalg.norm(Z)),
            "minimum_resistance_eigenvalue_Ns_per_m": float(
                np.linalg.eigvalsh(resistance).min()
            ),
            "even_pressure_fraction": float(
                np.linalg.norm(a[~odd_columns]) / max(np.linalg.norm(a), 1e-300)
            )
            if omega
            else None,
            "quadrature": integration,
        }
        modal.append(d)
        dry.append(d0)
        coefficients.append(a)
        impedances.append(Z)
        diagnostics.append(row)
        if i in {0, len(frequencies) // 2, len(frequencies) - 1}:
            saved_H.append((i, H))
        print(
            f"{i + 1}/{len(frequencies)}: {hz:.1f} Hz; monitor {abs(monitor_shapes @ d) * 1e9:.4g} nm; heldout {error:.2%}",
            flush=True,
        )
    if before != source_hashes():
        raise RuntimeError("Production source changed during the experiment.")
    modal, dry, coefficients = (
        np.asarray(modal),
        np.asarray(dry),
        np.asarray(coefficients),
    )
    response, dry_response = modal @ monitor_shapes, dry @ monitor_shapes
    displacement = (modal @ shapes.T).reshape(len(frequencies), len(x), len(y))
    pressure = (coefficients @ pressure_basis.T).reshape(
        len(frequencies), len(px), len(py)
    )
    peak = int(np.argmax(abs(response)))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "spectrum.npz",
        frequencies_Hz=frequencies,
        x=x,
        y=y,
        displacement=displacement,
        pressure_x=px,
        pressure_y=py,
        pressure=pressure,
        modal_displacement=modal,
        dry_modal_displacement=dry,
        monitor_displacement=response,
        pressure_coefficients=coefficients,
        impedance=np.asarray(impedances),
        modal_projection=C,
        collocation=points,
        check_points=check_points,
        mass=modes.mass,
        stiffness=modes.stiffness,
        generalized_force=force,
        dry_frequencies_Hz=modes.dry_omega / (2 * np.pi),
        dry_mode_shapes=shapes.reshape(len(x), len(y), args.modes),
        dry_mode_dofs=modes.dof_modes,
        dof_coordinates=plate.function_space.tabulate_dof_coordinates(),
        saved_H_indices=np.array([i for i, _ in saved_H]),
        saved_H=np.array([h for _, h in saved_H]),
        transverse_degrees=basis.transverse_degrees,
    )
    np.savetxt(
        args.output / "monitor_spectrum.csv",
        np.column_stack(
            (
                frequencies,
                response.real,
                response.imag,
                abs(response),
                np.angle(response),
            )
        ),
        delimiter=",",
        header="frequency_Hz,real_m,imag_m,amplitude_m,phase_rad",
        comments="",
    )
    report = {
        "configuration": vars(args) | {"output": str(args.output)},
        "coefficient_count": basis.count,
        "matrix_shape": [len(points), basis.count],
        "dry_frequencies_Hz": (modes.dry_omega / (2 * np.pi)).tolist(),
        "dry_eigen_residuals": modes.eigen_residuals.tolist(),
        "mode_reflection": parity,
        "force_projection": projection_report,
        "diagnostics": diagnostics,
        "sampled_peak_Hz": float(frequencies[peak]),
        "sampled_peak_m": float(abs(response[peak])),
        "fluid_sweep_seconds": perf_counter() - started,
        "source_hashes": before,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    summary = (
        "# Weighted-pressure Kirchhoff-Love plate spectrum\n\n"
        f"{args.boundary_condition} plate; {args.mesh_x} x {args.mesh_y} crossed P2 mesh; {args.modes} dry FEM modes.\n\n"
        f"M={args.M}, K={args.K}, including all y degrees 0..K: {basis.count} coefficients; H shape {len(points)} x {basis.count}.\n\n"
        f"- Load: {args.load}, amplitude {args.force:g} N at x/L={args.force_x}, y/W={args.force_y}.\n"
        f"- Dry frequencies (Hz): {report['dry_frequencies_Hz']}.\n"
        f"- Measured mode parities: {[p['parity'] for p in parity]}.\n"
        f"- Largest sampled monitor response: {abs(response[peak]) * 1e9:.5g} nm at {frequencies[peak]:.5g} Hz.\n"
        f"- Maximum independent-point response error: {max(r['response_heldout_error'] for r in diagnostics):.3%}.\n"
        f"- Maximum reciprocity defect: {max(r['reciprocity_error'] for r in diagnostics):.3%}.\n"
        f"- Minimum resistance eigenvalue: {min(r['minimum_resistance_eigenvalue_Ns_per_m'] for r in diagnostics):.6g} N s/m.\n"
        f"- Fluid sweep elapsed: {report['fluid_sweep_seconds']:.2f} s.\n\n"
        "The frequency grid, structural mesh, modal truncation, and pressure degree need separate convergence checks. "
        "This is a sampled peak, not a refined resonance. Quadrature agreement is an empirical estimate. "
        "The isotropic unbounded fluid sheet has no separate support or finite-thickness fluid faces. "
        "Production source hashes were unchanged.\n"
    )
    (args.output / "report.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    if not args.no_plots:
        plots(
            args.output,
            frequencies,
            response,
            dry_response,
            x,
            y,
            displacement[peak],
            px,
            py,
            pressure[peak],
            shapes.reshape(len(x), len(y), args.modes),
            modes.dry_omega / (2 * np.pi),
            frequencies[peak],
        )


def plots(
    output,
    frequencies,
    response,
    dry_response,
    x,
    y,
    displacement,
    px,
    py,
    pressure,
    shapes,
    dry_hz,
    peak_hz,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    axes[0].loglog(
        frequencies[frequencies > 0],
        abs(response[frequencies > 0]) * 1e9,
        "o-",
        label="3D weighted fluid + KL",
    )
    axes[0].loglog(
        frequencies[frequencies > 0],
        abs(dry_response[frequencies > 0]) * 1e9,
        "--",
        label="Dry KL",
    )
    axes[0].set(ylabel="Monitor displacement (nm)")
    axes[0].legend()
    axes[1].semilogx(
        frequencies[frequencies > 0],
        np.unwrap(np.angle(response[frequencies > 0])),
        "o-",
    )
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")
    fig.savefig(output / "monitor_spectrum.png", dpi=150)
    plt.close(fig)
    for name, field, xx, yy, unit in (
        ("displacement", displacement * 1e9, x, y, "nm"),
        ("pressure", pressure, px, py, "Pa"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, component, label in zip(
            axes, (field.real, field.imag), ("Real", "Imaginary")
        ):
            limit = max(np.max(abs(component)), 1e-300)
            im = ax.pcolormesh(
                xx * 1e6,
                yy * 1e6,
                component.T,
                shading="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            ax.set(
                xlabel="x (µm)",
                ylabel="y (µm)",
                title=f"{label} {name}, {peak_hz:.1f} Hz",
                aspect="equal",
            )
            fig.colorbar(im, ax=ax, label=unit)
        fig.savefig(output / f"{name}.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(
        1,
        len(dry_hz),
        figsize=(4 * len(dry_hz), 4),
        squeeze=False,
        constrained_layout=True,
    )
    for i, ax in enumerate(axes[0]):
        im = ax.pcolormesh(
            x * 1e6,
            y * 1e6,
            shapes[:, :, i].T,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            shading="auto",
        )
        ax.set(
            title=f"Mode {i + 1}: {dry_hz[i] / 1000:.2f} kHz",
            xlabel="x (µm)",
            ylabel="y (µm)",
            aspect="equal",
        )
        fig.colorbar(im, ax=ax)
    fig.savefig(output / "dry_modes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
