"""Assemble H and solve prescribed first-EB-mode pressure, not a panel fit.

Run: python research/weighted_pressure_projection.py --M 4 --K 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from pressure_polynomial import first_mode
from weighted_pressure_mobility import (
    DEFAULT_K,
    DEFAULT_M,
    ROOT,
    Fluid,
    PlateGeometry,
    Quadrature,
    WeightedBasis,
    WeightedMobility,
    solve_coefficients,
)


def source_hashes():
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src").rglob("*.py"))
    }


def add_arguments(parser):
    parser.add_argument("--M", type=int, default=DEFAULT_M, help="Maximum x degree")
    parser.add_argument(
        "--K", type=int, default=DEFAULT_K, help="Even-y index: degree 2*K"
    )
    parser.add_argument("--length", type=float, default=500e-6)
    parser.add_argument("--width", type=float, default=25e-6)
    parser.add_argument("--thickness", type=float, default=5e-6)
    parser.add_argument("--density", type=float, default=997)
    parser.add_argument("--viscosity", type=float, default=890e-6)
    parser.add_argument("--backend", choices=("quadpy", "gauss"), default="quadpy")
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument(
        "--orders", type=int, nargs="+", default=[12, 20, 32, 48, 72, 104]
    )
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny-half", type=int)
    parser.add_argument("--no-plots", action="store_true")


def build_model(args):
    geometry = PlateGeometry(args.length, args.width, args.thickness)
    fluid = Fluid(args.density, args.viscosity)
    basis = WeightedBasis(geometry, args.M, args.K)
    quadrature = Quadrature(args.backend, tuple(args.orders), args.rtol, args.atol)
    return WeightedMobility(basis, fluid, quadrature)


def modal_projection(basis, shape, order=100):
    """C_ij = integral phi_i(x)*psi_j(x,y) dA; only k=0 survives y integral."""
    from numpy.polynomial.legendre import leggauss

    t, weights = leggauss(order)
    alpha, weights = np.pi * (t + 1) / 2, np.pi * weights / 2
    x = basis.geometry.length / 2 * (1 + np.cos(alpha))
    shapes = np.asarray(shape(x))
    if shapes.ndim == 1:
        shapes = shapes[:, None]
    longitudinal = shapes.T @ (
        weights[:, None] * np.cos(alpha[:, None] * np.arange(basis.M + 1))
    )
    result = np.zeros((shapes.shape[1], basis.count))
    result[:, :: basis.K + 1] = (
        np.pi * basis.geometry.length * basis.geometry.width / 4 * longitudinal
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--frequency", type=float, default=None, help="Hz")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/results/weighted_pressure_projection",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Compare modal force to existing analytic panel solver",
    )
    parser.add_argument("--reference-grid", type=int, nargs=2, default=[48, 32])
    args = parser.parse_args()
    before = source_hashes()
    solver = build_model(args)
    b, g, f = solver.basis, solver.basis.geometry, solver.fluid
    hz = (
        f.kinematic_viscosity / (2 * np.pi * (g.width / 2) ** 2)
        if args.frequency is None
        else args.frequency
    )
    points = b.collocation(args.nx, args.ny_half)
    print(
        f"M={b.M}, K={b.K}: {b.count} coefficients, {len(points)} no-slip rows",
        flush=True,
    )
    H, integration = solver.assemble(2 * np.pi * hz, points)
    velocity = first_mode(points[:, 0], g.length)
    started = perf_counter()
    coefficients, fit = solve_coefficients(H, velocity)
    fit["solve_seconds"] = perf_counter() - started
    # Different locations, not rows reused from the fitted matrix.
    nx = args.nx or 2 * (b.M + 1)
    ny = args.ny_half or 2 * (b.K + 1)
    check_points = b.collocation(nx + 1, ny + 1)[::2]
    Hcheck, check_integration = solver.assemble(2 * np.pi * hz, check_points)
    expected = first_mode(check_points[:, 0], g.length)
    predicted = Hcheck @ coefficients
    heldout = float(np.linalg.norm(predicted - expected) / np.linalg.norm(expected))
    # Tighten on representative rows, with a different order sequence.
    subset = np.unique(np.linspace(0, len(points) - 1, 5).astype(int))
    tighter = WeightedMobility(
        b,
        f,
        Quadrature(
            args.backend, (16, 24, 40, 64, 96, 144), args.rtol / 10, args.atol / 10
        ),
    )
    Htight, tight_report = tighter.assemble(2 * np.pi * hz, points[subset])
    sensitivity = float(np.linalg.norm(Htight - H[subset]) / np.linalg.norm(Htight))
    C = modal_projection(b, lambda x: first_mode(x, g.length))
    impedance = complex((C @ coefficients).item())
    report = {
        "configuration": vars(args) | {"output": str(args.output), "frequency": hz},
        "quadrature": asdict(solver.quadrature),
        "integration": integration,
        "fit": fit,
        "heldout_velocity_relative_error": heldout,
        "heldout_integration": check_integration,
        "tightening_H_relative_difference": sensitivity,
        "tightening_integration": tight_report,
        "modal_impedance_Ns_per_m": [impedance.real, impedance.imag],
        "source_hashes": before,
    }
    if args.reference:
        from pressure_polynomial import areas

        from mufsi.hydrodynamics.legacy.stokes_3d_analytic import (
            Stokes3DAnalytic,
            analytic_fluid_grid,
        )

        grid = analytic_fluid_grid(
            g, nx=args.reference_grid[0], ny=args.reference_grid[1]
        )
        hydro = Stokes3DAnalytic(f, grid)
        phi = first_mode(grid.points[:, 0], g.length)
        pref = hydro.pressure_from_velocity(2 * np.pi * hz, phi)
        zref = np.sum(areas(grid) * phi * pref)
        report["panel_reference"] = {
            "grid": args.reference_grid,
            "modal_impedance_Ns_per_m": [float(zref.real), float(zref.imag)],
            "relative_modal_force_difference": float(abs(impedance - zref) / abs(zref)),
        }
    if before != source_hashes():
        raise RuntimeError("Production source changed during the experiment.")
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "projection.npz",
        H=H,
        points=points,
        coefficients=coefficients.reshape(b.M + 1, b.K + 1),
        velocity=velocity,
        predicted=H @ coefficients,
        check_points=check_points,
        check_velocity=expected,
        check_predicted=predicted,
        modal_projection=C,
        frequency=hz,
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    summary = (
        f"# Continuous weighted-pressure projection\n\n"
        f"Frequency: {hz:.6g} Hz. M={b.M}, K={b.K}, {b.count} coefficients.\n\n"
        f"- H: {H.shape}; backend: {args.backend}.\n"
        f"- Assembly: {integration['seconds']:.3f} s; solve: {fit['solve_seconds']:.5f} s.\n"
        f"- Rank: {fit['rank']}; scaled condition: {fit['scaled_condition']:.4g}.\n"
        f"- Collocation velocity error: {fit['relative_residual']:.3%}.\n"
        f"- Independent-point velocity error: {heldout:.3%}.\n"
        f"- H difference after tenfold tolerance tightening: {sensitivity:.3g}.\n"
        f"- First-mode impedance: {impedance:.6g} N s/m.\n\n"
        "These are finite-basis diagnostics; corner behaviour and spectral-order "
        "convergence remain separate questions. Production source hashes were unchanged.\n"
    )
    if args.reference:
        summary += (
            f"\n- Modal-force difference from {args.reference_grid} analytic panels: "
            f"{report['panel_reference']['relative_modal_force_difference']:.3%}.\n"
        )
    (args.output / "report.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.linspace(0.005 * g.length, 0.995 * g.length, 180)
        y = np.linspace(-0.995 * g.width / 2, 0.995 * g.width / 2, 160)
        plot_points = np.column_stack((np.repeat(x, len(y)), np.tile(y, len(x))))
        pressure = (b.values(plot_points) @ coefficients).reshape(len(x), len(y))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, values, name in zip(
            axes, (pressure.real, pressure.imag), ("Real", "Imaginary")
        ):
            artist = ax.pcolormesh(x * 1e6, y * 1e6, values.T, shading="auto")
            ax.set(
                xlabel="x (µm)", ylabel="y (µm)", title=f"{name} p / unit tip velocity"
            )
            fig.colorbar(artist, ax=ax, label="Pa s/m")
        fig.savefig(args.output / "pressure.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
