"""Compare continuous weighted 3D Stokeslet, Sader, and 2D Tuck spectra.

Same beam/load as examples/beam_2d.py; focus on the first bending peak.
Everything, including the order study and generated results, stays in research.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss
from weighted_beam_spectrum import CantileverModes, solve_modal_response
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
from weighted_pressure_projection import modal_projection, source_hashes

from mufsi.hydrodynamics.sader import SaderMethod
from mufsi.hydrodynamics.section_force import SectionForce2D
from mufsi.models.material import Material


def uniform_modal_force(beam, line_load):
    t, w = leggauss(100)
    x = beam.geometry.length * (t + 1) / 2
    return line_load * beam.geometry.length / 2 * (w @ beam.values(x))


def local_modal_impedance(beam, model, omega):
    """Same consistent phi_i*phi_j line integral as the FEM section example."""
    return np.diag(beam.mass / beam.line_mass) * model.line_impedance(omega)


def modal_fluid(solver, beam, points, check_points):
    """Closure with frequency-independent E/C matrices prepared once."""
    E = beam.values(points[:, 0])
    C = modal_projection(solver.basis, beam.values)
    Echeck = beam.values(check_points[:, 0])

    def evaluate(hz):
        H, integration = solver.assemble(2 * np.pi * hz, points)
        A, fit = solve_coefficients(H, E)
        Z = C @ A
        Hcheck, _ = solver.assemble(2 * np.pi * hz, check_points)
        mode_errors = np.linalg.norm(Hcheck @ A - Echeck, axis=0) / np.linalg.norm(
            Echeck, axis=0
        )
        return (
            Z,
            A,
            Hcheck,
            Echeck,
            {
                "integration": integration,
                "fit": fit,
                "modal_heldout_errors": mode_errors.tolist(),
                "reciprocity_error": float(np.linalg.norm(Z - Z.T) / np.linalg.norm(Z)),
                "minimum_resistance": float(
                    np.linalg.eigvalsh((Z + Z.conj().T) / 2).min()
                ),
            },
        )

    return evaluate


def order_study(geometry, fluid, beam, frequencies, quadrature):
    """Compare nested polynomial spaces on exactly the same quadrature/grid.

    These are direct coefficient solves; nothing is fitted to a reference
    pressure field. Shared targets prevent changing collocation density from
    masquerading as pressure-order convergence.
    """
    maximum = WeightedBasis(geometry, M=20, K=6)
    points = maximum.collocation()
    check_points = maximum.collocation(43, 15)[::5]
    solver = WeightedMobility(maximum, fluid, quadrature)
    E, Echeck = beam.values(points[:, 0]), beam.values(check_points[:, 0])
    C = modal_projection(maximum, beam.values)
    force = uniform_modal_force(beam, 1e-3)
    candidates = ((8, 4), (12, 4), (16, 4), (20, 4), (16, 6), (20, 6))
    table = []
    for hz in frequencies:
        H, integration = solver.assemble(2 * np.pi * hz, points)
        Hcheck, _ = solver.assemble(2 * np.pi * hz, check_points)
        solutions = {}
        for M, K in candidates:
            indices = np.array(
                [m * (maximum.K + 1) + k for m in range(M + 1) for k in range(K + 1)]
            )
            A, fit = solve_coefficients(H[:, indices], E)
            Z = C[:, indices] @ A
            d, _ = solve_modal_response(beam, 2 * np.pi * hz, Z, force)
            err = np.linalg.norm(
                Hcheck[:, indices] @ A - Echeck, axis=0
            ) / np.linalg.norm(Echeck, axis=0)
            solutions[M, K] = (Z, d.sum())
            table.append(
                {
                    "frequency_Hz": float(hz),
                    "M": M,
                    "K": K,
                    "coefficients": len(indices),
                    "modal_heldout_errors": err.tolist(),
                    "scaled_condition": fit["scaled_condition"],
                    "tip_real_m": float(d.sum().real),
                    "tip_imag_m": float(d.sum().imag),
                }
            )
        for row in table[-len(candidates) :]:
            Z, tip = solutions[row["M"], row["K"]]
            Zfine, tipfine = solutions[20, 6]
            row["impedance_difference_from_20_6"] = float(
                np.linalg.norm(Z - Zfine) / np.linalg.norm(Zfine)
            )
            row["complex_tip_difference_from_20_6"] = float(
                abs(tip - tipfine) / abs(tipfine)
            )
        print(
            f"Order study {hz:g} Hz: largest H {H.shape}, assembly {integration['seconds']:.2f} s",
            flush=True,
        )
        for row in table[-len(candidates) :]:
            print(
                f"  M={row['M']:2}, K={row['K']}: mode errors="
                f"{np.round(100 * np.array(row['modal_heldout_errors']), 3)}%, "
                f"tip difference={100 * row['complex_tip_difference_from_20_6']:.3f}%",
                flush=True,
            )
    return {
        "common_grid_shape": list(H.shape),
        "rows": table,
        "reference_orders": [20, 6],
        "frequencies_Hz": list(frequencies),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=DEFAULT_M)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--modes", type=int, default=3)
    parser.add_argument("--length", type=float, default=800e-6)
    parser.add_argument("--width", type=float, default=50e-6)
    parser.add_argument("--thickness", type=float, default=10e-6)
    parser.add_argument("--young-modulus", type=float, default=169e9)
    parser.add_argument("--solid-density", type=float, default=2330)
    parser.add_argument("--density", type=float, default=997)
    parser.add_argument("--viscosity", type=float, default=890e-6)
    parser.add_argument("--line-load", type=float, default=1e-3)
    parser.add_argument("--start", type=float, default=1000)
    parser.add_argument("--stop", type=float, default=30000)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--tuck-panels", type=int, default=64)
    parser.add_argument("--backend", choices=("quadpy", "gauss"), default="quadpy")
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--study-only", action="store_true")
    parser.add_argument("--skip-study", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/results/weighted_beam_comparison",
    )
    args = parser.parse_args()
    if args.samples < 3 or not 0 < args.start < args.stop or args.tuck_panels < 2:
        parser.error("Require samples>=3, 0<start<stop, and tuck-panels>=2.")
    if not np.isfinite(args.line_load) or args.line_load == 0:
        parser.error("line-load must be finite and nonzero.")
    before = source_hashes()
    geometry = PlateGeometry(args.length, args.width, args.thickness)
    fluid = Fluid(args.density, args.viscosity)
    material = Material(args.young_modulus, args.solid_density, 0.3)
    beam = CantileverModes(geometry, args.young_modulus, args.solid_density, args.modes)
    basis = WeightedBasis(geometry, args.M, args.K)
    quadrature = Quadrature(backend=args.backend, rtol=args.rtol)
    args.output.mkdir(parents=True, exist_ok=True)
    study = None
    if not args.skip_study:
        study_frequencies = sorted(
            {args.start, float(np.clip(12000.0, args.start, args.stop)), args.stop}
        )
        study = order_study(geometry, fluid, beam, study_frequencies, quadrature)
        (args.output / "order_study.json").write_text(
            json.dumps(study, indent=2), encoding="utf-8"
        )
    if args.study_only:
        if before != source_hashes():
            raise RuntimeError("Production source changed during this run.")
        return
    frequencies = np.linspace(args.start, args.stop, args.samples)
    force = uniform_modal_force(beam, args.line_load)
    models = {
        "sader": SectionForce2D(geometry, fluid, method="sader"),
        "tuck": SectionForce2D(geometry, fluid, method="tuck", ny=args.tuck_panels),
    }
    solver = WeightedMobility(basis, fluid, quadrature)
    points = basis.collocation()
    check_points = basis.collocation(2 * (basis.M + 1) + 1, 2 * (basis.K + 1) + 1)[::5]
    evaluate = modal_fluid(solver, beam, points, check_points)
    x = np.linspace(0, geometry.length, 201)
    shapes = beam.values(x)
    modal = {
        name: np.empty((args.samples, beam.count), complex)
        for name in ("stokes3d", "sader", "tuck")
    }
    coefficients = np.empty((args.samples, basis.count), complex)
    impedances = np.empty((args.samples, beam.count, beam.count), complex)
    diagnostics = []
    started = perf_counter()
    for i, hz in enumerate(frequencies):
        omega = 2 * np.pi * hz
        Z, A, Hcheck, Echeck, row = evaluate(hz)
        impedances[i] = Z
        d, eq = solve_modal_response(beam, omega, Z, force)
        modal["stokes3d"][i] = d
        coefficients[i] = 1j * omega * (A @ d)
        wanted = Echeck @ d
        no_slip = float(
            np.linalg.norm(Hcheck @ (A @ d) - wanted) / np.linalg.norm(wanted)
        )
        row.update(
            frequency_Hz=float(hz), response_heldout_error=no_slip, equilibrium_error=eq
        )
        diagnostics.append(row)
        for name, model in models.items():
            modal[name][i], _ = solve_modal_response(
                beam, omega, local_modal_impedance(beam, model, omega), force
            )
        print(
            f"{i + 1}/{args.samples}: {hz:.1f} Hz, 3D heldout {no_slip:.3%}", flush=True
        )
    fields = {name: d @ shapes.T for name, d in modal.items()}
    tips = {name: d.sum(axis=1) for name, d in modal.items()}
    analytic_sader = (
        SaderMethod(geometry, material, fluid).displacement_per_line_force(
            frequencies, x
        )
        * args.line_load
    )
    sader_modal_error = float(
        np.max(abs(tips["sader"] - analytic_sader[:, -1]) / abs(analytic_sader[:, -1]))
    )
    # Four-mode local response independently checks structural truncation.
    finer_beam = CantileverModes(
        geometry, args.young_modulus, args.solid_density, min(6, args.modes + 1)
    )
    finer_force = uniform_modal_force(finer_beam, args.line_load)
    truncation = {}
    for name, model in models.items():
        fine = np.array(
            [
                solve_modal_response(
                    finer_beam,
                    2 * np.pi * hz,
                    local_modal_impedance(finer_beam, model, 2 * np.pi * hz),
                    finer_force,
                )[0].sum()
                for hz in frequencies
            ]
        )
        truncation[name] = float(np.max(abs(tips[name] - fine) / abs(fine)))
    # Tuck panel refinement uses the same forcing and structural basis.
    tuckfine = SectionForce2D(geometry, fluid, method="tuck", ny=2 * args.tuck_panels)
    fine = np.array(
        [
            solve_modal_response(
                beam,
                2 * np.pi * hz,
                local_modal_impedance(beam, tuckfine, 2 * np.pi * hz),
                force,
            )[0].sum()
            for hz in frequencies
        ]
    )
    tuck_refinement = float(np.max(abs(tips["tuck"] - fine) / abs(fine)))
    if before != source_hashes():
        raise RuntimeError("Production source changed during this run.")
    np.savez_compressed(
        args.output / "comparison.npz",
        frequencies_Hz=frequencies,
        x_m=x,
        stokes3d=fields["stokes3d"],
        sader=fields["sader"],
        tuck=fields["tuck"],
        sader_analytic=analytic_sader,
        pressure_coefficients=coefficients,
        stokes3d_impedance=impedances,
        collocation=points,
        check_points=check_points,
    )
    with (args.output / "tip_spectrum.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["frequency_Hz"]
            + [
                f"{name}_{part}"
                for name in tips
                for part in ("real_m", "imag_m", "abs_m")
            ]
        )
        writer.writerows(
            [hz]
            + [
                v
                for name in tips
                for v in (tips[name][i].real, tips[name][i].imag, abs(tips[name][i]))
            ]
            for i, hz in enumerate(frequencies)
        )
    peaks = {
        name: {
            "sampled_frequency_Hz": float(frequencies[np.argmax(abs(tip))]),
            "sampled_amplitude_m": float(np.max(abs(tip))),
        }
        for name, tip in tips.items()
    }
    report = {
        "configuration": vars(args) | {"output": str(args.output)},
        "geometry": asdict(geometry),
        "fluid": asdict(fluid),
        "matrix_shape": [len(points), basis.count],
        "total_sweep_seconds": perf_counter() - started,
        "peaks": peaks,
        "max_sader_modal_vs_analytic_error": sader_modal_error,
        "local_structural_mode_refinement_errors": truncation,
        "tuck_panel_refinement_error": tuck_refinement,
        "max_response_heldout_error": max(
            r["response_heldout_error"] for r in diagnostics
        ),
        "max_reciprocity_error": max(r["reciprocity_error"] for r in diagnostics),
        "minimum_resistance": min(r["minimum_resistance"] for r in diagnostics),
        "diagnostics": diagnostics,
        "source_hashes": before,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    text = (
        "# Matched beam spectrum comparison\n\n"
        f"Beam {geometry.length * 1e6:g} x {geometry.width * 1e6:g} x {geometry.thickness * 1e6:g} um; uniform line load "
        f"{args.line_load:g} N/m; {beam.count} shared EB modes.\n\n"
        f"3D pressure M={basis.M}, K={basis.K}, {basis.count} coefficients, H={len(points)} x {basis.count}.\n\n"
        "| Fluid model | Largest sampled response frequency (Hz) | Tip amplitude (nm) |\n"
        "|---|---:|---:|\n"
    )
    for name, peak in peaks.items():
        text += f"| {name} | {peak['sampled_frequency_Hz']:.3f} | {peak['sampled_amplitude_m'] * 1e9:.3f} |\n"
    text += (
        f"\nMaximum 3D independent-point velocity error: {report['max_response_heldout_error']:.3%}.\n\n"
        f"Maximum Sader modal/analytical tip difference: {sader_modal_error:.3%}.\n\n"
        f"Tuck {args.tuck_panels} to {2 * args.tuck_panels} panel response difference: {tuck_refinement:.3%}.\n\n"
        "All curves use identical geometry, material, fluid, forcing and phase convention. "
        "The 2D force is the same Tuck/Kelvin section model as examples/beam_2d.py; "
        "the common modal structural model isolates fluid-model differences. "
        "This figure focuses on the first bending peak, rather than the full 500 kHz "
        "range of the FEM example. Peaks are sampled, not fitted resonances.\n"
    )
    (args.output / "report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"stokes3d": "#1864ab", "sader": "#d9480f", "tuck": "#2b8a3e"}
        labels = {
            "stokes3d": f"3D weighted Stokeslet (M={basis.M}, K={basis.K})",
            "sader": "Sader",
            "tuck": "2D Tuck / Kelvin flow",
        }
        fig, axes = plt.subplots(
            2, 1, figsize=(9, 7.5), sharex=True, constrained_layout=True
        )
        for name, tip in tips.items():
            axes[0].plot(
                frequencies / 1e3,
                abs(tip) * 1e9,
                color=colors[name],
                label=labels[name],
                linestyle="-"
                if name == "stokes3d"
                else "--"
                if name == "sader"
                else ":",
                linewidth=2,
            )
            axes[1].plot(
                frequencies / 1e3,
                np.unwrap(np.angle(tip)),
                color=colors[name],
                linestyle="-"
                if name == "stokes3d"
                else "--"
                if name == "sader"
                else ":",
                linewidth=2,
            )
        axes[0].set(
            ylabel="Tip displacement (nm)",
            title="Cantilever in water · identical uniform line forcing",
        )
        axes[0].legend()
        axes[1].set(xlabel="Frequency (kHz)", ylabel="Phase (rad)")
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.savefig(args.output / "spectrum_comparison.png", dpi=180)
        fig.savefig(args.output / "spectrum_comparison.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
