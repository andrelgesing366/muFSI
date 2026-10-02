"""Small full-FE spectra and Q-versus-frequency comparison for five formulations.

Run: PYTHONPATH=src python examples/formulation_comparison.py
Default: first two bending resonances, 25 samples each, modest meshes/orders.
Use --resonances 1 for a shorter run. Increase pressure orders and meshes
independently for convergence studies; this example is a demonstration.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mufsi import (
    BeamFrequencyResponseSolver,
    BeamGeometry,
    CoupledProblem,
    EulerBernoulliBeam,
    Fluid,
    FluidGrid,
    FrequencyResponseResult,
    FrequencyResponseSolver,
    KirchhoffPlate,
    Material,
    PlateGeometry,
    SaderMethod,
    SectionForce2D,
    Stokes2D,
    Stokes3D,
    WeightedCouplingOperator,
    analyze_q_factor,
    plot_flow,
    reconstruct_flow_from_response,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resonances", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--f-min", type=float)
    parser.add_argument("--f-max", type=float)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--x-degree", type=int, default=12)
    parser.add_argument("--y-degree", type=int, default=4)
    parser.add_argument("--beam-mesh", type=int, default=16)
    parser.add_argument("--plate-mesh", type=int, nargs=2, default=(10, 3))
    parser.add_argument(
        "--models", nargs="+", choices=("EB_3D", "KL_3D", "EB_2D", "KL_2D", "EB_Sader")
    )
    parser.add_argument(
        "--symmetry", choices=("symmetric", "antisymmetric"), default="symmetric"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/formulation_comparison")
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots/2D flow from the saved spectra; no frequency sweeps.",
    )
    args = parser.parse_args(argv)
    saved = None
    if args.plots_only:
        saved = json.loads((args.output / "report.json").read_text())
        for key in (
            "resonances",
            "samples",
            "x_degree",
            "y_degree",
            "beam_mesh",
            "plate_mesh",
            "models",
            "symmetry",
            "f_min",
            "f_max",
        ):
            setattr(args, key, saved["parameters"][key])
    if args.samples < 21 or args.x_degree < 1 or args.y_degree < 0:
        parser.error("Use at least 21 samples, x-degree>=1 and y-degree>=0.")
    if args.symmetry == "antisymmetric" and args.y_degree < 1:
        parser.error("Antisymmetric KL motion requires y-degree>=1.")
    geometry = BeamGeometry(800e-6, 50e-6, 5e-6)
    plate_geometry = PlateGeometry(geometry.length, geometry.width, geometry.thickness)
    material, fluid = Material(169e9, 2330, 0.3), Fluid(997, 890e-6)
    beam = EulerBernoulliBeam(geometry, material, mesh_resolution=args.beam_mesh)
    plate = KirchhoffPlate(
        plate_geometry, material, mesh_resolution=tuple(args.plate_mesh)
    )
    grid = FluidGrid.chebyshev_gauss(plate_geometry, nx=17, ny=24)
    solvers = {
        "EB_3D": FrequencyResponseSolver(
            CoupledProblem(
                beam,
                Stokes3D(
                    fluid,
                    geometry,
                    formulation="EB",
                    x_degree=args.x_degree,
                    y_degree=args.y_degree,
                    nx=args.x_degree + 4,
                    ny=args.y_degree // 2 + 2,
                ),
            )
        ),
        "KL_3D": FrequencyResponseSolver(
            CoupledProblem(
                plate,
                Stokes3D(
                    fluid,
                    plate_geometry,
                    formulation="KL",
                    x_degree=args.x_degree,
                    y_degree=args.y_degree,
                    nx=args.x_degree + 4,
                    ny=2 * (args.y_degree + 1),
                ),
            )
        ),
        "EB_2D": BeamFrequencyResponseSolver(
            beam, SectionForce2D(geometry, fluid, ny=24)
        ),
        "KL_2D": FrequencyResponseSolver(CoupledProblem(plate, Stokes2D(fluid, grid))),
        "EB_Sader": BeamFrequencyResponseSolver(
            beam, SectionForce2D(geometry, fluid, method="sader")
        ),
    }
    selected = args.models or (
        list(solvers) if args.symmetry == "symmetric" else ["KL_3D", "KL_2D"]
    )
    if args.symmetry == "antisymmetric" and any(
        name.startswith("EB") for name in selected
    ):
        parser.error("EB supports only symmetric transverse motion; select KL models.")
    labels = {
        "EB_3D": "EB + weighted 3D",
        "KL_3D": "KL + weighted 3D",
        "EB_2D": "EB + 2D",
        "KL_2D": "KL + 2D",
        "EB_Sader": "EB + Sader",
    }
    beta = np.array([1.875104068711961, 4.694091132974175, 7.854757438237612])[
        : args.resonances
    ]
    vacuum = (
        beta**2
        / (2 * np.pi * geometry.length**2)
        * np.sqrt(beam.flexural_rigidity / beam.line_density)
    )
    centers, qs = SaderMethod(geometry, material, fluid).resonance_and_q(vacuum)
    custom_window = args.f_min is not None or args.f_max is not None
    if custom_window and (
        args.f_min is None
        or args.f_max is None
        or not np.isfinite([args.f_min, args.f_max]).all()
        or args.f_min <= 0
        or args.f_max <= args.f_min
        or args.resonances != 1
    ):
        parser.error(
            "Use positive f-min<f-max and --resonances 1 for an explicit window."
        )
    if args.symmetry == "antisymmetric" and not custom_window:
        parser.error(
            "Antisymmetric motion requires an explicit --f-min/--f-max window and --resonances 1."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if saved is not None:
        with np.load(args.output / "spectra.npz") as data:
            arrays = {key: data[key] for key in data.files}
        rows = saved["results"]
        # Force integration is frequency independent; retain its diagnostics
        # when regenerating figures from an earlier saved result schema.
        for name in selected:
            projection_error = None
            if name.endswith("3D"):
                problem = solvers[name].problem
                projection_error = WeightedCouplingOperator.from_structure(
                    problem.structure, problem.hydrodynamics
                ).projection_error
            for row in rows:
                if row["model"] == name:
                    row["force_projection_error"] = projection_error
    else:
        arrays, rows = {}, []
    for resonance, (center, qref) in enumerate(
        zip(centers, qs) if saved is None else [], 1
    ):
        half = min(0.6 * center, 1.8 * center / qref / 2)
        frequencies = (
            np.linspace(args.f_min, args.f_max, args.samples)
            if custom_window
            else np.linspace(center - half, center + half, args.samples)
        )
        for name in selected:
            analysis = analyze_q_factor(
                solvers[name], frequencies, symmetry=args.symmetry, fit_background=True
            )
            fit, energy = analysis.sho, analysis.energy
            no_slip = getattr(analysis.response, "fluid_errors", None)
            error = None if no_slip is None else float(np.nanmax(no_slip))
            rows.append(
                {
                    "model": name,
                    "resonance": resonance,
                    "frequency_Hz": fit.resonance_frequency,
                    "Q_SHO": fit.q_factor,
                    "Q_energy": float(energy.q_factor[0]),
                    "fit_error": fit.relative_error,
                    "no_slip_error": error,
                    "force_projection_error": getattr(
                        analysis.response, "force_projection_error", None
                    ),
                    "work_balance_error": float(energy.work_balance_errors[0]),
                }
            )
            key = f"{name}_{resonance}_"
            arrays.update(
                {
                    key + "frequency_Hz": frequencies,
                    key + "displacement_m": analysis.observable,
                    key + "full_displacement_m": analysis.response.displacement,
                    key + "fit_m": fit.fitted_amplitudes,
                }
            )
            if getattr(analysis.response, "pressure_coefficients", None) is not None:
                arrays[key + "pressure_coefficients_Pa"] = (
                    analysis.response.pressure_coefficients
                )
            print(
                f"{labels[name]:20s} mode {resonance}: f0={fit.resonance_frequency / 1e3:.3f} kHz, "
                f"Q SHO/energy={fit.q_factor:.3f}/{energy.q_factor[0]:.3f}, "
                f"fit={fit.relative_error:.2%}, no-slip={error}",
                flush=True,
            )
    with (args.output / "qfactor.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output / "spectra.npz", **arrays)
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "parameters": vars(args) | {"output": str(args.output)},
                "results": rows,
                "scope": "Modest full-FE demonstration; no continuum convergence claim. Energy Q uses structural stored energy and model-specific fluid work.",
            },
            indent=2,
        )
        + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        args.resonances,
        figsize=(6 * args.resonances, 4),
        squeeze=False,
        layout="constrained",
    )
    for resonance, ax in enumerate(axes[0], 1):
        for name in selected:
            prefix = f"{name}_{resonance}_"
            ax.plot(
                arrays[prefix + "frequency_Hz"] / 1e3,
                abs(arrays[prefix + "displacement_m"]) * 1e9,
                label=labels[name],
            )
        ax.set(
            xlabel="Drive frequency [kHz]",
            ylabel="Corner displacement [nm]",
            title=f"Bending resonance {resonance}",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(args.output / "spectral_displacement.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for name in selected:
        data = [row for row in rows if row["model"] == name]
        f = np.array([row["frequency_Hz"] for row in data]) / 1e3
        (line,) = ax.plot(
            f, [row["Q_SHO"] for row in data], "o-", label=labels[name] + " / SHO"
        )
        ax.plot(
            f,
            [row["Q_energy"] for row in data],
            "x--",
            color=line.get_color(),
            label=labels[name] + " / energy",
        )
    ax.set(xlabel="Fitted resonance frequency [kHz]", ylabel="Q factor")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(args.output / "qfactor_vs_frequency.png", dpi=160)
    plt.close(fig)
    # One compact field example from the 3D displacement, using a 2D pressure solve.
    name = next((n for n in selected if n.endswith("3D")), selected[0])
    prefix = f"{name}_1_"
    response = FrequencyResponseResult(
        arrays[prefix + "frequency_Hz"], arrays[prefix + "full_displacement_m"]
    )
    frequency_index = int(np.argmax(abs(arrays[prefix + "displacement_m"])))
    structure = (
        solvers[name].problem.structure
        if isinstance(solvers[name], FrequencyResponseSolver)
        else beam
    )
    y, z = np.meshgrid(
        np.linspace(-geometry.width, geometry.width, 61),
        np.linspace(-geometry.width, geometry.width, 61),
    )
    field = reconstruct_flow_from_response(
        structure,
        response,
        fluid,
        grid,
        np.stack((y, z), axis=-1),
        frequency_index=frequency_index,
        section_index=-2,
        singular="nan",
    )
    fig, _ = plot_flow(field)
    fig.suptitle(
        labels[name]
        + f" displacement at {response.frequencies[frequency_index] / 1e3:.2f} kHz: 2D field approximation"
    )
    fig.savefig(args.output / "flow_2d_approximation.png", dpi=140)
    plt.close(fig)
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
