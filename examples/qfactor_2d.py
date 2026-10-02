"""Corner-driven Q: Euler-Bernoulli/Tuck, Kirchhoff/F2D and Sader in water.

Run in the serial DOLFINx environment:
    PYTHONPATH=src python examples/qfactor_2d.py --quick
    PYTHONPATH=src python examples/qfactor_2d.py

No modal forces or response truncation. Sader's loaded frequency only selects
an isolated sweep window. The comparison uses the first flexural resonances
of an 800 x 50 x 5 um silicon cantilever, with 1 nN at each near-tip corner.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from mufsi import (
    BeamFrequencyResponseSolver,
    BeamGeometry,
    CoupledProblem,
    EulerBernoulliBeam,
    Fluid,
    FluidGrid,
    FrequencyResponseSolver,
    KirchhoffPlate,
    Material,
    PlateGeometry,
    SaderMethod,
    SectionForce2D,
    Stokes2D,
    analyze_q_factor,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="Smaller discretization, one mode."
    )
    parser.add_argument(
        "--modes", type=int, help="First 1 to 4 flexural resonances (default 3)."
    )
    parser.add_argument(
        "--samples", type=int, help="Samples per resonance (default 81, quick 61)."
    )
    parser.add_argument(
        "--fit-background", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--window-linewidths",
        type=float,
        default=1.5,
        help="Total sweep width / Sader linewidth f0/Q (default 1.5).",
    )
    parser.add_argument("--output", type=Path, default=Path("results/qfactor_2d"))
    args = parser.parse_args()
    modes = args.modes if args.modes is not None else (1 if args.quick else 3)
    samples = args.samples if args.samples is not None else (61 if args.quick else 81)
    if not 1 <= modes <= 4 or samples < 21:
        parser.error("Use 1 to 4 modes and at least 21 samples per resonance.")
    if not np.isfinite(args.window_linewidths) or args.window_linewidths <= 0:
        parser.error("--window-linewidths must be positive and finite.")
    geometry = BeamGeometry(800e-6, 50e-6, 5e-6)
    material, fluid = Material(169e9, 2330, 0.3), Fluid(997, 890e-6)
    beam = EulerBernoulliBeam(
        geometry,
        material,
        mesh_resolution=24 if args.quick else 48,
        element_degree=3,
    )
    plate = KirchhoffPlate(
        PlateGeometry(**asdict(geometry)),
        material,
        mesh_resolution=(16, 4) if args.quick else (48, 6),
    )
    if beam.mesh.comm.size != 1:
        parser.error("Run on one MPI rank.")
    grid = FluidGrid.chebyshev_gauss(
        plate.geometry,
        nx=17 if args.quick else 49,
        ny=32 if args.quick else 64,
    )
    solvers = {
        "beam_tuck": BeamFrequencyResponseSolver(
            beam,
            SectionForce2D(geometry, fluid, method="tuck", ny=grid.ny),
        ),
        "plate_f2d": FrequencyResponseSolver(
            CoupledProblem(plate, Stokes2D(fluid, grid))
        ),
        "beam_sader": BeamFrequencyResponseSolver(
            beam,
            SectionForce2D(geometry, fluid, method="sader"),
        ),
    }
    labels = {
        "beam_tuck": "Euler-Bernoulli + Tuck",
        "plate_f2d": "Kirchhoff-Love + F2D",
        "beam_sader": "Euler-Bernoulli + Sader force",
    }
    beta = np.array(
        [1.875104068711961, 4.694091132974175, 7.854757438237612, 10.995540734875467]
    )[:modes]
    vacuum = (
        beta**2
        / (2 * np.pi * geometry.length**2)
        * np.sqrt(
            beam.flexural_rigidity / beam.line_density,
        )
    )
    reference = SaderMethod(geometry, material, fluid)
    loaded, q_sader = reference.resonance_and_q(vacuum)
    rows, arrays, analyses, runtimes = [], {}, {}, {}
    print(
        "Model                          mode    f0 [kHz]   Q SHO   Q energy   Q Sader   fit error",
        flush=True,
    )
    for mode, (center, qref) in enumerate(zip(loaded, q_sader), 1):
        # A local fit limits the frequency variation of other modes' background.
        half = min(0.6 * center, args.window_linewidths * center / qref / 2)
        frequencies = np.linspace(center - half, center + half, samples)
        for name, solver in solvers.items():
            start = perf_counter()
            analysis = analyze_q_factor(
                solver,
                frequencies,
                symmetry="symmetric",
                amplitude=1e-9,
                inset=1e-3,
                fit_background=args.fit_background,
            )
            runtimes[f"{name}_{mode}"] = perf_counter() - start
            analyses[name, mode] = analysis
            fit, energy = analysis.sho, analysis.energy
            qenergy = float(energy.q_factor[0])
            balance = float(energy.work_balance_errors[0])
            rows.append(
                (
                    mode,
                    name,
                    fit.resonance_frequency,
                    analysis.peak_frequency,
                    fit.q_factor,
                    qenergy,
                    float(qref),
                    float(center),
                    fit.relative_error,
                    balance,
                    analysis.response.relative_errors.max(),
                )
            )
            print(
                f"{labels[name]:30s} {mode:3d} {fit.resonance_frequency / 1e3:11.4f} "
                f"{fit.q_factor:8.4f} {qenergy:10.4f} {qref:9.4f} "
                f"{100 * fit.relative_error:9.3f}%",
                flush=True,
            )
            prefix = f"{name}_mode{mode}_"
            arrays.update(
                {
                    prefix + "frequencies_Hz": frequencies,
                    prefix + "observable_m": analysis.observable,
                    prefix + "fit_m": fit.fitted_amplitudes,
                    prefix + "displacement_m": analysis.response.displacement,
                    prefix
                    + "resonance_displacement_m": analysis.resonance_response.displacement,
                    prefix + "stored_energy_J": energy.stored_energy,
                    prefix + "work_per_cycle_J": energy.dissipated_energy,
                    prefix + "fluid_work_per_cycle_J": energy.fluid_dissipated_energy,
                }
            )
            if getattr(analysis.response, "pressure", None) is not None:
                arrays[prefix + "pressure_Pa"] = analysis.response.pressure
            if hasattr(analysis.response, "line_force"):
                arrays[prefix + "line_force_N_per_m"] = analysis.response.line_force
    args.output.mkdir(parents=True, exist_ok=True)
    header = "mode,model,f0_Hz,peak_Hz,Q_SHO,Q_energy,Q_Sader,Sader_f0_Hz,fit_relative_error,work_balance_error,equilibrium_error"
    with (args.output / "comparison.csv").open("w", encoding="utf-8") as stream:
        import csv

        writer = csv.writer(stream)
        writer.writerow(header.split(","))
        writer.writerows(rows)
    arrays.update(
        fluid_points_m=grid.points,
        fluid_weights_m2=grid.weights,
        beam_coordinates_m=beam.function_space.tabulate_dof_coordinates(),
        plate_coordinates_m=plate.function_space.tabulate_dof_coordinates(),
    )
    np.savez_compressed(args.output / "response.npz", **arrays)
    import dolfinx
    import scipy

    metadata = {
        "geometry_SI": asdict(geometry),
        "material_SI": asdict(material),
        "fluid_SI": asdict(fluid),
        "beam_elements": beam.mesh_resolution,
        "beam_degree": beam.element_degree,
        "plate_mesh": plate.mesh_resolution,
        "plate_degree": plate.element_degree,
        "fluid_grid": [grid.nx, grid.ny],
        "amplitude_N_per_corner": 1e-9,
        "tip_inset_fraction": 1e-3,
        "symmetry": "symmetric",
        "samples_per_resonance": samples,
        "fit_background": args.fit_background,
        "window_linewidths": args.window_linewidths,
        "harmonic_convention": "exp(+i omega t)",
        "energy_definition": "max structural bending + kinetic energy; input work per cycle",
        "energy_evaluation_frequency": "fitted undamped SHO f0",
        "sader_reference": "https://sadermethod.org/Overview_files/docs/JAP_1998.pdf, Eqs. 33 and 35",
        "vacuum_frequencies_Hz": vacuum.tolist(),
        "runtime_seconds": runtimes,
        "versions": {
            "dolfinx": dolfinx.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    (args.output / "parameters.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        modes, 2, figsize=(12, 4 * modes), squeeze=False, layout="constrained"
    )
    for mode in range(1, modes + 1):
        left, right = axes[mode - 1]
        for i, name in enumerate(solvers):
            analysis = analyses[name, mode]
            maximum = abs(analysis.observable).max()
            (line,) = left.plot(
                analysis.response.frequencies / 1e3,
                abs(analysis.observable) / maximum,
                label=labels[name],
            )
            left.plot(
                analysis.response.frequencies / 1e3,
                analysis.sho.fitted_amplitudes / maximum,
                "--",
                color=line.get_color(),
            )
            right.plot(i - 0.1, analysis.sho.q_factor, "o", color=line.get_color())
            right.plot(
                i + 0.1, analysis.energy.q_factor[0], "s", color=line.get_color()
            )
        right.axhline(
            q_sader[mode - 1], color="black", linestyle=":", label="Sader Eq. 35"
        )
        right.plot([], [], "o", color="gray", label="SHO fit")
        right.plot([], [], "s", color="gray", label="Energy")
        right.set(
            xticks=range(3),
            xticklabels=["Beam/Tuck", "Plate/F2D", "Beam/Sader"],
            ylabel="Q",
        )
        left.set(
            xlabel="Frequency [kHz]",
            ylabel="Normalized corner displacement",
            title=f"Flexural resonance {mode}: solid response; dashed SHO fit",
        )
        for ax in (left, right):
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
    fig.savefig(args.output / "comparison.png", dpi=160)
    plt.close(fig)
    print(f"Results saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
