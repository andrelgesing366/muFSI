"""Euler-Bernoulli FEM with local Sader and Tuck section forces.

Based on 1D Cantilever in Fluid/main.py: silicon 800x50x10 um in water,
uniform line loading 1e-3 N/m. Compare full FEM fields with Sader compliance.
Run: PYTHONPATH=src python3 examples/beam_2d.py
     PYTHONPATH=src python3 examples/beam_2d.py --quick
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
    DistributedLoad,
    EulerBernoulliBeam,
    Fluid,
    Material,
    SaderMethod,
    SectionForce2D,
)
from mufsi.coupling.basis_evaluation import build_evaluation_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--nx", type=int, help="Number of beam FEM elements.")
    parser.add_argument("--ny", type=int, help="Tuck section panels, 2 to 64.")
    parser.add_argument("--samples", type=int)
    parser.add_argument(
        "--line-load", type=float, default=1e-3, help="Uniform line load in N/m."
    )
    parser.add_argument(
        "--integration", choices=("panel", "legacy_trapezoid"), default="panel"
    )
    parser.add_argument("--output", type=Path, default=Path("results/beam_2d"))
    args = parser.parse_args()
    nx = args.nx if args.nx is not None else (24 if args.quick else 40)
    ny = args.ny if args.ny is not None else (16 if args.quick else 64)
    samples = args.samples if args.samples is not None else (48 if args.quick else 200)
    if nx < 2 or not 2 <= ny <= 64 or samples < 2 or not np.isfinite(args.line_load):
        parser.error("Require nx>=2, 2<=ny<=64, samples>=2 and a finite line load.")
    beam = EulerBernoulliBeam(
        BeamGeometry(800e-6, 50e-6, 10e-6),
        Material(169e9, 2330, 0.3),
        mesh_resolution=nx,
        element_degree=3,
    )
    if beam.mesh.comm.size != 1:
        parser.error("The beam response example requires one MPI rank.")
    fluid = Fluid(997, 890e-6)
    frequencies = np.geomspace(1e3, 500e3, samples)
    load = DistributedLoad(lambda x: np.full(x.shape[1], args.line_load))
    x = np.linspace(0, beam.geometry.length, 201)
    E = build_evaluation_matrix(beam.function_space, x[:, None])
    models = {
        "sader": SectionForce2D(beam.geometry, fluid, method="sader"),
        "tuck": SectionForce2D(
            beam.geometry, fluid, method="tuck", ny=ny, integration=args.integration
        ),
    }
    curves, results, runtimes = {}, {}, {}
    for name, model in models.items():
        started = perf_counter()
        result = BeamFrequencyResponseSolver(beam, model).solve(frequencies, load)
        results[name] = result
        curves[name] = E @ result.displacement.T
        runtimes[name] = perf_counter() - started
        print(
            f"{name}: {runtimes[name]:.2f} s, maximum force residual "
            f"{result.relative_errors.max():.2e}"
        )
    reference = (
        SaderMethod(beam.geometry, beam.material, fluid)
        .displacement_per_line_force(frequencies, x)
        .T
        * args.line_load
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "response.npz",
        frequencies_Hz=frequencies,
        x_m=x,
        sader_fem=curves["sader"].T,
        tuck_fem=curves["tuck"].T,
        sader_analytic=reference.T,
        sader_displacement=results["sader"].displacement,
        tuck_displacement=results["tuck"].displacement,
        sader_line_force=results["sader"].line_force,
        tuck_line_force=results["tuck"].line_force,
        structural_coordinates_m=beam.function_space.tabulate_dof_coordinates(),
        section_points_m=models["tuck"].section_grid.points,
        section_edges_m=models["tuck"].section_grid.panel_edges,
    )
    np.savetxt(
        args.output / "spectrum.csv",
        np.column_stack(
            (
                frequencies,
                reference[-1].real,
                reference[-1].imag,
                curves["sader"][-1].real,
                curves["sader"][-1].imag,
                curves["tuck"][-1].real,
                curves["tuck"][-1].imag,
                results["sader"].relative_errors,
                results["tuck"].relative_errors,
            )
        ),
        delimiter=",",
        comments="",
        header="frequency_Hz,analytic_real_m,analytic_imag_m,sader_fem_real_m,"
        "sader_fem_imag_m,tuck_fem_real_m,tuck_fem_imag_m,sader_force_error,tuck_force_error",
    )
    import dolfinx
    import scipy

    metadata = {
        "geometry_SI": asdict(beam.geometry),
        "material_SI": asdict(beam.material),
        "fluid_SI": asdict(fluid),
        "line_load_N_per_m": args.line_load,
        "mesh_elements": nx,
        "element_degree": beam.element_degree,
        "tuck_panels": ny,
        "pressure_integration": args.integration,
        "harmonic_convention": "exp(+i omega t)",
        "force_sign": "resisting line force; actual fluid force is its negative",
        "runtime_seconds": runtimes,
        "versions": {
            "dolfinx": dolfinx.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source": "1D Cantilever in Fluid/main.py, Cantilever_in_Fluid.py; "
        "PlAFeM/beam/beam_fluid_coupling.py and fluid/fluid_dynamics.py",
    }
    (args.output / "parameters.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for name, data, style in (
        ("Analytical Sader beam", reference[-1], "-"),
        ("FEM + Sader force", curves["sader"][-1], "--"),
        ("FEM + Tuck force", curves["tuck"][-1], ":"),
    ):
        axes[0].loglog(frequencies / 1e3, abs(data) * 1e9, style, label=name)
        axes[1].semilogx(frequencies / 1e3, np.angle(data), style, label=name)
    axes[0].set(
        xlabel="Frequency [kHz]",
        ylabel="Tip displacement [nm]",
        title="Uniform line loading: 1D beam in water",
    )
    axes[1].set(
        xlabel="Frequency [kHz]", ylabel="Phase [rad]", title="Tip response phase"
    )
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend()
    fig.savefig(args.output / "spectrum.png", dpi=180)
    plt.close(fig)
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
