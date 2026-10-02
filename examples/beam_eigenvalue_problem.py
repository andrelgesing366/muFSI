"""In-vacuo FEniCSx beam modes versus the analytical cantilever solution.

Legacy geometry: L=2 mm, W=100 um, t=10 um, E=200 GPa, rho=2650 kg/m^3.
Run: PYTHONPATH=src python3 examples/beam_eigenvalue_problem.py --plot
Defaults use 40 cubic elements and six modes. Higher degrees are supported.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from mufsi import BeamGeometry, EigenSolver, EulerBernoulliBeam, Material


def analytical_cantilever(n_modes, x, beam):
    """Roots of cos(beta)+sech(beta)=0 and stable clamped/free mode shapes."""
    beta = np.array(
        [
            brentq(
                lambda b: np.cos(b) + 2 * np.exp(-b) / (1 + np.exp(-2 * b)),
                n * np.pi,
                (n + 1) * np.pi,
            )
            for n in range(n_modes)
        ]
    )
    frequencies = (
        beta**2
        / (2 * np.pi * beam.geometry.length**2)
        * np.sqrt(
            beam.flexural_rigidity / beam.line_density,
        )
    )
    r = x / beam.geometry.length
    shapes = []
    for b in beta:
        e = np.exp(-b)
        denominator = 1 - e**2 + 2 * e * np.sin(b)
        sigma = (1 + e**2 + 2 * e * np.cos(b)) / denominator
        hyperbolic = (
            -np.exp(-b * (2 - r))
            + (np.sin(b) - np.cos(b)) * np.exp(-b * (1 - r))
            + (1 + e * (np.sin(b) + np.cos(b))) * np.exp(-b * r)
        ) / denominator
        w = hyperbolic - np.cos(b * r) + sigma * np.sin(b * r)
        shapes.append(w / w[-1])
    return frequencies, np.array(shapes).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--penalty", type=float, default=16)
    parser.add_argument("--factor-solver-type")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/beam_eigenvalue_problem")
    )
    args = parser.parse_args()
    from dolfinx import fem, io
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if args.plot and comm.size != 1:
        parser.error("--plot requires one MPI rank.")
    if args.modes < 1 or args.nx < 1 or args.degree < 2:
        parser.error("Require modes>=1, nx>=1 and degree>=2.")
    beam = EulerBernoulliBeam(
        BeamGeometry(2e-3, 100e-6, 10e-6),
        Material(200e9, 2650, 0.3),
        mesh_resolution=args.nx,
        element_degree=args.degree,
        penalty=args.penalty,
        comm=comm,
    )
    result = EigenSolver(beam, factor_solver_type=args.factor_solver_type).solve(
        args.modes
    )
    x = np.linspace(0, beam.geometry.length, 501)
    exact, exact_shapes = analytical_cantilever(args.modes, x, beam)
    relative_frequency_error = abs(result.frequencies / exact - 1)
    if comm.rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        print(
            f"Cantilever beam, {beam.function_space.dofmap.index_map.size_global} DOFs"
        )
        print("Mode      FEM [kHz]   analytical [kHz]   frequency error   residual")
        for i, (f, a, e, r) in enumerate(
            zip(
                result.frequencies,
                exact,
                relative_frequency_error,
                result.relative_errors,
            ),
            1,
        ):
            print(f"{i:4d} {f / 1e3:14.6f} {a / 1e3:16.6f} {e:16.3e} {r:12.3e}")
        np.savetxt(
            args.output / "frequencies.csv",
            np.column_stack(
                (
                    np.arange(1, args.modes + 1),
                    result.frequencies,
                    exact,
                    relative_frequency_error,
                    result.relative_errors,
                )
            ),
            delimiter=",",
            comments="",
            header="mode,frequency_Hz,analytic_frequency_Hz,relative_frequency_error,relative_residual",
        )
        import dolfinx

        metadata = {
            "geometry_SI": asdict(beam.geometry),
            "material_SI": asdict(beam.material),
            "mesh_elements": args.nx,
            "element_degree": args.degree,
            "penalty": args.penalty,
            "dolfinx_version": dolfinx.__version__,
            "source": "1D Cantilever in Vacuum/EigenValue.py and EigenValue_v2.py",
            "normalization": "consistent modal mass = 1",
        }
        (args.output / "parameters.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    comm.barrier()
    V1 = fem.functionspace(beam.mesh, ("Lagrange", 1))
    sample = fem.Function(V1)
    with io.XDMFFile(comm, args.output / "mode_shapes.xdmf", "w") as out:
        out.write_mesh(beam.mesh)
        for i, function in enumerate(result.mode_functions):
            sample.name = f"mode_{i + 1}"
            sample.interpolate(function)
            peak = comm.allreduce(
                float(np.max(abs(function.x.array), initial=0)), op=MPI.MAX
            )
            sample.x.array[:] /= peak
            sample.x.scatter_forward()
            out.write_function(sample, float(result.frequencies[i]))
    if comm.size == 1:
        from mufsi.coupling.basis_evaluation import build_evaluation_matrix

        E = build_evaluation_matrix(beam.function_space, x[:, None])
        shapes = E @ result.modes
        shapes /= shapes[-1]
        np.savez_compressed(
            args.output / "eigenmodes.npz",
            frequencies_Hz=result.frequencies,
            analytic_frequencies_Hz=exact,
            relative_errors=result.relative_errors,
            modes=result.modes,
            dof_coordinates_m=beam.function_space.tabulate_dof_coordinates(),
            sampled_x_m=x,
            sampled_modes=shapes,
            analytic_modes=exact_shapes,
        )
        if args.plot:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            columns = min(3, args.modes)
            fig, axes = plt.subplots(
                int(np.ceil(args.modes / columns)),
                columns,
                figsize=(4.5 * columns, 3 * np.ceil(args.modes / columns)),
                squeeze=False,
                layout="constrained",
            )
            for i, ax in enumerate(axes.flat):
                if i >= args.modes:
                    ax.set_visible(False)
                    continue
                ax.plot(x * 1e3, shapes[:, i].real, label="FEniCSx")
                ax.plot(x * 1e3, exact_shapes[:, i], "--", label="Analytical")
                ax.set(
                    title=f"Mode {i + 1}: {result.frequencies[i] / 1e3:.3f} kHz",
                    xlabel="x [mm]",
                    ylabel="Displacement / tip displacement",
                )
                ax.grid(alpha=0.25)
                if i == 0:
                    ax.legend()
            fig.savefig(args.output / "mode_shapes.png", dpi=180)
            plt.close(fig)
    if comm.rank == 0:
        print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
