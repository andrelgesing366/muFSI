"""In-vacuo plate eigenmodes, matching the original Example_1 notebook.

From the repository root, in a FEniCSx environment with mufsi installed:
    python examples/plate_eigenvalue_problem.py
    python examples/plate_eigenvalue_problem.py --plot
    python examples/plate_eigenvalue_problem.py --boundary-condition bridge

Defaults: 500 x 500 x 5 micrometres, E=169 GPa, rho=2330 kg/m^3,
nu=0.3, 64 x 64 crossed cells, P2 elements, and ten modes.
"""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path
from time import perf_counter


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--penalty", type=float, default=16.0)
    parser.add_argument(
        "--boundary-condition", default="cantilever",
        choices=("cantilever", "bridge", "clamped", "simply_supported"),
    )
    parser.add_argument("--factor-solver-type", default=None)
    parser.add_argument("--output", type=Path, default=Path("results/plate_eigenvalue_problem"))
    parser.add_argument("--plot", action="store_true", help="Save mode-shape PNG (serial).")
    args = parser.parse_args(argv)

    import numpy as np
    from dolfinx import fem, io
    from mpi4py import MPI

    from mufsi import EigenSolver, KirchhoffPlate, Material, PlateGeometry

    comm = MPI.COMM_WORLD
    if args.plot and comm.size != 1:
        parser.error("--plot requires a serial run; XDMF export also works with MPI.")
    start = perf_counter()
    plate = KirchhoffPlate(
        geometry=PlateGeometry(length=500e-6, width=500e-6, thickness=5e-6),
        material=Material(young_modulus=169e9, density=2330.0, poisson_ratio=0.3),
        mesh_resolution=(args.nx, args.ny),
        boundary_condition=args.boundary_condition,
        element_degree=args.degree,
        penalty=args.penalty,
        comm=comm,
    )
    result = EigenSolver(
        plate, factor_solver_type=args.factor_solver_type
    ).solve(args.modes)
    elapsed = comm.allreduce(perf_counter() - start, op=MPI.MAX)
    output = args.output.resolve()
    if comm.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        print(f"Isotropic {args.boundary_condition} plate")
        print(f"DOFs: {plate.function_space.dofmap.index_map.size_global}")
        print(" Mode       f [kHz]        relative residual")
        for i, (frequency, error) in enumerate(
            zip(result.frequencies, result.relative_errors), start=1
        ):
            print(f" {i:4d}  {frequency / 1e3:13.6f}    {error:.3e}")
        print(f"Solved in {elapsed:.2f} s ({result.iterations} SLEPc iterations).")
        np.savetxt(
            output / "frequencies.csv",
            np.column_stack((
                np.arange(1, args.modes + 1), result.frequencies,
                result.angular_frequencies, result.relative_errors,
            )),
            delimiter=",",
            header="mode,frequency_Hz,angular_frequency_rad_s,relative_residual",
            comments="", fmt=["%d", "%.12e", "%.12e", "%.6e"],
        )
    comm.barrier()

    # XDMF visualization uses peak-normalized P1 interpolation on the same mesh.
    # The original mass-normalized P2/higher-degree modes remain in result.
    V1 = fem.functionspace(plate.mesh, ("Lagrange", 1))
    sample = fem.Function(V1, name="mode_shape")
    plot_values = []
    with io.XDMFFile(comm, output / "mode_shapes.xdmf", "w") as file:
        file.write_mesh(plate.mesh)
        for function, frequency in zip(result.mode_functions, result.frequencies):
            sample.interpolate(function)
            peak = comm.allreduce(
                float(np.max(np.abs(function.x.array), initial=0)), op=MPI.MAX
            )
            sample.x.array[:] = sample.x.array.real / peak
            sample.x.scatter_forward()
            file.write_function(sample, float(frequency))
            if args.plot:
                plot_values.append(sample.x.array.real.copy())

    if comm.size == 1:
        np.savez_compressed(
            output / "eigenmodes.npz",
            frequencies_Hz=result.frequencies,
            angular_frequencies_rad_s=result.angular_frequencies,
            eigenvalues=result.eigenvalues,
            relative_errors=result.relative_errors,
            modes=result.modes,
            dof_coordinates_m=plate.function_space.tabulate_dof_coordinates(),
        )
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from dolfinx import plot

        cells, _, points = plot.vtk_mesh(V1)
        triangulation = mtri.Triangulation(
            points[:, 0] * 1e6, points[:, 1] * 1e6, cells.reshape(-1, 4)[:, 1:]
        )
        columns = min(3, args.modes)
        rows = ceil(args.modes / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False)
        for i, axis in enumerate(axes.flat):
            if i >= args.modes:
                axis.set_visible(False)
                continue
            image = axis.tripcolor(
                triangulation, plot_values[i], shading="gouraud",
                cmap="RdBu_r", vmin=-1, vmax=1,
            )
            axis.set_title(f"Mode {i + 1}: {result.frequencies[i] / 1e3:.3f} kHz")
            axis.set_xlabel("x [µm]")
            axis.set_ylabel("y [µm]")
            axis.set_aspect("equal")
            fig.colorbar(image, ax=axis, label="normalized displacement")
        fig.tight_layout()
        fig.savefig(output / "mode_shapes.png", dpi=180)
        plt.close(fig)
    if comm.rank == 0:
        print(f"Saved frequencies and mode shapes in {output}")


if __name__ == "__main__":
    main()
