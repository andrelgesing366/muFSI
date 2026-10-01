"""Adaptive F3D versus F2D and Sader for slender and wide cantilevers.

Run from the repository with the DOLFINx/Quadpy environment:
    PYTHONPATH=src .venv/bin/python examples/plate_3d.py --quick
    PYTHONPATH=src .venv/bin/python examples/plate_3d.py
Use --quadrature gauss to run without Quadpy. Every comparison grid is <=32x64.
"""

import argparse
import json
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np

from mufsi import (
    CoupledProblem,
    DistributedLoad,
    Fluid,
    FluidGrid,
    FrequencyResponseSolver,
    KirchhoffPlate,
    Material,
    PlateGeometry,
    SaderMethod,
    Stokes2D,
    Stokes3D,
)
from mufsi.coupling.basis_evaluation import build_evaluation_matrix


def run_case(name, args, nx, ny, samples, output):
    length = 500e-6
    width = 50e-6 if name == "slender" else 250e-6
    geometry = PlateGeometry(length, width, 5e-6)
    material = Material(169e9, 2330, 0.3)
    fluid = Fluid(997, 890e-6)
    mesh_x = 24 if args.quick else 40
    mesh_y = max(4, round(mesh_x * width / length))
    plate = KirchhoffPlate(geometry, material, mesh_resolution=(mesh_x, mesh_y))
    if plate.mesh.comm.size != 1:
        raise RuntimeError("Run the coupled spectrum example on one MPI rank.")
    grid3 = FluidGrid.cantilever(
        geometry,
        nx=nx,
        ny=ny,
        x_uniform=args.uniform_x,
    )
    # Simpson requires odd count; use the previous odd count to respect nx<=32.
    grid2 = FluidGrid.chebyshev_gauss(geometry, nx=nx if nx % 2 else nx - 1, ny=ny)
    fmax = (
        args.f_max
        if args.f_max is not None
        else (400e3 if name == "slender" else 150e3)
    )
    frequencies = np.geomspace(args.f_min, fmax, samples)
    load = DistributedLoad(lambda x: np.ones(x.shape[1]))
    tip = build_evaluation_matrix(
        plate.function_space,
        [[length, width / 2]],
    )
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"{name}: {length * 1e6:g} x {width * 1e6:g} um, "
        f"{plate.function_space.dofmap.index_map.size_global} FE DOFs, "
        f"F3D {grid3.nx}x{grid3.ny}, F2D {grid2.nx}x{grid2.ny}, "
        f"{samples} frequencies.",
        flush=True,
    )
    reports = []
    h3 = Stokes3D(
        fluid,
        grid3,
        tolerance=args.tolerance,
        quadrature_backend=args.quadrature,
    )
    started = perf_counter()

    def progress3(done, total):
        reports.append(asdict(h3.integration_report))
        if done == 1 or done == total or done % max(1, total // 10) == 0:
            print(
                f"  F3D {done}/{total}, elapsed {perf_counter() - started:.1f} s, "
                f"panel error/target {h3.integration_report.max_error_ratio:.3f}",
                flush=True,
            )

    r3 = FrequencyResponseSolver(CoupledProblem(plate, h3)).solve(
        frequencies,
        load,
        progress=progress3,
    )
    duration3 = perf_counter() - started
    started2 = perf_counter()
    r2 = CoupledProblem(plate, Stokes2D(fluid, grid2)).frequency_response(
        frequencies, load
    )
    duration2 = perf_counter() - started2
    response3 = (tip @ r3.displacement.T).ravel()
    response2 = (tip @ r2.displacement.T).ravel()
    sader = (
        SaderMethod(geometry, material, fluid)
        .displacement_per_pressure(frequencies)
        .ravel()
    )
    table = np.column_stack(
        (
            frequencies,
            response3.real,
            response3.imag,
            response2.real,
            response2.imag,
            sader.real,
            sader.imag,
            abs(response3),
            abs(response2),
            abs(sader),
            r3.relative_errors,
            r3.fluid_errors,
            r2.relative_errors,
            r2.fluid_errors,
        )
    )
    np.savetxt(
        output / "spectrum.csv",
        table,
        delimiter=",",
        comments="",
        header="frequency_Hz,f3d_real_m_per_Pa,f3d_imag_m_per_Pa,"
        "f2d_real_m_per_Pa,f2d_imag_m_per_Pa,sader_real_m_per_Pa,sader_imag_m_per_Pa,"
        "f3d_abs_m_per_Pa,f2d_abs_m_per_Pa,sader_abs_m_per_Pa,"
        "f3d_force_error,f3d_no_slip_error,f2d_force_error,f2d_no_slip_error",
    )
    np.savez_compressed(
        output / "response.npz",
        frequencies=frequencies,
        f3d_tip=response3,
        f2d_tip=response2,
        sader_tip=sader,
        f3d_displacement=r3.displacement,
        f3d_pressure=r3.pressure,
        f2d_displacement=r2.displacement,
        f2d_pressure=r2.pressure,
        f3d_points=grid3.points,
        f2d_points=grid2.points,
        f3d_weights=grid3.weights,
        f2d_weights=grid2.weights,
        f3d_panel_bounds=grid3.panel_bounds,
        structural_coordinates=plate.function_space.tabulate_dof_coordinates(),
    )
    import dolfinx
    import matplotlib
    import scipy

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from petsc4py import PETSc
    from scipy.signal import find_peaks

    metadata = {
        "geometry_SI": asdict(geometry),
        "material_SI": asdict(material),
        "fluid_SI": asdict(fluid),
        "mesh_resolution": plate.mesh_resolution,
        "fluid_3d_shape": [grid3.nx, grid3.ny],
        "fluid_2d_shape": [grid2.nx, grid2.ny],
        "quadrature_backend": args.quadrature,
        "panel_relative_tolerance": args.tolerance,
        "uniform_x": args.uniform_x,
        "load_pressure_Pa": 1,
        "measurement_point_m": [length, width / 2],
        "harmonic_convention": "exp(+i omega t)",
        "runtime_seconds": {"f3d": duration3, "f2d": duration2},
        "versions": {
            "dolfinx": dolfinx.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "petsc": PETSc.Sys.getVersion(),
        },
        "manuscript": "doi:10.1016/j.compfluid.2025.106677",
        "sader_reference": "doi:10.1063/1.368002",
        "integration_reports": reports,
    }
    if args.quadrature == "quadpy":
        for distribution in ("legacy-quadpy", "quadpy"):
            try:
                metadata["versions"][distribution] = version(distribution)
            except PackageNotFoundError:
                pass
    (output / "parameters.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(8.5, 5), layout="constrained")
    ax.loglog(frequencies / 1e3, abs(response3) * 1e9, label="F3D + plate", linewidth=2)
    ax.loglog(frequencies / 1e3, abs(response2) * 1e9, "--", label="F2D + plate")
    label = (
        "Sader beam reference"
        if name == "slender"
        else "Sader slender-beam extrapolation"
    )
    ax.loglog(frequencies / 1e3, abs(sader) * 1e9, ":", label=label)
    ax.set(
        title=f"{name.capitalize()} cantilever: {length * 1e6:g} x {width * 1e6:g} um",
        xlabel="Frequency [kHz]",
        ylabel="Tip displacement [nm/Pa]",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(output / "spectrum.png", dpi=180)
    plt.close(fig)
    for model, response in (("F3D", response3), ("F2D", response2), ("Sader", sader)):
        indices, _ = find_peaks(abs(response))
        print(
            f"  {model} sampled peaks [kHz]: {np.round(frequencies[indices] / 1e3, 3)}"
        )
    print(
        f"  Maximum F3D force/no-slip residual: {r3.relative_errors.max():.2e} / "
        f"{r3.fluid_errors.max():.2e}",
        flush=True,
    )
    h3.clear_cache()
    return frequencies, response3, response2, sader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("both", "slender", "wide"), default="both")
    parser.add_argument(
        "--quick", action="store_true", help="6x12 fluid grid, 24 frequencies."
    )
    parser.add_argument("--nx", type=int, help="F3D x count, between 3 and 32.")
    parser.add_argument("--ny", type=int, help="Fluid y count, between 1 and 64.")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--tolerance", type=float, default=2e-3)
    parser.add_argument("--quadrature", choices=("quadpy", "gauss"), default="quadpy")
    parser.add_argument(
        "--uniform-x",
        action="store_true",
        help="Use uniform x panels with translation reuse.",
    )
    parser.add_argument("--f-min", type=float, default=1e3)
    parser.add_argument("--f-max", type=float)
    parser.add_argument("--output", type=Path, default=Path("results/plate_3d"))
    args = parser.parse_args()
    nx = args.nx if args.nx is not None else (6 if args.quick else 12)
    ny = args.ny if args.ny is not None else (12 if args.quick else 24)
    samples = args.samples if args.samples is not None else (24 if args.quick else 72)
    if not 3 <= nx <= 32 or not 1 <= ny <= 64:
        parser.error("Comparison grids require 3 <= nx <= 32 and 1 <= ny <= 64.")
    if samples < 2 or not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("Require samples >= 2 and a finite positive tolerance.")
    if not np.isfinite(args.f_min) or args.f_min <= 0:
        parser.error("--f-min must be finite and positive.")
    if args.f_max is not None and (
        not np.isfinite(args.f_max) or args.f_max <= args.f_min
    ):
        parser.error("--f-max must be finite and greater than --f-min.")
    if args.f_max is None and args.f_min >= 150e3:
        parser.error("Set --f-max explicitly for --f-min >= 150000.")
    cases = ("slender", "wide") if args.case == "both" else (args.case,)
    curves = [
        run_case(name, args, nx, ny, samples, args.output / name) for name in cases
    ]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(cases),
        figsize=(8 * len(cases), 4.8),
        squeeze=False,
        layout="constrained",
    )
    for ax, name, (f, u3, u2, us) in zip(axes.ravel(), cases, curves):
        ax.loglog(f / 1e3, abs(u3) * 1e9, label="F3D", linewidth=2)
        ax.loglog(f / 1e3, abs(u2) * 1e9, "--", label="F2D")
        ax.loglog(
            f / 1e3,
            abs(us) * 1e9,
            ":",
            label="Sader" if name == "slender" else "Sader extrapolation",
        )
        ax.set(
            title=name.capitalize(),
            xlabel="Frequency [kHz]",
            ylabel="Tip displacement [nm/Pa]",
        )
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
