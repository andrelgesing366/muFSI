"""Full FE spectral displacement: padded multigrid versus Quadpy.

Run: PYTHONPATH=src .venv/bin/python examples/legacy/plate_multigrid.py
Both methods use the same coarse, edge-refined pressure grid and area weights.
A third Quadpy run on its native Chebyshev grid measures discretization effects
separately; this latter comparison is not a test of matrix assembly accuracy.
Requires the existing DOLFINx/PETSc/Quadpy environment, on one MPI rank.
"""

import argparse
import json
from dataclasses import asdict
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
)
from mufsi.coupling.basis_evaluation import build_evaluation_matrix
from mufsi.hydrodynamics.legacy.stokes_3d import Stokes3D
from mufsi.hydrodynamics.legacy.stokeslet_multigrid import (
    Stokes3DMultigrid,
    multigrid_fluid_grid,
)


def row_relative(a, b):
    return np.linalg.norm(a - b, axis=1) / np.maximum(
        np.linalg.norm(b, axis=1), np.finfo(float).tiny
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-partitions", default="3,3")
    parser.add_argument("--y-partitions", default="3,3")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--f-min", type=float, default=1e3)
    parser.add_argument("--f-max", type=float, default=400e3)
    parser.add_argument("--width", type=float, default=50e-6)
    parser.add_argument("--mesh-x", type=int, default=24)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-refinements", type=int, default=20)
    parser.add_argument("--agreement", type=float, default=1e-4)
    parser.add_argument(
        "--force-residual-tolerance",
        type=float,
        default=1e-6,
        help="Load-relative FE equilibrium residual; stiffness is ill-conditioned.",
    )
    parser.add_argument("--skip-native-grid", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/multigrid_spectrum")
    )
    args = parser.parse_args()
    if (
        args.samples < 2
        or args.mesh_x < 2
        or not np.isfinite(
            [
                args.f_min,
                args.f_max,
                args.width,
                args.agreement,
                args.force_residual_tolerance,
            ]
        ).all()
        or min(args.f_min, args.width, args.agreement, args.force_residual_tolerance)
        <= 0
        or args.f_max <= args.f_min
    ):
        parser.error("Require >=2 samples, mesh-x>=2 and positive ordered frequencies.")
    geometry = PlateGeometry(500e-6, args.width, 5e-6)
    material, fluid = Material(169e9, 2330, 0.3), Fluid(997, 890e-6)
    mesh_y = max(4, round(args.mesh_x * args.width / geometry.length))
    plate = KirchhoffPlate(geometry, material, mesh_resolution=(args.mesh_x, mesh_y))
    grid = multigrid_fluid_grid(
        geometry,
        x_partitions=tuple(int(n) for n in args.x_partitions.split(",")),
        y_partitions=tuple(int(n) for n in args.y_partitions.split(",")),
    )
    frequencies = np.geomspace(args.f_min, args.f_max, args.samples)
    load = DistributedLoad(lambda x: np.ones(x.shape[1]))
    tip = build_evaluation_matrix(
        plate.function_space, [[geometry.length, geometry.width / 2]]
    )
    models = {
        "multigrid": Stokes3DMultigrid(
            fluid,
            grid,
            tolerance=args.tolerance,
            max_refinements=args.max_refinements,
            max_subpanels=65536,
        ),
        "quadpy_shared": Stokes3D(
            fluid,
            grid,
            tolerance=args.tolerance,
            max_refinements=args.max_refinements,
            max_subpanels=65536,
        ),
    }
    if not args.skip_native_grid:
        native = FluidGrid.cantilever(geometry, nx=grid.nx, ny=grid.ny)
        models["quadpy_native"] = Stokes3D(
            fluid,
            native,
            tolerance=args.tolerance,
            max_refinements=args.max_refinements,
            max_subpanels=65536,
        )
    print(
        f"Pressure grid {grid.nx}x{grid.ny}; {args.samples} frequencies; "
        f"{plate.function_space.dofmap.index_map.size_global} structural DOFs",
        flush=True,
    )
    results, tips, times, reports = {}, {}, {}, {}
    for name, hydro in models.items():
        started = perf_counter()
        reports[name] = []

        def progress(done, total, name=name, hydro=hydro, started=started):
            reports[name].append(asdict(hydro.integration_report))
            if done == 1 or done == total or done % 8 == 0:
                print(
                    f"{name}: {done}/{total}, {perf_counter() - started:.1f} s",
                    flush=True,
                )

        result = FrequencyResponseSolver(CoupledProblem(plate, hydro)).solve(
            frequencies,
            load,
            progress=progress,
        )
        times[name] = perf_counter() - started
        results[name] = result
        tips[name] = np.asarray(tip @ result.displacement.T).ravel()
        hydro.clear_cache()
    m, q = results["multigrid"], results["quadpy_shared"]
    displacement_error = row_relative(m.displacement, q.displacement)
    pressure_error = row_relative(m.pressure, q.pressure)
    # Normalize tip errors by the maximum reference amplitude across the sweep
    # as well as reporting pointwise relative errors (which grow at antiresonance).
    tip_error = abs(tips["multigrid"] - tips["quadpy_shared"]) / np.maximum(
        abs(tips["quadpy_shared"]), np.finfo(float).tiny
    )
    residuals = {
        name: {
            "force": float(r.relative_errors.max()),
            "no_slip": float(r.fluid_errors.max()),
        }
        for name, r in results.items()
    }
    errors = {
        "max_full_displacement_relative_l2": float(displacement_error.max()),
        "max_pressure_relative_l2": float(pressure_error.max()),
        "max_tip_relative": float(tip_error.max()),
        "tip_relative_global_scale": float(
            np.max(abs(tips["multigrid"] - tips["quadpy_shared"]))
            / np.max(abs(tips["quadpy_shared"]))
        ),
    }
    # Keep a separate equilibrium criterion: subtracting large stiffness terms
    # against a 1 Pa load loses digits even when the two methods agree closely.
    passed = max(errors.values()) <= args.agreement and all(
        r["force"] < args.force_residual_tolerance and r["no_slip"] < 1e-8
        for r in residuals.values()
    )
    native_difference = None
    if "quadpy_native" in results:
        native_difference = {
            "max_full_displacement_relative_l2": float(
                row_relative(
                    m.displacement, results["quadpy_native"].displacement
                ).max()
            ),
            "tip_relative_global_scale": float(
                np.max(abs(tips["multigrid"] - tips["quadpy_native"]))
                / np.max(abs(tips["quadpy_native"]))
            ),
            "interpretation": "Different grids and force quadrature; discretization difference.",
        }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "passed": passed,
        "agreement": args.agreement,
        "force_residual_tolerance": args.force_residual_tolerance,
        "no_slip_residual_tolerance": 1e-8,
        "geometry_SI": asdict(geometry),
        "fluid_SI": asdict(fluid),
        "material_SI": asdict(material),
        "mesh_resolution": [args.mesh_x, mesh_y],
        "fluid_shape_x_y": [grid.nx, grid.ny],
        "x_partitions": args.x_partitions,
        "y_partitions": args.y_partitions,
        "load_pressure_Pa": 1,
        "tolerance": args.tolerance,
        "max_refinements": args.max_refinements,
        "harmonic_convention": "exp(+i omega t)",
        "shared_grid_errors": errors,
        "native_grid_difference": native_difference,
        "maximum_residuals": residuals,
        "runtime_seconds": times,
        "integration_reports": reports,
    }
    (output / "report.json").write_text(json.dumps(metadata, indent=2) + "\n")
    arrays = {
        "frequencies_Hz": frequencies,
        "points": grid.points,
        "weights": grid.weights,
        "structural_coordinates": plate.function_space.tabulate_dof_coordinates(),
    }
    for name, result in results.items():
        arrays.update(
            {
                f"{name}_displacement": result.displacement,
                f"{name}_pressure": result.pressure,
                f"{name}_tip": tips[name],
                f"{name}_force_residual": result.relative_errors,
                f"{name}_no_slip_residual": result.fluid_errors,
                f"{name}_points": models[name].grid.points,
                f"{name}_weights": models[name].grid.weights,
            }
        )
    np.savez_compressed(output / "response.npz", **arrays)
    columns = [frequencies, displacement_error, pressure_error, tip_error]
    headers = [
        "frequency_Hz",
        "displacement_relative",
        "pressure_relative",
        "tip_relative",
    ]
    for name, response in tips.items():
        columns.extend([response.real, response.imag, abs(response)])
        headers.extend(
            [f"{name}_real_m_per_Pa", f"{name}_imag_m_per_Pa", f"{name}_abs_m_per_Pa"]
        )
    np.savetxt(
        output / "spectrum.csv",
        np.column_stack(columns),
        delimiter=",",
        comments="",
        header=",".join(headers),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    styles = {"multigrid": "-", "quadpy_shared": "--", "quadpy_native": ":"}
    for name, response in tips.items():
        axes[0].loglog(frequencies / 1e3, abs(response) * 1e9, styles[name], label=name)
    axes[0].set(
        xlabel="Frequency [kHz]",
        ylabel="Tip displacement [nm/Pa]",
        title=f"Full plate response, coarse {grid.nx}x{grid.ny} fluid grid",
    )
    axes[0].legend()
    for error, label in (
        (displacement_error, "Full displacement"),
        (pressure_error, "Pressure"),
        (tip_error, "Tip"),
    ):
        axes[1].loglog(frequencies / 1e3, np.maximum(error, 1e-16), label=label)
    axes[1].axhline(args.agreement, color="black", linestyle=":", label="Acceptance")
    axes[1].set(
        xlabel="Frequency [kHz]",
        ylabel="Relative difference",
        title="Multigrid versus Quadpy on identical panels",
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    fig.savefig(output / "spectrum.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "shared_grid_errors": errors,
                "native_grid_difference": native_difference,
                "maximum_residuals": residuals,
                "runtime_seconds": times,
            },
            indent=2,
        )
    )
    print(f"{'PASS' if passed else 'FAIL'}: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
