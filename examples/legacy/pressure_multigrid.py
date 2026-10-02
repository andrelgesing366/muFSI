"""Single-frequency, constant-displacement pressure: multigrid/analytic/Quadpy.

Run: PYTHONPATH=src .venv/bin/python examples/legacy/pressure_multigrid.py
The three models share panels, collocation points, area weights and conventions.
Uniform x is required by the existing analytic implementation; y is refined.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from mufsi import Fluid, PlateGeometry
from mufsi.hydrodynamics.legacy.stokes_3d import Stokes3D
from mufsi.hydrodynamics.legacy.stokes_3d_analytic import Stokes3DAnalytic
from mufsi.hydrodynamics.legacy.stokeslet_multigrid import (
    Stokes3DMultigrid,
    multigrid_fluid_grid,
)


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(float).tiny))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=1e3)
    parser.add_argument("--displacement", type=float, default=1e-9, help="Metres.")
    parser.add_argument("--nx", type=int, default=5, help="Odd uniform x count >=3.")
    parser.add_argument("--y-partitions", default="5,3")
    parser.add_argument("--width", type=float, default=50e-6)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--agreement", type=float, default=1e-4)
    parser.add_argument(
        "--output", type=Path, default=Path("results/multigrid_pressure")
    )
    args = parser.parse_args()
    if (
        not np.isfinite([args.frequency, args.displacement, args.agreement]).all()
        or min(args.frequency, args.displacement, args.agreement) <= 0
    ):
        parser.error("Frequency, displacement and agreement must be positive finite.")
    geometry, fluid = PlateGeometry(500e-6, args.width, 5e-6), Fluid(997, 890e-6)
    grid = multigrid_fluid_grid(
        geometry,
        x_partitions=(args.nx,),
        y_partitions=tuple(int(n) for n in args.y_partitions.split(",")),
    )
    omega = 2 * np.pi * args.frequency
    velocity = np.full(len(grid.points), 1j * omega * args.displacement)
    models = {
        "multigrid": Stokes3DMultigrid(fluid, grid, tolerance=args.tolerance),
        "analytic": Stokes3DAnalytic(fluid, grid, tolerance=args.tolerance),
        "quadpy": Stokes3D(fluid, grid, tolerance=args.tolerance),
    }
    matrices, pressures, durations, residuals, reports = {}, {}, {}, {}, {}
    for name, hydro in models.items():
        started = perf_counter()
        matrices[name] = hydro.assemble_matrix(omega)
        assembly = perf_counter() - started
        pressures[name] = hydro.pressure_from_velocity(omega, velocity)
        durations[name] = {
            "assembly_seconds": assembly,
            "assembly_and_pressure_seconds": perf_counter() - started,
        }
        residuals[name] = relative(matrices[name] @ pressures[name], velocity)
        reports[name] = asdict(hydro.integration_report)
        print(
            f"{name}: {assembly:.3f} s assembly; residual {residuals[name]:.3e}",
            flush=True,
        )
    differences = {}
    root_weights = np.sqrt(grid.weights)
    for reference in ("analytic", "quadpy"):
        pm, pr = pressures["multigrid"], pressures[reference]
        fm, fr = grid.weights @ pm, grid.weights @ pr
        differences[reference] = {
            "matrix_relative_frobenius": relative(
                matrices["multigrid"], matrices[reference]
            ),
            "pressure_relative_l2": relative(pm, pr),
            "pressure_relative_area_l2": relative(root_weights * pm, root_weights * pr),
            "force_relative": relative(np.asarray(fm), np.asarray(fr)),
            "max_pressure_difference_Pa": float(np.max(abs(pm - pr))),
        }
    passed = (
        all(
            values[key] <= args.agreement
            for values in differences.values()
            for key in (
                "matrix_relative_frobenius",
                "pressure_relative_l2",
                "pressure_relative_area_l2",
                "force_relative",
            )
        )
        and max(residuals.values()) < 1e-10
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "passed": passed,
        "agreement": args.agreement,
        "frequency_Hz": args.frequency,
        "constant_displacement_m": args.displacement,
        "geometry_SI": asdict(geometry),
        "fluid_SI": asdict(fluid),
        "pressure_shape_x_y": [grid.nx, grid.ny],
        "weights": "actual panel areas [m^2]",
        "harmonic_convention": "exp(+i omega t)",
        "differences": differences,
        "runtimes": durations,
        "residuals": residuals,
        "integration_reports": reports,
    }
    (output / "report.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(
        output / "pressure.npz",
        points=grid.points,
        weights=grid.weights,
        x_edges=grid.x_panel_edges,
        y_edges=grid.panel_edges,
        velocity=velocity,
        **{f"pressure_{name}": p for name, p in pressures.items()},
        **{f"matrix_{name}": b for name, b in matrices.items()},
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, (name, p) in zip(axes[0], pressures.items()):
        image = ax.pcolormesh(
            grid.x_panel_edges * 1e6,
            grid.panel_edges * 1e6,
            abs(p).reshape(grid.nx, grid.ny).T,
            shading="flat",
        )
        ax.set(title=f"{name}: |pressure|", xlabel="x [um]", ylabel="y [um]")
        fig.colorbar(image, ax=ax, label="Pa")
    center = grid.nx // 2
    for name, p in pressures.items():
        section = p.reshape(grid.nx, grid.ny)[center]
        axes[1, 0].plot(grid.y * 1e6, section.real, ".-", label=name)
        axes[1, 1].plot(grid.y * 1e6, section.imag, ".-", label=name)
    for ax, title in zip(axes[1, :2], ("Real pressure", "Imaginary pressure")):
        ax.set(title=title + " at mid-length", xlabel="y [um]", ylabel="Pa")
        ax.legend()
    for name in ("analytic", "quadpy"):
        delta = abs(pressures["multigrid"] - pressures[name]).reshape(grid.nx, grid.ny)[
            center
        ]
        axes[1, 2].semilogy(grid.y * 1e6, np.maximum(delta, 1e-20), ".-", label=name)
    axes[1, 2].set(title="Absolute multigrid difference", xlabel="y [um]", ylabel="Pa")
    axes[1, 2].legend()
    fig.suptitle(
        f"Constant displacement {args.displacement * 1e9:g} nm, {args.frequency / 1e3:g} kHz"
    )
    fig.savefig(output / "pressure.png", dpi=180)
    plt.close(fig)
    print(json.dumps(differences, indent=2))
    print(f"{'PASS' if passed else 'FAIL'}: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
