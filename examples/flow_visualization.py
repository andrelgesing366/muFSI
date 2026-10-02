"""F2D section flow: prescribed displacement demo or saved coupled pressure.

PYTHONPATH=src python examples/flow_visualization.py
PYTHONPATH=src python examples/flow_visualization.py --mode rigid
PYTHONPATH=src python examples/flow_visualization.py \
    --response results/cantilever_2d/response.npz --frequency 100000

Needs NumPy, SciPy, and Matplotlib, no FEM backend or 3D fluid solve.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mufsi import (
    Fluid,
    FluidGrid,
    PlateGeometry,
    Stokes2D,
    plot_flow,
    reconstruct_flow,
    reconstruct_section,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--response", type=Path, help="F2D response.npz from cantilever_2d.py."
    )
    parser.add_argument(
        "--frequency",
        type=float,
        help="Hz; nearest saved frequency, or demo frequency.",
    )
    parser.add_argument(
        "--section", type=int, default=-1, help="x section index (default: last)."
    )
    parser.add_argument(
        "--mode", choices=("rigid", "beam", "roof-tile"), default="roof-tile"
    )
    parser.add_argument(
        "--panels", type=int, default=128, help="Demo transverse pressure panels."
    )
    parser.add_argument(
        "--ny", type=int, default=241, help="Observation points across y."
    )
    parser.add_argument(
        "--nz", type=int, default=120, help="Observation points across z."
    )
    parser.add_argument("--output", type=Path, default=Path("results/flow_2d"))
    args = parser.parse_args()
    if min(args.ny, args.nz) < 2 or args.panels < 2:
        parser.error("--ny, --nz, and --panels must be at least 2.")
    if args.frequency is not None and (
        not np.isfinite(args.frequency) or args.frequency <= 0
    ):
        parser.error("--frequency must be finite and positive.")

    if args.response is None:
        geometry = PlateGeometry(800e-6, 400e-6, 5e-6)
        fluid = Fluid(997, 890e-6)
        grid = FluidGrid.chebyshev_gauss(geometry, nx=3, ny=args.panels)
        hz = 300e3 if args.frequency is None else args.frequency
        # A prescribed displacement, not a computed structural eigenmode.
        x, y = grid.points.T
        shape = np.ones_like(y)
        if args.mode == "beam":
            shape = (x / geometry.length) ** 2
        elif args.mode == "roof-tile":
            shape = (x / geometry.length) ** 2 * np.cos(
                3 * np.pi * (y / geometry.width + 0.5)
            )
        phase = np.exp(0.3j * y / geometry.width) if args.mode == "roof-tile" else 1
        displacement = 1e-9 * shape * phase
        hydro = Stokes2D(fluid, grid)
        pressure = hydro.pressure_from_velocity(
            2 * np.pi * hz, 1j * 2 * np.pi * hz * displacement
        )
        source = f"Prescribed {args.mode} displacement, peak amplitude 1 nm"
    else:
        metadata = json.loads(args.response.with_name("parameters.json").read_text())
        if (
            metadata.get("method", "Stokes2D") not in {"Stokes2D", "F2D"}
            or "fluid_3d_shape" in metadata
            or "fluid_nx" not in metadata
            or "fluid_ny" not in metadata
        ):
            parser.error("Use a saved F2D response from cantilever_2d.py.")
        geometry = PlateGeometry(**metadata["geometry_SI"])
        fluid = Fluid(**metadata["fluid_SI"])
        with np.load(args.response) as data:
            frequencies = data["frequencies"]
            index = (
                0
                if args.frequency is None
                else int(np.argmin(abs(frequencies - args.frequency)))
            )
            hz = float(frequencies[index])
            points, weights = data["fluid_points"], data["fluid_weights"]
            ny = int(metadata["fluid_ny"])
            nx = int(metadata["fluid_nx"])
            nodes = points[:ny, 1]
            edges = (
                data["fluid_panel_edges"]
                if "fluid_panel_edges" in data
                else np.concatenate(
                    (
                        [-geometry.width / 2],
                        (nodes[:-1] + nodes[1:]) / 2,
                        [geometry.width / 2],
                    )
                )
            )
            grid = FluidGrid(points, weights, edges, nx, ny)
            pressure = data["pressure"][index].copy()
        hydro = Stokes2D(fluid, grid)
        source = str(args.response.resolve())
    if not -grid.nx <= args.section < grid.nx:
        parser.error("--section must identify an existing fluid x section.")

    omega = 2 * np.pi * hz
    # Even nz avoids sampling the singular pressure-panel endpoints on z=0.
    y = np.linspace(-0.75 * geometry.width, 0.75 * geometry.width, args.ny)
    z = np.linspace(-0.15 * geometry.width, 0.15 * geometry.width, args.nz)
    field = reconstruct_section(
        omega, y, z, pressure, hydro, section_index=args.section
    )
    section_p = pressure.reshape(grid.nx, grid.ny)[args.section]
    wall_velocity = hydro.section_mobility(omega) @ section_p
    wall = reconstruct_flow(
        omega,
        np.column_stack((grid.y, np.zeros(grid.ny))),
        pressure,
        hydro,
        section_index=args.section,
    )
    residual = np.linalg.norm(wall.velocity[:, 1] - wall_velocity) / max(
        np.linalg.norm(wall_velocity),
        np.finfo(float).tiny,
    )
    # Use one scalar phase reference; do not discard spatial phase differences.
    reference_phase = float(np.angle(wall_velocity[np.argmax(abs(wall_velocity))]))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "flow.npz",
        frequency_Hz=hz,
        section_x_m=grid.x[args.section],
        points=field.points,
        streamfunction=field.streamfunction,
        velocity=field.velocity,
        strain_rate=field.strain_rate,
        mean_dissipation=field.mean_dissipation,
        energy_dissipation=field.energy_dissipation,
        singular_points=field.singular_points,
        phase_reference_rad=reference_phase,
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, phase in (
        ("in_phase", -reference_phase),
        ("quadrature", -reference_phase - np.pi / 2),
    ):
        fig, _ = plot_flow(field, phase=phase)
        fig.suptitle(
            f"F2D at {hz / 1e3:.3g} kHz, x = {grid.x[args.section] * 1e6:.3g} µm ({name})"
        )
        fig.savefig(args.output / f"flow_{name}.png", dpi=180)
        plt.close(fig)
    metadata = {
        "source": source,
        "method": "Stokes2D",
        "frequency_Hz": hz,
        "section_index": args.section,
        "section_x_m": float(grid.x[args.section]),
        "geometry_SI": vars(geometry),
        "fluid_SI": vars(fluid),
        "harmonic_convention": "exp(+i omega t)",
        "pressure_convention": "resisting traction",
        "phase_reference_rad": reference_phase,
        "surface_velocity_relative_error": float(residual),
        "mean_dissipation_units": "W/m^3",
        "energy_dissipation_units": "J/m^3 per cycle",
    }
    (args.output / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(source)
    print(f"Surface velocity relative error: {residual:.3e}")
    print(f"Flow fields and phase plots saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
