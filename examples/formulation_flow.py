"""Two phase snapshots of the wide plate's antisymmetric 3D-driven 2D field.

Requires the extensive study's antisymmetric.json and the scientific runtime.
The field is a fresh local 2D approximation to the weighted-3D displacement;
its dissipation is not substituted for the weighted-3D Q calculation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.formulation_study import Study, source_hash
from mufsi import FluidGrid, plot_flow, reconstruct_flow_from_response
from mufsi.coupling.weighted import surface_evaluation

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results/formulation_study"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/formulation_study/flow"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads(
        (args.results / "wide/antisymmetric.json").read_text(encoding="utf-8")
    )
    row = next(r for r in report["results"] if r["model"] == "KL_3D_odd")
    frequency = row["fitted_f0_Hz"]
    study = Study("wide")
    study.prepare("KL_3D")
    response, _ = study.solve("KL_3D_odd", [frequency])
    grid = FluidGrid.chebyshev_gauss(study.plate_geometry, nx=33, ny=128)
    g = study.geometry
    y, z = np.meshgrid(
        np.linspace(-0.75 * g.width, 0.75 * g.width, 71),
        np.linspace(-0.15 * g.width, 0.15 * g.width, 72),
    )
    field = reconstruct_flow_from_response(
        study.plate,
        response,
        study.fluid,
        grid,
        np.stack((y, z), axis=-1),
        section_index=-2,
    )
    velocity = (
        2j
        * np.pi
        * frequency
        * (surface_evaluation(study.plate, grid.points) @ response.displacement[0])
    )
    local = velocity.reshape(grid.nx, grid.ny)[-2]
    positive = local[grid.y > 0]
    reference = float(np.angle(positive[np.argmax(abs(positive))]))
    np.savez_compressed(
        args.output / "field.npz",
        frequency_Hz=frequency,
        points=field.points,
        velocity=field.velocity,
        streamfunction=field.streamfunction,
        mean_dissipation=field.mean_dissipation,
        energy_dissipation=field.energy_dissipation,
        full_displacement_m=response.displacement,
        pressure_coefficients=response.pressure_coefficients,
        phase_reference_rad=reference,
    )
    for name, phase in (
        ("in_phase", -reference),
        ("quadrature", -reference - np.pi / 2),
    ):
        fig, _ = plot_flow(field, phase=phase)
        fig.suptitle(
            f"500 × 250 × 5 µm · antisymmetric KL at {frequency / 1e3:.2f} kHz\n"
            f"2D field approximation driven by weighted-3D displacement · {name.replace('_', ' ')}"
        )
        fig.savefig(args.output / f"flow_{name}.png", dpi=200)
        fig.savefig(args.output / f"flow_{name}.pdf")
        plt.close(fig)
    (args.output / "parameters.json").write_text(
        json.dumps(
            {
                "frequency_Hz": frequency,
                "section_x_m": float(grid.x[-2]),
                "source_model": "KL + weighted 3D, opposite corner forces",
                "field_model": "Local 2D approximation, 128 transverse panels",
                "phase_reference_rad": reference,
                "configuration": study.config,
                "source_hash": source_hash(),
                "energy_note": "Plotted field dissipation belongs to the 2D approximation and is not the weighted-3D Q denominator.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved phase snapshots and complex field to {args.output.resolve()}")


if __name__ == "__main__":
    main()
