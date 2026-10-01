"""F2D versus Sader: port of Example_2_F2D_spectrum.

Run in the DOLFINx environment:
    PYTHONPATH=src python examples/cantilever_2d.py
    PYTHONPATH=src python examples/cantilever_2d.py --quick
Defaults reproduce the notebook's geometry, mesh, quadrature, and 200-point sweep.
"""

import argparse
import json
from pathlib import Path

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
)
from mufsi.coupling.basis_evaluation import build_evaluation_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="Smaller mesh/grid and 40 frequencies.",
    )
    parser.add_argument("--samples", type=int, help="Override frequency sample count.")
    parser.add_argument("--output", type=Path, default=Path("results/cantilever_2d"))
    args = parser.parse_args()
    samples = args.samples if args.samples is not None else (40 if args.quick else 200)
    if samples < 2:
        parser.error("--samples must be at least 2.")

    geometry = PlateGeometry(length=500e-6, width=50e-6, thickness=5e-6)
    material = Material(young_modulus=169e9, density=2330, poisson_ratio=0.3)
    fluid = Fluid(density=997, dynamic_viscosity=890e-6)
    plate = KirchhoffPlate(
        geometry, material, mesh_resolution=(24, 4) if args.quick else (64, 6),
        boundary_condition="cantilever", element_degree=2, penalty=16,
    )
    if plate.mesh.comm.size != 1:
        raise RuntimeError("Run this coupled example on one MPI rank.")
    grid = FluidGrid.chebyshev_gauss(
        geometry, nx=16 if args.quick else 32, ny=32 if args.quick else 128,
    )
    problem = CoupledProblem(plate, Stokes2D(fluid, grid))
    frequencies = np.geomspace(1e3, 400e3, samples)
    load = DistributedLoad(lambda x: np.ones(x.shape[1]))  # Uniform 1 Pa.
    print(
        f"F2D: {plate.function_space.dofmap.index_map.size_global} structural DOFs, "
        f"{grid.nx} x {grid.ny} fluid points, {samples} frequencies.", flush=True,
    )

    def progress(done, total):
        if done == 1 or done == total or done % max(1, total // 10) == 0:
            print(f"  Solved {done}/{total}", flush=True)

    result = FrequencyResponseSolver(problem).solve(
        frequencies, load, progress=progress,
    )
    # Match the old notebook's tip at the positive-width edge.
    tip = build_evaluation_matrix(
        plate.function_space, np.array([[geometry.length, geometry.width / 2]]),
    )
    f2d_tip = (tip @ result.displacement.T).ravel()
    sader = SaderMethod(geometry, material, fluid)
    sader_tip = sader.displacement_per_pressure(frequencies).ravel()

    args.output.mkdir(parents=True, exist_ok=True)
    table = np.column_stack((
        frequencies, f2d_tip.real, f2d_tip.imag, sader_tip.real, sader_tip.imag,
        np.abs(f2d_tip), np.abs(sader_tip), result.relative_errors, result.fluid_errors,
    ))
    np.savetxt(
        args.output / "spectrum.csv", table, delimiter=",",
        header="frequency_Hz,f2d_real_m_per_Pa,f2d_imag_m_per_Pa,"
        "sader_real_m_per_Pa,sader_imag_m_per_Pa,f2d_abs_m_per_Pa,"
        "sader_abs_m_per_Pa,equilibrium_relative_error,fluid_relative_error",
        comments="",
    )
    np.savez_compressed(
        args.output / "response.npz", frequencies=frequencies,
        displacement=result.displacement, pressure=result.pressure,
        fluid_points=grid.points, fluid_weights=grid.weights,
        structural_coordinates=plate.function_space.tabulate_dof_coordinates(),
        f2d_tip=f2d_tip, sader_tip=sader_tip,
    )
    metadata = {
        "geometry_SI": vars(geometry), "material_SI": vars(material),
        "fluid_SI": vars(fluid), "mesh_resolution": plate.mesh_resolution,
        "element_degree": plate.element_degree,
        "penalty": plate.penalty, "fluid_nx": grid.nx, "fluid_ny": grid.ny,
        "harmonic_convention": "exp(+i omega t)", "load_pressure_Pa": 1,
        "measurement_point_m": [geometry.length, geometry.width / 2],
        "sader_reference": "doi:10.1063/1.368002, Eqs. 18,20-22,B4",
    }
    (args.output / "parameters.json").write_text(json.dumps(metadata, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.loglog(
        frequencies / 1e3, np.abs(f2d_tip) * 1e9, label="F2D + Kirchhoff–Love plate",
    )
    ax.loglog(
        frequencies / 1e3, np.abs(sader_tip) * 1e9, "--", label="Sader beam reference",
    )
    ax.set(xlabel="Frequency [kHz]", ylabel="Tip displacement [nm/Pa]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(args.output / "spectrum.png", dpi=180)
    plt.close(fig)
    for name, response in (("F2D", f2d_tip), ("Sader", sader_tip)):
        peaks, _ = find_peaks(np.abs(response))
        print(f"{name} sampled peaks [kHz]:", np.round(frequencies[peaks] / 1e3, 3))
    print(f"Maximum equilibrium residual: {result.relative_errors.max():.3e}")
    print(f"Maximum no-slip residual: {result.fluid_errors.max():.3e}")
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
