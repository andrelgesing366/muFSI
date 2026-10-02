"""Compare analytic F3D and Quadpy F3D, including time and peak memory.

From the repository, with the existing NumPy/SciPy/legacy-quadpy environment:
    PYTHONPATH=src .venv/bin/python examples/legacy/compare_stokes_3d.py --quick --plot

Every trial runs in a fresh process. Both models use the same uniform-x,
Chebyshev-y grid and symmetry. Timing workers have no allocation tracing;
separate memory workers measure OS high-water RSS and traced allocation peaks.
This is a fluid-model comparison, not a coupled FEM spectrum or grid convergence
study. A comparison outside --agreement or a failed quadrature exits nonzero.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import tracemalloc
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from mufsi.hydrodynamics.legacy.stokes_3d import Stokes3D
from mufsi.hydrodynamics.legacy.stokes_3d_analytic import (
    Stokes3DAnalytic,
    analytic_fluid_grid,
)
from mufsi.models.fluid import Fluid
from mufsi.models.geometry import PlateGeometry
from mufsi.solvers.linear import SciPyLUSolver

MODELS = ("analytic", "quadpy")
MIB = 1024**2


def velocity_fields(grid, geometry):
    """Unit-amplitude rigid, bending, and torsional transverse velocities."""
    x, y = grid.points.T
    bending = (x / geometry.length) ** 2
    return np.column_stack(
        (np.ones(len(x)), bending, (1 + 0.5j) * bending * 2 * y / geometry.width)
    )


def make_model(name, grid, *, tolerance, solver=None):
    """Construct either implementation with identical physical inputs."""
    fluid = Fluid(997.0, 890e-6)
    common = {
        "fluid": fluid,
        "grid": grid,
        "solver": solver,
        "tolerance": tolerance,
        "absolute_tolerance": 1e-15,
        "batch_size": 256,
        "use_symmetry": True,
    }
    if name == "analytic":
        return Stokes3DAnalytic(**common)
    if name == "quadpy":
        return Stokes3D(**common, quadrature_backend="quadpy")
    raise ValueError(f"Unknown model: {name}")


def relative_error(actual, reference):
    """Relative Euclidean/Frobenius norm, with an absolute floor for zeros."""
    return float(
        np.linalg.norm(actual - reference)
        / max(
            np.linalg.norm(reference),
            np.finfo(float).tiny,
        )
    )


def compare_arrays(analytic, quadpy, weights, velocity):
    """Compare complex mobility, pressure, force and generalized resistance.

    Quadpy is the comparison baseline, not an exact solution. Generalized
    resistance V^H Q P avoids a misleading relative net-force error for the
    antisymmetric torsional field, whose integrated net force is nearly zero.
    """
    ba, pa = analytic
    bq, pq = quadpy
    delta = np.abs(ba - bq)
    magnitude = np.abs(bq)
    significant = magnitude > 1e-10 * np.max(magnitude)
    resistance_a = velocity.conj().T @ (weights[:, None] * pa)
    resistance_q = velocity.conj().T @ (weights[:, None] * pq)
    pressure_errors = [
        relative_error(pa[:, i], pq[:, i]) for i in range(velocity.shape[1])
    ]
    force_a, force_q = weights @ pa[:, 0], weights @ pq[:, 0]
    return {
        "matrix_relative_frobenius": relative_error(ba, bq),
        "matrix_max_relative_entry": float(
            np.max(delta[significant] / magnitude[significant])
        ),
        "pressure_rigid_relative_l2": pressure_errors[0],
        "pressure_bending_relative_l2": pressure_errors[1],
        "pressure_torsion_relative_l2": pressure_errors[2],
        "resistance_relative_frobenius": relative_error(resistance_a, resistance_q),
        "rigid_force_relative": float(abs(force_a - force_q) / abs(force_q)),
    }


class MeasuredLUSolver(SciPyLUSolver):
    """Split LU setup from the first pressure_from_velocity call."""

    factorization_seconds = 0.0

    def factorize(self, matrix):
        started = perf_counter()
        super().factorize(matrix)
        self.factorization_seconds = perf_counter() - started


def process_memory():
    """Return current RSS and high-water RSS in bytes, or unavailable values."""
    if sys.platform.startswith("linux"):
        fields = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, number, _ = line.split()
                fields[key] = int(number) * 1024
        return fields.get("VmRSS:"), fields.get("VmHWM:")
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if get_memory(get_process(), ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
        return None, None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return None, int(rss if sys.platform == "darwin" else rss * 1024)
    except ImportError:
        return None, None


def run_worker(config):
    """Execute one model's timing or memory trial; never mix the two modes."""
    geometry = PlateGeometry(config["length"], config["width"], 5e-6)
    grid = analytic_fluid_grid(geometry, nx=config["nx"], ny=config["ny"])
    velocity = velocity_fields(grid, geometry)
    # Import/rule/BLAS warmup is outside the measurement, equally for both models.
    warm_grid = analytic_fluid_grid(geometry, nx=2, ny=3)
    warm = make_model(config["model"], warm_grid, tolerance=config["tolerance"])
    warm.pressure_from_velocity(2 * np.pi * config["frequencies"][0], np.ones(6))
    warm.clear_cache()
    del warm, warm_grid
    gc.collect()
    baseline_rss, baseline_peak = process_memory()
    trace_allocations = config["measurement"] == "allocations"
    if trace_allocations:
        tracemalloc.start()
    solver = MeasuredLUSolver()
    started = perf_counter()
    model = make_model(
        config["model"], grid, tolerance=config["tolerance"], solver=solver
    )
    setup_seconds = perf_counter() - started
    rows = []
    for index, frequency in enumerate(config["frequencies"]):
        omega = 2 * np.pi * frequency
        gc.collect()
        started = perf_counter()
        matrix = model.assemble_matrix(omega)
        assembly = perf_counter() - started
        started = perf_counter()
        pressure = model.pressure_from_velocity(omega, velocity)
        first_pressure = perf_counter() - started
        factorization = solver.factorization_seconds
        repeats = 100 if config["measurement"] == "timing" else 1
        started = perf_counter()
        for _ in range(repeats):
            model.pressure_from_velocity(omega, velocity)
        cached_solve = (perf_counter() - started) / repeats
        traced_peak = tracemalloc.get_traced_memory()[1] if trace_allocations else None
        _, rss_peak = process_memory()
        residuals = np.linalg.norm(
            matrix @ pressure - velocity, axis=0
        ) / np.linalg.norm(velocity, axis=0)
        if not np.isfinite(pressure).all() or np.max(residuals) > 1e-10:
            raise RuntimeError("Non-finite pressure or excessive no-slip residual.")
        row = {
            "frequency_Hz": frequency,
            "assembly_seconds": assembly,
            "factorization_seconds": factorization,
            "first_solve_seconds": max(0.0, first_pressure - factorization),
            "cached_solve_seconds": cached_solve,
            "assembly_and_first_pressure_seconds": assembly + first_pressure,
            "dense_matrix_bytes": matrix.nbytes,
            "cached_block_bytes": getattr(model, "_blocks", np.empty(0)).nbytes,
            "baseline_rss_bytes": baseline_rss,
            "baseline_peak_rss_bytes": baseline_peak,
            "peak_rss_bytes": rss_peak,
            "peak_rss_growth_bytes": None
            if rss_peak is None or baseline_peak is None
            else max(0, rss_peak - baseline_peak),
            "traced_peak_bytes": traced_peak,
            "no_slip_max_relative": float(np.max(residuals)),
            "integration_report": asdict(model.integration_report),
        }
        rows.append(row)
        if config.get("save_arrays"):
            np.savez_compressed(
                Path(config["output"]) / f"frequency_{index}.npz",
                matrix=matrix,
                pressure=pressure,
                velocity=velocity,
                points=grid.points,
                weights=grid.weights,
                resistance=velocity.conj().T @ (grid.weights[:, None] * pressure),
            )
        del matrix, pressure
        model.clear_cache()
    if trace_allocations:
        tracemalloc.stop()
    result = {"config": config, "setup_seconds": setup_seconds, "rows": rows}
    Path(config["output"], "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


def write_csv(path, rows):
    """Save a table with stable field order and actual numeric values."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment_metadata():
    versions = {}
    for distribution in ("numpy", "scipy", "legacy-quadpy", "quadpy", "matplotlib"):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "versions": versions,
        "worker_thread_environment": {
            key: "1"
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
        "harmonic_convention": "exp(+i omega t)",
        "grid": "uniform x midpoints / Chebyshev y; same symmetry for both models",
        "timing": "median of fresh-process trials; small backend/BLAS warmup excluded; no tracemalloc",
        "memory": "untraced RSS worker, plus separate allocation-tracing worker; peak RSS includes interpreter/libraries; growth is above warmup high-water; traced peak includes Python and registered NumPy allocations, not every native allocation",
        "scope": "fluid matrices, pressures and generalized resistances; no coupled FEM spectrum",
        "source_sha256": implementation_hashes(),
    }


def implementation_hashes():
    root = Path(__file__).resolve().parents[2]
    files = (
        "src/mufsi/hydrodynamics/legacy/stokes_3d.py",
        "src/mufsi/hydrodynamics/legacy/panel_quadrature.py",
        "src/mufsi/hydrodynamics/stokeslet.py",
        "src/mufsi/hydrodynamics/legacy/stokes_3d_analytic.py",
        "src/mufsi/hydrodynamics/legacy/panel_analytic.py",
        "src/mufsi/hydrodynamics/legacy/grid_analytic.py",
        "src/mufsi/hydrodynamics/grid.py",
        "src/mufsi/solvers/linear.py",
    )
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
    }


def save_plot(output, summaries, comparisons):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["scenario"] for row in summaries]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for model, shift, color in (
        ("analytic", -0.18, "#176d98"),
        ("quadpy", 0.18, "#ce6633"),
    ):
        for axis, key, scale in (
            (axes[0, 0], "assembly_median_seconds", 1),
            (axes[0, 1], "factorization_median_seconds", 1e3),
            (axes[1, 0], "traced_peak_MiB", 1),
        ):
            axis.bar(
                positions + shift,
                [row[f"{model}_{key}"] * scale for row in summaries],
                width=0.36,
                label=model,
                color=color,
            )
    for axis, ylabel in zip(
        axes.ravel()[:3],
        (
            "Assembly per frequency [s]",
            "LU per frequency [ms]",
            "Peak traced allocations [MiB]",
        ),
    ):
        axis.set(xticks=positions, xticklabels=labels, ylabel=ylabel)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    for scenario in labels:
        rows = [r for r in comparisons if r["scenario"] == scenario]
        axes[1, 1].loglog(
            [r["frequency_Hz"] / 1e3 for r in rows],
            [r["matrix_relative_frobenius"] for r in rows],
            "o-",
            label=scenario,
        )
    axes[1, 1].set(xlabel="Frequency [kHz]", ylabel="Relative mobility difference")
    axes[1, 1].grid(True, which="both", alpha=0.2)
    axes[1, 1].legend(fontsize=8)
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="6x12 and 12x24 grids, three frequencies."
    )
    parser.add_argument("--case", choices=("slender", "wide", "both"), default="both")
    parser.add_argument("--grids", nargs="+", help="Grid sizes, e.g. 6x12 12x24.")
    parser.add_argument("--frequencies", type=float, nargs="+", default=[1e3, 1e4, 1e5])
    parser.add_argument(
        "--repeats", type=int, default=3, help="Fresh timing trials per model/scenario."
    )
    parser.add_argument("--tolerance", type=float, default=2e-3)
    parser.add_argument(
        "--agreement",
        type=float,
        default=1e-2,
        help="Maximum allowed relative difference.",
    )
    parser.add_argument(
        "--timeout", type=float, default=180, help="Seconds allowed per worker."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/stokes_3d_comparison")
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        run_worker(json.loads(args.worker.read_text()))
        return 0
    if args.repeats < 1 or args.timeout <= 0 or not np.isfinite(args.timeout):
        parser.error("Require repeats >= 1 and a finite positive timeout.")
    if (
        not np.isfinite([args.tolerance, args.agreement]).all()
        or min(args.tolerance, args.agreement) <= 0
    ):
        parser.error("Require finite positive tolerance and agreement.")
    if not np.isfinite(args.frequencies).all() or min(args.frequencies) <= 0:
        parser.error("frequencies must be finite and positive Hz.")
    grids = args.grids or (["6x12", "12x24"] if args.quick else ["12x24", "24x36"])
    shapes = []
    for spec in grids:
        try:
            nx, ny = map(int, spec.lower().split("x"))
        except ValueError:
            parser.error("Grid sizes must have the form 6x12.")
        if nx < 1 or ny < 1 or nx * ny > 4096:
            parser.error("Require positive grid sizes with at most 4096 points.")
        shapes.append((nx, ny))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[key] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (
                str(Path(__file__).resolve().parents[2] / "src"),
                env.get("PYTHONPATH"),
            ),
        )
    )
    cases = ("slender", "wide") if args.case == "both" else (args.case,)
    comparisons, summaries, frequency_timings, failures = [], [], [], []
    detailed = {
        "environment": environment_metadata(),
        "parameters": {
            "frequencies_Hz": args.frequencies,
            "repeats": args.repeats,
            "tolerance": args.tolerance,
            "agreement": args.agreement,
            "fluid_SI": asdict(Fluid(997.0, 890e-6)),
        },
        "scenarios": [],
    }
    for case in cases:
        for nx, ny in shapes:
            scenario = f"{case}_{nx}x{ny}"
            records = {}
            geometry = PlateGeometry(
                500e-6, 50e-6 if case == "slender" else 250e-6, 5e-6
            )
            print(f"{scenario}: {nx * ny} points, rtol={args.tolerance:g}", flush=True)
            for measurement in ("timing", "rss", "allocations"):
                repeats = args.repeats if measurement == "timing" else 1
                for trial in range(repeats):
                    # Alternate the order to reduce systematic machine-load bias.
                    order = MODELS if trial % 2 == 0 else MODELS[::-1]
                    for model in order:
                        destination = (
                            output / "raw" / scenario / model / f"{measurement}_{trial}"
                        )
                        destination.mkdir(parents=True, exist_ok=True)
                        config = {
                            "model": model,
                            "measurement": measurement,
                            "length": geometry.length,
                            "width": geometry.width,
                            "nx": nx,
                            "ny": ny,
                            "tolerance": args.tolerance,
                            "frequencies": args.frequencies,
                            "output": str(destination),
                            "save_arrays": measurement == "timing" and trial == 0,
                        }
                        with tempfile.TemporaryDirectory(dir=output) as temp:
                            config_path = Path(temp) / "worker.json"
                            config_path.write_text(json.dumps(config))
                            try:
                                subprocess.run(
                                    [
                                        sys.executable,
                                        str(Path(__file__).resolve()),
                                        "--worker",
                                        str(config_path),
                                    ],
                                    env=env,
                                    check=True,
                                    capture_output=True,
                                    text=True,
                                    timeout=args.timeout,
                                )
                            except (
                                subprocess.CalledProcessError,
                                subprocess.TimeoutExpired,
                            ) as error:
                                detail = getattr(error, "stderr", None) or str(error)
                                raise RuntimeError(
                                    f"{scenario}/{model}/{measurement} failed: {detail}"
                                ) from error
                        records.setdefault(
                            model, {"timing": [], "rss": [], "allocations": []}
                        )[measurement].append(
                            json.loads((destination / "metrics.json").read_text()),
                        )
                        print(
                            f"  {model}: {measurement} {trial + 1}/{repeats} complete",
                            flush=True,
                        )
            summary = {"scenario": scenario, "nx": nx, "ny": ny}
            for model in MODELS:
                for source, label in (
                    ("assembly_seconds", "assembly_median_seconds"),
                    ("factorization_seconds", "factorization_median_seconds"),
                    ("cached_solve_seconds", "cached_solve_median_seconds"),
                    ("assembly_and_first_pressure_seconds", "total_median_seconds"),
                ):
                    summary[f"{model}_{label}"] = median(
                        sum(row[source] for row in run["rows"]) / len(args.frequencies)
                        for run in records[model]["timing"]
                    )
                memory = records[model]["rss"][0]["rows"]
                allocations = records[model]["allocations"][0]["rows"]
                for source, label in (
                    ("peak_rss_bytes", "peak_rss_MiB"),
                    ("peak_rss_growth_bytes", "peak_rss_growth_MiB"),
                    ("traced_peak_bytes", "traced_peak_MiB"),
                    ("dense_matrix_bytes", "dense_matrix_MiB"),
                    ("cached_block_bytes", "cached_block_MiB"),
                ):
                    source_rows = (
                        allocations if source == "traced_peak_bytes" else memory
                    )
                    available = [
                        row[source] for row in source_rows if row[source] is not None
                    ]
                    summary[f"{model}_{label}"] = (
                        max(available) / MIB if available else None
                    )
            summary["assembly_speedup"] = (
                summary["quadpy_assembly_median_seconds"]
                / summary["analytic_assembly_median_seconds"]
            )
            summary["total_speedup"] = (
                summary["quadpy_total_median_seconds"]
                / summary["analytic_total_median_seconds"]
            )
            for index, frequency in enumerate(args.frequencies):
                frequency_timing = {"scenario": scenario, "frequency_Hz": frequency}
                for model in MODELS:
                    for key in (
                        "assembly_seconds",
                        "factorization_seconds",
                        "cached_solve_seconds",
                        "assembly_and_first_pressure_seconds",
                    ):
                        frequency_timing[f"{model}_{key}"] = median(
                            run["rows"][index][key] for run in records[model]["timing"]
                        )
                frequency_timing["assembly_speedup"] = (
                    frequency_timing["quadpy_assembly_seconds"]
                    / frequency_timing["analytic_assembly_seconds"]
                )
                frequency_timings.append(frequency_timing)
                paths = [
                    output
                    / "raw"
                    / scenario
                    / model
                    / "timing_0"
                    / f"frequency_{index}.npz"
                    for model in MODELS
                ]
                with np.load(paths[0]) as a, np.load(paths[1]) as q:
                    differences = compare_arrays(
                        (a["matrix"], a["pressure"]),
                        (q["matrix"], q["pressure"]),
                        a["weights"],
                        a["velocity"],
                    )
                    row = {
                        "scenario": scenario,
                        "frequency_Hz": frequency,
                        **differences,
                    }
                    row["passed"] = max(differences.values()) <= args.agreement
                    comparisons.append(row)
                    if not row["passed"]:
                        failures.append(row)
            summaries.append(summary)
            detailed["scenarios"].append(
                {
                    "scenario": scenario,
                    "geometry_SI": asdict(geometry),
                    "summary": summary,
                    "measurements": records,
                }
            )
            worst = max(
                row["matrix_relative_frobenius"]
                for row in comparisons
                if row["scenario"] == scenario
            )
            print(
                f"  assembly speedup {summary['assembly_speedup']:.2f}x; "
                f"total speedup {summary['total_speedup']:.2f}x; "
                f"max matrix difference {worst:.3e}",
                flush=True,
            )
    detailed["comparisons"] = comparisons
    detailed["sources_unchanged"] = (
        implementation_hashes() == detailed["environment"]["source_sha256"]
    )
    passed = not failures and detailed["sources_unchanged"]
    detailed["passed"] = passed
    (output / "report.json").write_text(json.dumps(detailed, indent=2) + "\n")
    write_csv(output / "comparison.csv", comparisons)
    write_csv(output / "benchmark.csv", summaries)
    write_csv(output / "frequency_timings.csv", frequency_timings)
    if args.plot:
        save_plot(output, summaries, comparisons)
    print(f"{'PASS' if passed else 'FAIL'}: results saved to {output}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
