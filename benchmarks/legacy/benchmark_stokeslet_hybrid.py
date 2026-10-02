"""Staged accuracy/time/memory study for the preserved hybrid Stokeslet model.

Run: PYTHONPATH=src:. python benchmarks/legacy/benchmark_stokeslet_hybrid.py --help

Stages: fixed-panel integration, fluid-grid refinement, full FE spectra.
Every timing repeat and every RSS measurement runs in a fresh process. BLAS
threads are fixed to one. References and uncertainty checks are saved rather
than silently treating a tight quadrature tolerance as continuum accuracy.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mufsi import Fluid, FluidGrid, PlateGeometry
from mufsi.hydrodynamics.legacy.stokes_3d import Stokes3D
from mufsi.hydrodynamics.legacy.stokes_3d_analytic import Stokes3DAnalytic
from mufsi.hydrodynamics.legacy.stokeslet_hybrid import (
    Stokes3DHybrid,
    Stokes3DHybridMultigrid,
    edge_clustered_grid,
    hierarchical_grid,
    lattice_edge_grid,
)
from mufsi.hydrodynamics.legacy.stokeslet_multigrid import Stokes3DMultigrid
from mufsi.solvers.linear import SciPyLUSolver

FLUID = Fluid(997, 890e-6)
FREQUENCIES = [1.0, 1e3, 1e4, 1e5, 4e5]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rss():
    data = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, _ = line.split()
            data[key[:-1]] = int(value) * 1024
    return data


def make_grid(geometry, spec):
    family = spec["family"]
    if family in ("edge", "tip", "translation"):
        return edge_clustered_grid(
            geometry,
            nx=spec["nx"],
            ny=spec["ny"],
            x_clustering={"edge": "both", "tip": "tip", "translation": "uniform"}[
                family
            ],
        )
    if family == "hierarchical":
        return hierarchical_grid(
            geometry,
            x_partitions=spec["x"],
            y_partitions=spec["y"],
            both_x_edges=spec.get("both_x_edges", True),
        )
    if family == "lattice_edge":
        return lattice_edge_grid(geometry, nx=spec["nx"], ny=spec["ny"])
    if family == "native":
        return FluidGrid.cantilever(geometry, nx=spec["nx"], ny=spec["ny"])
    if family == "uniform":
        return FluidGrid.midpoint(geometry, nx=spec["nx"], ny=spec["ny"])
    raise ValueError(f"Unknown grid {family}")


def make_model(method, grid, tolerance, near_ratio=0.5):
    kwargs = {
        "tolerance": tolerance,
        "absolute_tolerance": 1e-16,
        "max_refinements": 24,
        "max_subpanels": 65536,
        "batch_size": 128,
    }
    if method == "quadpy":
        return Stokes3D(FLUID, grid, **kwargs)
    if method == "gauss":
        return Stokes3D(FLUID, grid, quadrature_backend="gauss", **kwargs)
    if method == "multigrid":
        return Stokes3DMultigrid(FLUID, grid, **kwargs)
    if method == "hybrid_multigrid":
        return Stokes3DHybridMultigrid(FLUID, grid, near_ratio=near_ratio, **kwargs)
    if method == "hybrid":
        return Stokes3DHybrid(FLUID, grid, near_ratio=near_ratio, **kwargs)
    if method == "analytic":
        dx = np.diff(grid.x_panel_edges)
        if np.allclose(dx, dx[0], rtol=1e-12, atol=1e-18) and np.allclose(
            grid.x,
            (grid.x_panel_edges[:-1] + grid.x_panel_edges[1:]) / 2,
            rtol=0,
            atol=1e-18,
        ):
            return Stokes3DAnalytic(
                FLUID,
                grid,
                tolerance=tolerance,
                absolute_tolerance=1e-16,
                max_refinements=6,
                batch_size=128,
            )
        # Existing radial primitive generalized to nonuniform x; no weighted basis.
        return Stokes3DHybrid(
            FLUID, grid, analytic_only=True, quadrature_backend="gauss", **kwargs
        )
    raise ValueError(f"Unknown method {method}")


def fields(grid, geometry):
    x, y = grid.points.T
    bend = (x / geometry.length) ** 2
    return np.column_stack((np.ones(len(x)), bend, bend * (2 * y / geometry.width)))


def integrated_metrics(grid, geometry, velocity, pressure):
    weighted = grid.weights[:, None] * pressure
    resistance = velocity.conj().T @ weighted
    moments = (
        np.column_stack(
            (
                np.ones(len(weighted)),
                grid.points[:, 0] / geometry.length,
                2 * grid.points[:, 1] / geometry.width,
            )
        ).T
        @ weighted
    )
    bounds = grid.panel_bounds
    regions = [
        (0, 0.25, -0.5, 0.5),
        (0.75, 1, -0.5, 0.5),
        (0, 1, -0.5, -0.4),
        (0, 1, 0.4, 0.5),
        (0.25, 0.75, -0.4, 0.4),
    ]
    area = (bounds[:, 1] - bounds[:, 0]) * (bounds[:, 3] - bounds[:, 2])
    regional = []
    for xa, xb, ya, yb in regions:
        overlap_x = np.maximum(
            0,
            np.minimum(bounds[:, 1], xb * geometry.length)
            - np.maximum(bounds[:, 0], xa * geometry.length),
        )
        overlap_y = np.maximum(
            0,
            np.minimum(bounds[:, 3], yb * geometry.width)
            - np.maximum(bounds[:, 2], ya * geometry.width),
        )
        regional.append((overlap_x * overlap_y / area) @ weighted)
    return resistance, moments, np.asarray(regional)


def integration_worker(config, output):
    geometry = PlateGeometry(500e-6, config["width"], 5e-6)
    method, tolerance = config["method"], config["tolerance"]
    warm_grid = FluidGrid.midpoint(geometry, nx=2, ny=3)
    warm = make_model(method, warm_grid, tolerance, config.get("near_ratio", 0.5))
    warm.pressure_from_velocity(2 * np.pi * 1e3, np.ones(6))
    warm.clear_cache()
    del warm
    gc.collect()
    baseline = rss()
    started = perf_counter()
    grid = make_grid(geometry, config["grid"])
    model = make_model(method, grid, tolerance, config.get("near_ratio", 0.5))
    setup = perf_counter() - started
    velocity = fields(grid, geometry)
    arrays, rows = {}, []
    retain = config.get("measure", "timing") != "memory"
    for i, hz in enumerate(config["frequencies"]):
        model.clear_cache()
        started = perf_counter()
        matrix = model.assemble_matrix(2 * np.pi * hz)
        assembly = perf_counter() - started
        started = perf_counter()
        pressure = model.pressure_from_velocity(2 * np.pi * hz, velocity)
        pressure_time = perf_counter() - started
        resistance, moments, regional = integrated_metrics(
            grid, geometry, velocity, pressure
        )
        report = getattr(model, "hybrid_report", model.integration_report)
        if is_dataclass(report):
            report = asdict(report)
        row = {
            "frequency": hz,
            "assembly": assembly,
            "pressure_solve": pressure_time,
            "total": assembly + pressure_time,
            "matrix_bytes": matrix.nbytes,
            "residual": float(
                np.linalg.norm(matrix @ pressure - velocity) / np.linalg.norm(velocity)
            ),
            "report": report,
        }
        rows.append(row)
        if retain:
            arrays[f"matrix_{i}"] = (
                matrix.copy() if config.get("matrices", False) else np.empty(0)
            )
            arrays[f"pressure_{i}"] = pressure
            arrays[f"resistance_{i}"] = resistance
            arrays[f"moments_{i}"] = moments
            arrays[f"regions_{i}"] = regional
    measured = rss()
    result = {
        "config": config,
        "shape": [grid.nx, grid.ny],
        "setup": setup,
        "rows": rows,
        "rss_baseline": baseline,
        "rss_peak": measured["VmHWM"],
        "rss_growth": max(0, measured["VmHWM"] - baseline["VmRSS"]),
    }
    if retain:
        arrays.update(
            points=grid.points,
            weights=grid.weights,
            frequencies=np.asarray(config["frequencies"]),
        )
        np.savez_compressed(output.with_suffix(".npz"), **arrays)
    dump(output, result)


def prepare_structure(geometry, mesh):
    from mufsi import DistributedLoad, KirchhoffPlate, Material
    from mufsi.coupling.basis_evaluation import build_evaluation_matrix
    from mufsi.solvers.frequency_response import _csr

    plate = KirchhoffPlate(
        geometry, Material(169e9, 2330, 0.3), mesh_resolution=tuple(mesh)
    )
    K, M = _csr(plate.stiffness_matrix()), _csr(plate.mass_matrix())
    handle = plate.force_vector(DistributedLoad(lambda x: np.ones(x.shape[1])))
    try:
        force = handle.getArray(readonly=True).copy().astype(complex)
    finally:
        handle.destroy()
    free = np.setdiff1d(np.arange(K.shape[0]), plate.constrained_dofs)
    tip = build_evaluation_matrix(plate.function_space, [[geometry.length, 0]])[:, free]
    probes = build_evaluation_matrix(
        plate.function_space,
        np.column_stack(
            (
                np.repeat(np.linspace(0, geometry.length, 33), 5),
                np.tile(np.linspace(-geometry.width / 2, geometry.width / 2, 5), 33),
            )
        ),
    )[:, free]
    return plate, K[free][:, free], M[free][:, free], force[free], free, tip, probes


def spectrum_worker(config, output):
    import mufsi.solvers.frequency_response as response
    from mufsi.coupling.basis_evaluation import build_evaluation_matrix

    geometry = PlateGeometry(500e-6, config["width"], 5e-6)
    started = perf_counter()
    plate, K, M, F, free, tip, probes = prepare_structure(geometry, config["mesh"])
    grid = make_grid(geometry, config["grid"])
    E = build_evaluation_matrix(plate.function_space, grid.points)[:, free].tocsr()
    G = (E.T @ sparse.diags(grid.weights)).tocsr()
    model = make_model(
        config["method"], grid, config["tolerance"], config.get("near_ratio", 0.5)
    )
    setup = perf_counter() - started
    k_scale = max(abs(K.diagonal()))
    # Warm the actual runtime and structural/Schur operations before RSS/timing.
    omega = 2 * np.pi * config["frequencies"][0]
    B = model.assemble_matrix(omega)
    response._fluid_schur_solve(K - omega**2 * M, G, E, B, F, omega, k_scale)
    model.clear_cache()
    del B
    gc.collect()
    baseline = rss()
    timings, tips, displacements, pressures, forces, powers, errors = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    retain = config.get("measure", "timing") != "memory"
    original_solver = response.SciPyLUSolver
    phases = {}

    class MeasuredLU(SciPyLUSolver):
        def factorize(self, matrix):
            self.label = "structural" if sparse.issparse(matrix) else "schur"
            start = perf_counter()
            super().factorize(matrix)
            phases[self.label + "_factor"] = (
                phases.get(self.label + "_factor", 0) + perf_counter() - start
            )

        def solve(self, rhs):
            start = perf_counter()
            answer = super().solve(rhs)
            phases[self.label + "_rhs"] = (
                phases.get(self.label + "_rhs", 0) + perf_counter() - start
            )
            return answer

    response.SciPyLUSolver = MeasuredLU
    try:
        for index, hz in enumerate(config["frequencies"]):
            omega = 2 * np.pi * hz
            model.clear_cache()
            start = perf_counter()
            B = model.assemble_matrix(omega)
            assembly = perf_counter() - start
            D = K - omega**2 * M
            phases.clear()
            start = perf_counter()
            u, p = response._fluid_schur_solve(D, G, E, B, F, omega, k_scale)
            coupled = perf_counter() - start
            timings.append(
                dict(
                    frequency=hz,
                    assembly=assembly,
                    coupled=coupled,
                    total=assembly + coupled,
                    **phases,
                    schur_build_and_recovery=coupled - sum(phases.values()),
                )
            )
            velocity = 1j * omega * (E @ u)
            errors.append(
                [
                    float(np.linalg.norm(D @ u + G @ p - F) / np.linalg.norm(F)),
                    float(np.linalg.norm(B @ p - velocity) / np.linalg.norm(velocity)),
                ]
            )
            if retain:
                tips.append(complex((tip @ u)[0]))
                displacements.append(np.asarray(probes @ u))
                pressures.append(p)
                forces.append(grid.weights @ p)
                powers.append(float(np.real(np.vdot(velocity, grid.weights * p)) / 2))
            if (
                index == 0
                or (index + 1) % 20 == 0
                or index + 1 == len(config["frequencies"])
            ):
                print(
                    f"  {config['method']} {grid.nx}x{grid.ny}: {index + 1}/{len(config['frequencies'])}",
                    flush=True,
                )
    finally:
        response.SciPyLUSolver = original_solver
    measured = rss()
    result = {
        "config": config,
        "shape": [grid.nx, grid.ny],
        "structural_dofs": K.shape[0],
        "setup": setup,
        "timings": timings,
        "residual_max": np.max(errors, axis=0).tolist(),
        "rss_baseline": baseline,
        "rss_peak": measured["VmHWM"],
        "rss_growth": max(0, measured["VmHWM"] - baseline["VmRSS"]),
    }
    if retain:
        np.savez_compressed(
            output.with_suffix(".npz"),
            frequencies=config["frequencies"],
            tip=tips,
            displacement=displacements,
            pressure=pressures,
            force=forces,
            power=powers,
            points=grid.points,
            weights=grid.weights,
        )
    dump(output, result)


def fingerprint():
    files = [Path(__file__), ROOT / "src/mufsi/hydrodynamics/legacy/stokeslet_hybrid.py"]
    files += sorted((ROOT / "src/mufsi").rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def worker_job(output, label, config, resume):
    output.mkdir(parents=True, exist_ok=True)
    config = dict(config, source_hash=fingerprint())
    path = output / (label + ".json")
    if resume and path.exists() and json.loads(path.read_text())["config"] == config:
        return json.loads(path.read_text()), path.with_suffix(".npz")
    config_path = output / (label + ".config.json")
    dump(config_path, config)
    environment = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(config_path),
            "--worker-output",
            str(path),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    return json.loads(path.read_text()), path.with_suffix(".npz")


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(float).tiny))


def fixed_errors(candidate, reference, count):
    matrix, pressure, resistance = [], [], []
    for i in range(count):
        matrix.append(relative(candidate[f"matrix_{i}"], reference[f"matrix_{i}"]))
        pressure.append(
            max(
                relative(
                    candidate[f"pressure_{i}"][:, j], reference[f"pressure_{i}"][:, j]
                )
                for j in range(3)
            )
        )
        resistance.append(
            relative(candidate[f"resistance_{i}"], reference[f"resistance_{i}"])
        )
    return {
        "matrix": max(matrix),
        "pressure": max(pressure),
        "resistance": max(resistance),
    }


def integration_stage(args):
    output = args.output / "integration"
    rows, reference_checks = [], []
    grids = [
        {"family": "edge", "nx": 6, "ny": 12},
        {"family": "translation", "nx": 6, "ny": 12},
        {"family": "hierarchical", "x": [3, 3], "y": [5, 3]},
    ]
    for width in (50e-6, 250e-6):
        for spec in grids:
            base = {
                "kind": "integration",
                "width": width,
                "grid": spec,
                "frequencies": FREQUENCIES,
                "matrices": True,
                "tolerance": 1e-9,
                "method": "analytic",
            }
            label = f"{round(width * 1e6)}_{spec['family']}"
            print(f"Integration reference: {label}", flush=True)
            _, ref_path = worker_job(output, label + "_reference", base, args.resume)
            _, check_path = worker_job(
                output,
                label + "_independent",
                dict(base, method="gauss", tolerance=1e-8),
                args.resume,
            )
            with np.load(ref_path) as reference, np.load(check_path) as independent:
                reference_checks.append(
                    dict(
                        case=label,
                        **fixed_errors(independent, reference, len(FREQUENCIES)),
                    )
                )
                methods = ["quadpy", "analytic", "hybrid"]
                if spec["family"] == "hierarchical":
                    methods += ["multigrid", "hybrid_multigrid"]
                for tolerance in (1e-3, 1e-5, 1e-7):
                    for method in methods:
                        ratios = (
                            (0.0, 0.5, 1.0)
                            if method == "hybrid" and tolerance == 1e-5
                            else (0.5,)
                        )
                        for ratio in ratios:
                            config = dict(
                                base,
                                tolerance=tolerance,
                                method=method,
                                near_ratio=ratio,
                            )
                            name = f"{label}_{method}_{tolerance:g}_{ratio:g}"
                            print(name, flush=True)
                            repeats, error = [], None
                            for repeat in range(args.repeats):
                                data, path = worker_job(
                                    output, name + f"_r{repeat}", config, args.resume
                                )
                                repeats.append(sum(r["total"] for r in data["rows"]))
                                if repeat == 0:
                                    with np.load(path) as candidate:
                                        error = fixed_errors(
                                            candidate, reference, len(FREQUENCIES)
                                        )
                            memory, _ = worker_job(
                                output,
                                name + "_memory",
                                dict(config, measure="memory", matrices=False),
                                args.resume,
                            )
                            rows.append(
                                {
                                    "case": label,
                                    "width": width,
                                    "grid": spec,
                                    "shape": data["shape"],
                                    "method": method,
                                    "tolerance": tolerance,
                                    "near_ratio": ratio,
                                    "seconds_median": float(np.median(repeats)),
                                    "seconds_min": min(repeats),
                                    "seconds_max": max(repeats),
                                    "rss_peak": memory["rss_peak"],
                                    "rss_growth": memory["rss_growth"],
                                    "error": error,
                                    "reports": [r["report"] for r in data["rows"]],
                                }
                            )
    dump(
        args.output / "integration.json",
        {"rows": rows, "reference_checks": reference_checks},
    )


def grid_errors(candidate, reference, count):
    errors = {
        "resistance": 0.0,
        "moments": 0.0,
        "regions": 0.0,
        "dissipated_power": 0.0,
    }
    for i in range(count):
        R, ref = candidate[f"resistance_{i}"], reference[f"resistance_{i}"]
        # Use diagonal scales for all three velocity fields, including torsion.
        errors["resistance"] = max(
            errors["resistance"],
            float(np.max(abs(np.diag(R - ref)) / abs(np.diag(ref)))),
        )
        for key in ("moments", "regions"):
            errors[key] = max(
                errors[key], relative(candidate[f"{key}_{i}"], reference[f"{key}_{i}"])
            )
        errors["dissipated_power"] = max(
            errors["dissipated_power"],
            float(
                np.max(
                    abs(np.diag(R).real - np.diag(ref).real)
                    / np.maximum(abs(np.diag(ref).real), 1e-30)
                )
            ),
        )
    return errors


def grid_stage(args):
    output = args.output / "grid"
    rows = []
    frequencies = FREQUENCIES[1:]
    for width in (50e-6, 250e-6):
        label = str(round(width * 1e6))
        base = {
            "kind": "integration",
            "width": width,
            "frequencies": frequencies,
            "matrices": False,
            "tolerance": 1e-7,
            "method": "hybrid",
        }
        nx, ny = args.grid_reference
        reference_spec = {"family": "edge", "nx": nx, "ny": ny}
        print(f"Grid reference {label}: {nx}x{ny}", flush=True)
        _, reference_path = worker_job(
            output, label + "_reference", dict(base, grid=reference_spec), args.resume
        )
        cases = []
        for family in ("uniform", "native", "edge", "tip", "translation"):
            cases += [
                {"family": family, "nx": x, "ny": y}
                for x, y in ((6, 12), (12, 24), (24, 36), (32, 48))
            ]
        cases += [{"family": "native", "nx": nx, "ny": ny}]
        cases += [
            {"family": "hierarchical", "x": x, "y": y}
            for x, y in (
                ([3, 3], [5, 3]),
                ([5, 3], [9, 3]),
                ([7, 3, 3], [13, 3, 3]),
                ([11, 3, 3], [21, 3, 3]),
                ([21, 3, 3], [41, 3, 3]),
            )
        ]
        cases += [
            {"family": "lattice_edge", "nx": x, "ny": y}
            for x, y in ((6, 12), (12, 24), (24, 36), (32, 48))
        ]
        with np.load(reference_path) as reference:
            for index, spec in enumerate(cases):
                method = (
                    "hybrid_multigrid"
                    if spec["family"] in ("hierarchical", "lattice_edge")
                    else "hybrid"
                )
                config = dict(base, grid=spec, method=method)
                print(f"Grid {label}: {index + 1}/{len(cases)}, {spec}", flush=True)
                data, path = worker_job(
                    output, label + f"_{index}", config, args.resume
                )
                with np.load(path) as candidate:
                    error = grid_errors(candidate, reference, len(frequencies))
                rows.append(
                    {
                        "width": width,
                        "grid": spec,
                        "shape": data["shape"],
                        "method": method,
                        "error": error,
                        "seconds": sum(r["total"] for r in data["rows"]),
                        "matrix_bytes": data["rows"][0]["matrix_bytes"],
                        "path": str(path.relative_to(args.output)),
                    }
                )
    dump(
        args.output / "grid.json",
        {
            "reference_shape": args.grid_reference,
            "rows": rows,
            "reference_uncertainty": "Assess 32x48 -> reference and native -> reference; successive differences are not rigorous bounds.",
        },
    )


def lattice_stage(args):
    """Extend a completed grid study without repeating its existing cases."""
    if not (args.output / "grid.json").exists():
        grid_stage(args)
        return
    data = json.loads((args.output / "grid.json").read_text())
    data["rows"] = [r for r in data["rows"] if r["grid"]["family"] != "lattice_edge"]
    for width in (50e-6, 250e-6):
        label = str(round(width * 1e6))
        reference_path = args.output / "grid" / (label + "_reference.npz")
        with np.load(reference_path) as reference:
            for nx, ny in ((6, 12), (12, 24), (24, 36), (32, 48)):
                spec = {"family": "lattice_edge", "nx": nx, "ny": ny}
                config = {
                    "kind": "integration",
                    "width": width,
                    "frequencies": FREQUENCIES[1:],
                    "matrices": False,
                    "tolerance": 1e-7,
                    "method": "hybrid_multigrid",
                    "grid": spec,
                }
                print(f"Lattice cosine grid {label}: {nx}x{ny}", flush=True)
                result, path = worker_job(
                    args.output / "grid",
                    f"{label}_lattice_{nx}_{ny}",
                    config,
                    args.resume,
                )
                with np.load(path) as candidate:
                    error = grid_errors(candidate, reference, len(FREQUENCIES[1:]))
                data["rows"].append(
                    {
                        "width": width,
                        "grid": spec,
                        "shape": result["shape"],
                        "method": "hybrid_multigrid",
                        "error": error,
                        "seconds": sum(r["total"] for r in result["rows"]),
                        "matrix_bytes": result["rows"][0]["matrix_bytes"],
                        "path": str(path.relative_to(args.output)),
                    }
                )
    dump(args.output / "grid.json", data)


def peak_metrics(frequencies, response, window):
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq

    select = (frequencies >= window[0]) & (frequencies <= window[1])
    f, amplitude = frequencies[select], abs(response[select])
    if len(f) < 5:
        return None
    # PCHIP gives stable half-power crossings. A local quadratic in log amplitude
    # estimates the peak between samples; the maximum of PCHIP alone would lock
    # the peak frequency to a sample. Half-sampling checks expose this sensitivity.
    interpolant = PchipInterpolator(f, amplitude)
    index = int(amplitude.argmax())
    if index in (0, len(f) - 1):
        return None
    scale = max(f[index] - f[index - 1], f[index + 1] - f[index])
    x = (f[index - 1 : index + 2] - f[index]) / scale
    coefficients = np.polyfit(
        x, np.log(amplitude[index - 1 : index + 2] / amplitude[index]), 2
    )
    offset = (
        float(np.clip(-coefficients[1] / (2 * coefficients[0]), x[0], x[-1]))
        if coefficients[0] < 0
        else 0.0
    )
    center = float(f[index] + offset * scale)
    height = float(amplitude[index] * np.exp(np.polyval(coefficients, offset)))
    target = height / np.sqrt(2)
    below_left = np.flatnonzero(amplitude[:index] < target)
    below_right = np.flatnonzero(amplitude[index + 1 :] < target)
    if not len(below_left) or not len(below_right):
        return {"frequency": center, "amplitude": height, "linewidth": None, "q": None}
    left_i, right_i = below_left[-1], index + 1 + below_right[0]
    left = brentq(lambda x: float(interpolant(x)) - target, f[left_i], center)
    right = brentq(lambda x: float(interpolant(x)) - target, center, f[right_i])
    return {
        "frequency": center,
        "amplitude": height,
        "linewidth": right - left,
        "q": center / (right - left),
    }


def spectrum_errors(candidate, reference, windows):
    f = reference["frequencies"]
    # Probe displacements use the same physical points across FE meshes.
    error = {
        "complex_tip_scaled": float(
            np.max(abs(candidate["tip"] - reference["tip"]))
            / max(abs(reference["tip"]))
        ),
        "displacement_scaled": float(
            np.max(
                np.linalg.norm(
                    candidate["displacement"] - reference["displacement"], axis=1
                )
            )
            / max(np.linalg.norm(reference["displacement"], axis=1))
        ),
        "force_scaled": float(
            np.max(abs(candidate["force"] - reference["force"]))
            / max(abs(reference["force"]))
        ),
        "power_scaled": float(
            np.max(abs(candidate["power"] - reference["power"]))
            / max(abs(reference["power"]))
        ),
        "peaks": [],
    }
    for window in windows:
        a, b = (
            peak_metrics(f, candidate["tip"], window),
            peak_metrics(f, reference["tip"], window),
        )
        if a is None or b is None:
            error["peaks"].append({"window": window, "resolved": False})
            continue
        record = {
            "window": window,
            "resolved": a["linewidth"] is not None and b["linewidth"] is not None,
            "candidate": a,
            "reference": b,
            "amplitude_relative": abs(a["amplitude"] / b["amplitude"] - 1),
        }
        selected = (f >= window[0]) & (f <= window[1])
        record["complex_window_scaled"] = float(
            np.max(abs(candidate["tip"][selected] - reference["tip"][selected]))
            / max(abs(reference["tip"][selected]))
        )
        if record["resolved"]:
            record.update(
                linewidth_relative=abs(a["linewidth"] / b["linewidth"] - 1),
                q_relative=abs(a["q"] / b["q"] - 1),
                frequency_shift_in_linewidths=abs(a["frequency"] - b["frequency"])
                / b["linewidth"],
            )
        error["peaks"].append(record)
    return error


def spectrum_stage(args):
    output = args.output / "spectrum"
    base = {
        "kind": "spectrum",
        "width": 50e-6,
        "method": "hybrid",
        "tolerance": 1e-5,
        "mesh": args.mesh,
        "grid": {"family": "edge", "nx": 12, "ny": 24},
    }
    scan = np.geomspace(1e3, 400e3, 49)
    _, path = worker_job(
        output, "pilot", dict(base, frequencies=scan.tolist()), args.resume
    )
    with np.load(path) as pilot:
        peaks, _ = find_peaks(
            abs(pilot["tip"]), prominence=0.015 * max(abs(pilot["tip"]))
        )
        windows = []
        for index in peaks[:3]:
            center = scan[index]
            windows.append(
                [float(max(1e3, center / 1.65)), float(min(400e3, center * 1.65))]
            )
    if not windows:
        raise RuntimeError("No resolved wet resonance found in the pilot spectrum.")
    frequencies = np.unique(
        np.r_[
            np.geomspace(1e3, 400e3, 33),
            *(np.linspace(*window, 65) for window in windows),
        ]
    )
    base["frequencies"] = frequencies.tolist()
    # Final and preceding fluid references use the same structurally resolved mesh.
    nx, ny = args.spectrum_reference
    reference_spec = {"family": "edge", "nx": nx, "ny": ny}
    print(
        f"Spectral reference: {nx}x{ny}, {len(frequencies)} frequencies, windows={windows}",
        flush=True,
    )
    ref_data, ref_path = worker_job(
        output,
        "reference",
        dict(base, grid=reference_spec, tolerance=1e-7),
        args.resume,
    )
    rows, checks = [], {}
    cases = [
        ("fluid_previous", "hybrid", {"family": "edge", "nx": 24, "ny": 36}, 1e-7),
        ("native_fine", "hybrid", {"family": "native", "nx": nx, "ny": ny}, 1e-7),
        ("edge_small", "hybrid", {"family": "edge", "nx": 6, "ny": 12}, 1e-5),
        ("edge_medium", "hybrid", {"family": "edge", "nx": 12, "ny": 24}, 1e-5),
        ("edge_large", "hybrid", {"family": "edge", "nx": 24, "ny": 36}, 1e-5),
        ("quadpy_shared", "quadpy", {"family": "edge", "nx": 6, "ny": 12}, 1e-5),
        ("native_medium", "quadpy", {"family": "native", "nx": 12, "ny": 24}, 1e-3),
        (
            "analytic_edge_medium",
            "analytic",
            {"family": "edge", "nx": 12, "ny": 24},
            1e-5,
        ),
        (
            "analytic_translation",
            "analytic",
            {"family": "translation", "nx": 12, "ny": 24},
            1e-5,
        ),
        (
            "reuse_medium",
            "hybrid_multigrid",
            {"family": "hierarchical", "x": [7, 3, 3], "y": [13, 3, 3]},
            1e-5,
        ),
        (
            "reuse_large",
            "hybrid_multigrid",
            {"family": "hierarchical", "x": [11, 3, 3], "y": [21, 3, 3]},
            1e-5,
        ),
        (
            "reuse_fine",
            "hybrid_multigrid",
            {"family": "hierarchical", "x": [21, 3, 3], "y": [41, 3, 3]},
            1e-5,
        ),
        (
            "reuse_large_loose",
            "hybrid_multigrid",
            {"family": "hierarchical", "x": [11, 3, 3], "y": [21, 3, 3]},
            1e-3,
        ),
        (
            "lattice_medium",
            "hybrid_multigrid",
            {"family": "lattice_edge", "nx": 12, "ny": 24},
            1e-5,
        ),
        (
            "lattice_large",
            "hybrid_multigrid",
            {"family": "lattice_edge", "nx": 24, "ny": 36},
            1e-5,
        ),
    ]
    with np.load(ref_path) as reference:
        for label, method, spec, tolerance in cases:
            print(f"Spectrum {label}", flush=True)
            config = dict(base, grid=spec, method=method, tolerance=tolerance)
            repeats = []
            # Uncertainty checks are single-run; candidate measurements repeated.
            for repeat in range(
                1 if label in ("fluid_previous", "native_fine") else args.repeats
            ):
                data, path = worker_job(
                    output, label + f"_r{repeat}", config, args.resume
                )
                repeats.append(sum(r["total"] for r in data["timings"]))
                if repeat == 0:
                    with np.load(path) as candidate:
                        error = spectrum_errors(candidate, reference, windows)
            if label in ("fluid_previous", "native_fine"):
                checks[label] = error
                continue
            memory, _ = worker_job(
                output, label + "_memory", dict(config, measure="memory"), args.resume
            )
            peaks_pass = all(
                p.get("resolved", False)
                and p["complex_window_scaled"] <= 0.01
                and p["amplitude_relative"] <= 0.01
                and p["q_relative"] <= 0.01
                and p["linewidth_relative"] <= 0.01
                and p["frequency_shift_in_linewidths"] <= 0.05
                for p in error["peaks"]
            )
            rows.append(
                {
                    "label": label,
                    "method": method,
                    "grid": spec,
                    "shape": data["shape"],
                    "tolerance": tolerance,
                    "error": error,
                    "target_met": error["complex_tip_scaled"] <= 0.01
                    and error["displacement_scaled"] <= 0.01
                    and peaks_pass,
                    "seconds_median": float(np.median(repeats)),
                    "seconds_min": min(repeats),
                    "seconds_max": max(repeats),
                    "rss_peak": memory["rss_peak"],
                    "rss_growth": memory["rss_growth"],
                    "residual_max": data["residual_max"],
                    "phase_seconds": {
                        key: sum(r.get(key, 0) for r in data["timings"])
                        for key in (
                            "assembly",
                            "structural_factor",
                            "structural_rhs",
                            "schur_factor",
                            "schur_rhs",
                            "schur_build_and_recovery",
                        )
                    },
                    "path": str(path.relative_to(args.output)),
                }
            )
        # Compare the next FE mesh on the same fluid reference. This includes all
        # frequencies and common displacement probes, avoiding DOF-norm ambiguity.
        fine_mesh = [round(args.mesh[0] * 4 / 3), max(4, round(args.mesh[1] * 4 / 3))]
        print(f"Structural reference check: {fine_mesh}", flush=True)
        _, fine_path = worker_job(
            output,
            "mesh_check",
            dict(base, grid=reference_spec, tolerance=1e-7, mesh=fine_mesh),
            args.resume,
        )
        with np.load(fine_path) as fine:
            checks["structural_mesh"] = spectrum_errors(reference, fine, windows)
        checks["sampling"] = []
        for window in windows:
            dense = peak_metrics(frequencies, reference["tip"], window)
            select = np.arange(len(frequencies)) % 2 == 0
            coarse = peak_metrics(frequencies[select], reference["tip"][select], window)
            checks["sampling"].append(
                {"window": window, "dense": dense, "half_samples": coarse}
            )
    dump(
        args.output / "spectrum.json",
        {
            "mesh": args.mesh,
            "reference_shape": args.spectrum_reference,
            "reference": ref_data,
            "windows": windows,
            "rows": rows,
            "uncertainty_checks": checks,
            "target": {
                "scaled_complex_displacement": 0.01,
                "amplitude": 0.01,
                "q": 0.01,
                "linewidth": 0.01,
                "shift_in_linewidths": 0.05,
                "desired_reference_displacement": 0.001,
            },
        },
    )


def refine_spectrum_stage(args):
    """Follow up the initial spectrum with transverse refinement and reuse."""
    path = args.output / "spectrum.json"
    if not path.exists():
        spectrum_stage(args)
    data = json.loads(path.read_text())
    output = args.output / "spectrum"
    base = dict(data["reference"]["config"])
    base.pop("source_hash", None)
    windows = data["windows"]
    names = ["lattice_16x48", "lattice_24x48", "lattice_fine"]
    data["rows"] = [r for r in data["rows"] if r["label"] not in names]
    with np.load(output / "reference.npz") as reference:
        for label, shape in zip(names, ((16, 48), (24, 48), (32, 48)), strict=True):
            config = dict(
                base,
                method="hybrid_multigrid",
                tolerance=1e-5,
                grid={"family": "lattice_edge", "nx": shape[0], "ny": shape[1]},
            )
            print(f"Transverse refinement {label}", flush=True)
            times = []
            for repeat in range(args.repeats):
                result, array_path = worker_job(
                    output, label + f"_r{repeat}", config, args.resume
                )
                times.append(sum(r["total"] for r in result["timings"]))
                if repeat == 0:
                    with np.load(array_path) as candidate:
                        error = spectrum_errors(candidate, reference, windows)
            memory, _ = worker_job(
                output, label + "_memory", dict(config, measure="memory"), args.resume
            )
            peaks_pass = all(
                p.get("resolved", False)
                and p["complex_window_scaled"] <= 0.01
                and p["amplitude_relative"] <= 0.01
                and p["q_relative"] <= 0.01
                and p["linewidth_relative"] <= 0.01
                and p["frequency_shift_in_linewidths"] <= 0.05
                for p in error["peaks"]
            )
            data["rows"].append(
                {
                    "label": label,
                    "method": "hybrid_multigrid",
                    "grid": config["grid"],
                    "shape": result["shape"],
                    "tolerance": config["tolerance"],
                    "error": error,
                    "target_met": error["complex_tip_scaled"] <= 0.01
                    and error["displacement_scaled"] <= 0.01
                    and peaks_pass,
                    "seconds_median": float(np.median(times)),
                    "seconds_min": min(times),
                    "seconds_max": max(times),
                    "rss_peak": memory["rss_peak"],
                    "rss_growth": memory["rss_growth"],
                    "residual_max": result["residual_max"],
                    "phase_seconds": {
                        key: sum(r.get(key, 0) for r in result["timings"])
                        for key in (
                            "assembly",
                            "structural_factor",
                            "structural_rhs",
                            "schur_factor",
                            "schur_rhs",
                            "schur_build_and_recovery",
                        )
                    },
                    "path": str(array_path.relative_to(args.output)),
                }
            )
        print("Finer fluid reference check: lattice 40x60", flush=True)
        config = dict(
            base,
            method="hybrid_multigrid",
            tolerance=1e-7,
            grid={"family": "lattice_edge", "nx": 40, "ny": 60},
        )
        fine_data, fine_path = worker_job(
            output, "fluid_finer_check", config, args.resume
        )
        with np.load(fine_path) as fine:
            data["uncertainty_checks"]["fluid_32_to_40"] = spectrum_errors(
                reference, fine, windows
            )
            for row in data["rows"]:
                with np.load(args.output / row["path"]) as candidate:
                    row["error_vs_finer_reference"] = spectrum_errors(
                        candidate, fine, windows
                    )
        data["finer_reference_check"] = {
            "metadata": fine_data,
            "path": str(fine_path.relative_to(args.output)),
        }
    dump(path, data)


def plot_results(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (args.output / "integration.json").exists():
        data = json.loads((args.output / "integration.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for method in ("quadpy", "analytic", "hybrid", "multigrid", "hybrid_multigrid"):
            rows = [
                r
                for r in data["rows"]
                if r["method"] == method and r["near_ratio"] == 0.5
            ]
            axes[0].loglog(
                [max(r["error"]["resistance"], 1e-12) for r in rows],
                [r["seconds_median"] for r in rows],
                "o",
                label=method,
            )
            axes[1].scatter(
                [r["rss_growth"] / 2**20 for r in rows],
                [r["rss_peak"] / 2**20 for r in rows],
                label=method,
            )
        axes[0].set(
            xlabel="Achieved generalized-resistance error",
            ylabel="Five frequencies, seconds",
            title="Same-panel integration accuracy and time",
        )
        axes[1].set(
            xlabel="Peak RSS above warmed baseline [MiB]",
            ylabel="Absolute peak RSS [MiB]",
            title="Memory measured in separate workers",
        )
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
        fig.savefig(args.output / "integration.png", dpi=180)
        plt.close(fig)
    if (args.output / "grid.json").exists():
        data = json.loads((args.output / "grid.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for ax, width in zip(axes, (50e-6, 250e-6), strict=True):
            for family in (
                "uniform",
                "native",
                "edge",
                "tip",
                "translation",
                "hierarchical",
                "lattice_edge",
            ):
                rows = [
                    r
                    for r in data["rows"]
                    if r["width"] == width and r["grid"]["family"] == family
                ]
                ax.loglog(
                    [np.prod(r["shape"]) for r in rows],
                    [max(r["error"]["resistance"], 1e-8) for r in rows],
                    "o-",
                    label=family,
                )
            ax.axhline(0.01, color="black", ls=":")
            ax.set(
                xlabel="Pressure unknowns",
                ylabel="Worst resistance diagonal error",
                title=f"500 x {round(width * 1e6)} µm; reference {data['reference_shape']}",
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
        fig.savefig(args.output / "grid.png", dpi=180)
        plt.close(fig)
    if (args.output / "spectrum.json").exists():
        data = json.loads((args.output / "spectrum.json").read_text())
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
        with np.load(args.output / "spectrum/reference.npz") as ref:
            axes[0].loglog(
                ref["frequencies"] / 1e3, abs(ref["tip"]) * 1e9, "k-", label="reference"
            )
            for row in data["rows"]:
                with np.load(args.output / row["path"]) as candidate:
                    axes[0].loglog(
                        candidate["frequencies"] / 1e3,
                        abs(candidate["tip"]) * 1e9,
                        label=row["label"],
                    )
                axes[1].loglog(
                    row["error"]["complex_tip_scaled"], row["seconds_median"], "o"
                )
                axes[1].annotate(
                    row["label"],
                    (row["error"]["complex_tip_scaled"], row["seconds_median"]),
                    fontsize=7,
                )
                axes[2].scatter(row["rss_peak"] / 2**20, row["seconds_median"])
                axes[2].annotate(
                    row["label"],
                    (row["rss_peak"] / 2**20, row["seconds_median"]),
                    fontsize=7,
                )
        axes[0].set(
            xlabel="Frequency [kHz]",
            ylabel="Tip displacement [nm/Pa]",
            title="Full FE displacement spectra",
        )
        axes[0].legend(fontsize=7)
        axes[1].axvline(0.01, color="k", ls=":")
        axes[1].set(
            xlabel="Complex tip error, global amplitude scale",
            ylabel="Complete sweep [s]",
            title="Accuracy / runtime compromise",
        )
        axes[2].set(
            xlabel="Absolute peak RSS [MiB]",
            ylabel="Complete sweep [s]",
            title="Memory / runtime compromise",
        )
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.savefig(args.output / "spectrum.png", dpi=180)
        plt.close(fig)


def plot_compromise(args):
    """Readable selection using the finer reference, alongside the complete plots."""
    if not (args.output / "spectrum.json").exists():
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((args.output / "spectrum.json").read_text())
    names = {
        "quadpy_shared": "Quadpy 6×12 (shared)",
        "native_medium": "Quadpy 12×24 (native)",
        "analytic_translation": "Radial 12×24 (uniform x)",
        "edge_large": "Direct hybrid 24×36",
        "reuse_fine": "Hierarchical reuse 29×49",
        "lattice_large": "Cosine reuse 24×36",
        "lattice_fine": "Cosine reuse 32×48",
    }
    rows = [r for r in data["rows"] if r["label"] in names]
    ref_path = data.get("finer_reference_check", {}).get(
        "path", "spectrum/reference.npz"
    )
    reference_shape = (
        data.get("finer_reference_check", {})
        .get("metadata", {})
        .get("shape", data["reference_shape"])
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), layout="constrained")
    error_fig, error_ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
    with np.load(args.output / ref_path) as reference:
        frequencies, tip = reference["frequencies"], reference["tip"]
        axes[0].loglog(
            frequencies / 1e3,
            abs(tip) * 1e9,
            "k-",
            linewidth=2,
            label=f"Reference {reference_shape[0]}×{reference_shape[1]}",
        )
        for row in rows:
            label = names[row["label"]]
            with np.load(args.output / row["path"]) as candidate:
                (line,) = axes[0].loglog(
                    frequencies / 1e3, abs(candidate["tip"]) * 1e9, label=label
                )
                difference = abs(candidate["tip"] - tip) / max(abs(tip))
                error_ax.loglog(
                    frequencies / 1e3,
                    np.maximum(difference, 1e-10),
                    color=line.get_color(),
                    label=label,
                )
            error = row.get("error_vs_finer_reference", row["error"])[
                "complex_tip_scaled"
            ]
            seconds, memory = row["seconds_median"], row["rss_peak"] / 2**20
            for ax, x, shift in (
                (axes[1], error, (-6, 5) if error > 0.08 else (6, 5)),
                (axes[2], memory, (-6, 5) if memory > 360 else (6, 5)),
            ):
                ax.scatter(x, seconds, color=line.get_color())
                ax.annotate(
                    label,
                    (x, seconds),
                    xytext=shift,
                    textcoords="offset points",
                    ha="right" if shift[0] < 0 else "left",
                    fontsize=7,
                )
    axes[0].set(
        xlabel="Frequency [kHz]",
        ylabel="Tip displacement [nm/Pa]",
        title="Full FE response",
    )
    axes[0].legend(fontsize=7)
    axes[1].set(
        xscale="log",
        yscale="log",
        xlim=(0.0035, 0.22),
        ylim=(6, 310),
        xlabel="Complex tip difference from finer grid",
        ylabel="98-frequency sweep [s]",
        title="Accuracy and runtime",
    )
    axes[1].axvline(0.01, color="k", ls=":", label="1% comparison target")
    axes[2].set(
        xlim=(185, 455),
        ylim=(0, 250),
        xlabel="Peak process RSS [MiB]",
        ylabel="98-frequency sweep [s]",
        title="Memory and runtime",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
    error_ax.axhline(0.01, color="k", ls=":", label="1% of reference peak")
    error_ax.set(
        xlabel="Frequency [kHz]",
        ylabel="Complex tip difference / reference peak",
        title=f"Differences from {reference_shape[0]}×{reference_shape[1]} refinement",
    )
    error_ax.legend(fontsize=8)
    error_ax.grid(True, which="both", alpha=0.25)
    fig.savefig(args.output / "compromise.png", dpi=180)
    error_fig.savefig(args.output / "displacement_error.png", dpi=180)
    plt.close(fig)
    plt.close(error_fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "integration", "grid", "lattice", "spectrum", "refine", "plot"),
        default="all",
    )
    parser.add_argument("--output", type=Path, default=Path("results/hybrid"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--grid-reference", type=int, nargs=2, default=[48, 72])
    parser.add_argument("--spectrum-reference", type=int, nargs=2, default=[32, 48])
    parser.add_argument("--mesh", type=int, nargs=2, default=[48, 6])
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker:
        config = json.loads(args.worker.read_text())
        (spectrum_worker if config["kind"] == "spectrum" else integration_worker)(
            config, args.worker_output
        )
        return
    if (
        args.repeats < 1
        or min(*args.grid_reference, *args.spectrum_reference, *args.mesh) < 2
    ):
        parser.error("Require positive repeats and grid/mesh counts >= 2.")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if not (ROOT / ".benchmark_snapshot").exists() and args.stage != "plot":
        # Freeze all numerical code before a long run, including worker imports.
        snapshot = args.output / ("source_" + fingerprint()[:12])
        files = [Path(__file__), ROOT / "src/mufsi/hydrodynamics/legacy/stokeslet_hybrid.py"]
        files += sorted((ROOT / "src/mufsi").rglob("*.py"))
        for source in files:
            destination = snapshot / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (snapshot / ".benchmark_snapshot").write_text(fingerprint() + "\n")
        environment = dict(
            os.environ,
            PYTHONPATH=str(snapshot / "src") + os.pathsep + str(snapshot),
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        subprocess.run(
            [
                sys.executable,
                str(snapshot / Path(__file__).resolve().relative_to(ROOT)),
                *sys.argv[1:],
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        return
    dump(
        args.output
        / ("environment_plot.json" if args.stage == "plot" else "environment.json"),
        {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threads": 1,
            "repeats": args.repeats,
            "source_hash": fingerprint(),
            "command": sys.argv,
        },
    )
    for stage, function in (
        ("integration", integration_stage),
        ("grid", grid_stage),
        ("lattice", lattice_stage),
        ("spectrum", spectrum_stage),
        ("refine", refine_spectrum_stage),
    ):
        if args.stage in (stage, "all"):
            function(args)
    plot_results(args)
    plot_compromise(args)
    print(f"Saved benchmark: {args.output}", flush=True)


if __name__ == "__main__":
    main()
