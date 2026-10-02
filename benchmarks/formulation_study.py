"""Reproducible full-FE geometry, spectrum, Q, accuracy and cost comparison.

Run in the matched serial DOLFINx environment, with one BLAS/OpenMP thread.
Stages save checkpoints so completed expensive sweeps can be resumed.
The plot stage draws only computed samples; no spline creates extra results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import resource
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import find_peaks

from mufsi import (
    BeamFrequencyResponseSolver,
    BeamGeometry,
    CoupledProblem,
    EigenSolver,
    EulerBernoulliBeam,
    Fluid,
    FluidGrid,
    FrequencyResponseSolver,
    KirchhoffPlate,
    Material,
    PlateGeometry,
    SaderMethod,
    SectionForce2D,
    Stokes2D,
    Stokes3D,
    WeightedCouplingOperator,
    corner_displacement,
    corner_loads,
    energy_from_response,
    fit_sho,
    resonance_frequency,
)
from mufsi.coupling.weighted import surface_evaluation

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "slender": {
        "length": 500e-6,
        "width": 50e-6,
        "thickness": 5e-6,
        "fmax": 400e3,
        "plate_mesh": (32, 6),
    },
    "wide": {
        "length": 500e-6,
        "width": 250e-6,
        "thickness": 5e-6,
        "fmax": 150e3,
        "plate_mesh": (32, 16),
    },
    "beam": {
        "length": 800e-6,
        "width": 50e-6,
        "thickness": 5e-6,
        "fmax": 200e3,
        "plate_mesh": (40, 6),
    },
}
MODELS = ("EB_3D", "KL_3D", "EB_2D", "KL_2D", "EB_Sader")
LABELS = {
    "EB_3D": "EB + weighted 3D",
    "KL_3D": "KL + weighted 3D",
    "EB_2D": "EB + 2D",
    "KL_2D": "KL + 2D",
    "EB_Sader": "EB + Sader",
    "KL_3D_odd": "KL + weighted 3D / odd",
    "KL_2D_odd": "KL + 2D / odd",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def source_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Study:
    def __init__(
        self,
        case,
        *,
        x_degree=16,
        y_degree=8,
        mesh_scale=1,
        nx=None,
        ny_half=None,
        grid_scale=1,
        section_nx=None,
        section_ny=None,
        tolerance=5e-6,
    ):
        self.case = case
        cfg = CASES[case]
        self.geometry = BeamGeometry(cfg["length"], cfg["width"], cfg["thickness"])
        g = self.geometry
        self.plate_geometry = PlateGeometry(g.length, g.width, g.thickness)
        self.material, self.fluid = Material(169e9, 2330, 0.3), Fluid(997, 890e-6)
        self.beam = EulerBernoulliBeam(
            g, self.material, mesh_resolution=48 * mesh_scale, element_degree=3
        )
        self.plate = KirchhoffPlate(
            self.plate_geometry,
            self.material,
            mesh_resolution=tuple(n * mesh_scale for n in cfg["plate_mesh"]),
        )
        nx = x_degree + 8 if nx is None else nx
        ny_half = y_degree // 2 + 2 if ny_half is None else ny_half
        eb = Stokes3D(
            self.fluid,
            g,
            formulation="EB",
            x_degree=x_degree,
            y_degree=y_degree,
            nx=nx,
            ny=ny_half,
            tolerance=tolerance,
        )
        kl = Stokes3D(
            self.fluid,
            self.plate_geometry,
            formulation="KL",
            x_degree=x_degree,
            y_degree=y_degree,
            nx=nx,
            ny=2 * ny_half,
            tolerance=tolerance,
        )
        section_nx = 64 * grid_scale + 1 if section_nx is None else section_nx
        section_ny = 64 * grid_scale if section_ny is None else section_ny
        grid = FluidGrid.chebyshev_gauss(
            self.plate_geometry, nx=section_nx, ny=section_ny
        )
        self.solvers = {
            "EB_3D": FrequencyResponseSolver(CoupledProblem(self.beam, eb)),
            "KL_3D": FrequencyResponseSolver(CoupledProblem(self.plate, kl)),
            "EB_2D": BeamFrequencyResponseSolver(
                self.beam, SectionForce2D(g, self.fluid, ny=section_ny)
            ),
            "KL_2D": FrequencyResponseSolver(
                CoupledProblem(self.plate, Stokes2D(self.fluid, grid))
            ),
            "EB_Sader": BeamFrequencyResponseSolver(
                self.beam, SectionForce2D(g, self.fluid, method="sader")
            ),
        }
        self.loads = {
            "EB": corner_loads(self.beam),
            "KL": corner_loads(self.plate),
            "odd": corner_loads(self.plate, symmetry="antisymmetric"),
        }
        self.config = {
            "case": case,
            "geometry": asdict(g),
            "x_degree": x_degree,
            "y_degree": y_degree,
            "nx": nx,
            "ny_half": ny_half,
            "mesh_scale": mesh_scale,
            "grid_scale": grid_scale,
            "tolerance": tolerance,
            "harmonic_convention": "exp(+i omega t)",
            "amplitude_N_per_corner": 1e-9,
            "tip_inset_fraction": 0.001,
        }
        self._shared = None
        if section_nx != 64 * grid_scale + 1 or section_ny != 64 * grid_scale:
            self.config["section_grid_shape"] = [section_nx, section_ny]

    def structure(self, model):
        return self.beam if model.startswith("EB") else self.plate

    def prepare(self, model):
        model = model.removesuffix("_odd")
        solver = self.solvers[model]
        if model.endswith("3D") and solver.problem.coupling is None:
            solver.problem.coupling = WeightedCouplingOperator.from_structure(
                solver.problem.structure,
                solver.problem.hydrodynamics,
                tolerance=1e-4,
                orders=(10, 16, 24, 36, 52),
            )

    def share_assembly(self):
        """Reuse identical kernel integrals across EB/KL, not coupled solutions.

        Matching independent positive-y rows and the even KL basis columns are
        precisely the EB mobility. Independent validation is in the pilot.
        Timings from shared sweeps are not used to rank model runtime.
        """
        eb, kl = (self.solvers[n].problem.hydrodynamics for n in ("EB_3D", "KL_3D"))
        rows = np.flatnonzero(kl.collocation_points[:, 1] > 0)
        columns = (
            np.arange(kl.coefficient_count)
            .reshape(kl.basis.M + 1, kl.basis.K + 1)[:, ::2]
            .ravel()
        )
        np.testing.assert_allclose(
            eb.collocation_points, kl.collocation_points[rows], rtol=0, atol=1e-18
        )
        direct = kl.mobility.assemble
        cache = {}

        def master(omega):
            if cache.get("omega") != omega:
                matrix, report = direct(omega, kl.collocation_points)
                cache.update(omega=omega, matrix=matrix, report=report)
            return cache["matrix"], cache["report"]

        def eb_assembly(omega, points=None, *, progress=None):
            if points is not None and not np.array_equal(points, eb.collocation_points):
                raise ValueError("Shared sweep only accepts the standard EB points.")
            matrix, report = master(omega)
            report = dict(report, shape=[len(rows), len(columns)], shared_from="KL")
            return matrix[np.ix_(rows, columns)], report

        def kl_assembly(omega, points=None, *, progress=None):
            if points is not None and not np.array_equal(points, kl.collocation_points):
                raise ValueError("Shared sweep only accepts the standard KL points.")
            return master(omega)

        eb.mobility.assemble, kl.mobility.assemble = eb_assembly, kl_assembly
        self._shared = cache

    def solve(self, model, frequencies):
        odd = model.endswith("_odd")
        base = model.removesuffix("_odd")
        load = self.loads["odd" if odd else base[:2]]
        response = self.solvers[base].solve(frequencies, load)
        observable = corner_displacement(
            self.structure(base),
            response.displacement,
            symmetry="antisymmetric" if odd else "symmetric",
        )
        return response, observable

    def energy(self, model, response):
        odd, base = model.endswith("_odd"), model.removesuffix("_odd")
        solver = self.solvers[base]
        coupling = (
            solver.problem.coupling
            if isinstance(solver, FrequencyResponseSolver)
            else None
        )
        return energy_from_response(
            self.structure(base),
            response,
            self.loads["odd" if odd else base[:2]],
            coupling=coupling,
        )

    def seeds(self):
        beta = np.array([1.875104068711961, 4.694091132974175, 7.854757438237612])
        dry = (
            beta**2
            / (2 * np.pi * self.geometry.length**2)
            * np.sqrt(self.beam.flexural_rigidity / self.beam.line_density)
        )
        wet, q = SaderMethod(self.geometry, self.material, self.fluid).resonance_and_q(
            dry
        )
        return dry, wet, q


def configuration(args, case, **overrides):
    config = {
        "x_degree": args.x_degree,
        "y_degree": args.y_degree,
        "tolerance": args.tolerance,
    }
    config.update(overrides)
    for name in ("section_nx", "section_ny"):
        value = getattr(args, name, None)
        if value is not None:
            config.setdefault(name, value)
    return Study(case, **config)


def pilot(args, case):
    out = args.output / case
    study = configuration(args, case)
    dry, wet, q = study.seeds()
    trials = []
    for hz in (float(wet[0]), float(wet[1]), float(wet[2])):
        for name in MODELS:
            study.prepare(name)
            start = perf_counter()
            response, tip = study.solve(name, [hz])
            seconds = perf_counter() - start
            energy = study.energy(name, response)
            rows = {
                "model": name,
                "frequency_Hz": hz,
                "seconds": seconds,
                "amplitude_m": float(abs(tip[0])),
                "Q_energy": float(energy.q_factor[0]),
                "equilibrium_error": float(response.relative_errors.max()),
                "work_balance_error": float(energy.work_balance_errors.max()),
            }
            if name.endswith("3D"):
                hydro = study.solvers[name].problem.hydrodynamics
                rows.update(
                    collocation_error=float(response.fluid_errors.max()),
                    force_projection_error=response.force_projection_error,
                    max_quadrature_order=hydro.integration_report["max_order"],
                    scaled_condition=hydro.solve_report["scaled_condition"],
                )
            trials.append(rows)
            print(case, name, f"{hz / 1e3:.2f} kHz", f"{seconds:.2f} s", flush=True)
    # Validate row/column reuse against an independently assembled EB matrix.
    eb = study.solvers["EB_3D"].problem.hydrodynamics
    direct = eb.assemble_matrix(2 * np.pi * wet[0]).copy()
    eb.clear_cache()
    study.share_assembly()
    reused = eb.assemble_matrix(2 * np.pi * wet[0])
    reuse_error = np.linalg.norm(reused - direct) / np.linalg.norm(direct)
    if reuse_error > 2 * args.tolerance:
        raise RuntimeError(f"Shared EB/KL mobility validation failed: {reuse_error:g}.")
    # Dry modes are diagnostics and seed references; response remains full FE.
    eig = EigenSolver(study.plate).solve(12)
    points = np.column_stack(
        (
            np.linspace(0.05, 0.95, 21) * study.geometry.length,
            np.full(21, 0.31 * study.geometry.width),
        )
    )
    positive = surface_evaluation(study.plate, points) @ eig.modes
    reflected = points.copy()
    reflected[:, 1] *= -1
    negative = surface_evaluation(study.plate, reflected) @ eig.modes
    parity = np.linalg.norm(positive - negative, axis=0) / np.maximum(
        np.linalg.norm(positive + negative, axis=0), 1e-300
    )
    report = {
        "configuration": study.config,
        "source_hash": source_hash(),
        "dry_EB_Hz": dry.tolist(),
        "reference_wet_Hz": wet.tolist(),
        "reference_Q": q.tolist(),
        "dry_KL_Hz": eig.frequencies.tolist(),
        "KL_parity": [
            "even" if p < 1e-3 else "odd" if p > 1e3 else "mixed" for p in parity
        ],
        "shared_mobility_relative_difference": float(reuse_error),
        "trials": trials,
    }
    write_json(out / "pilot.json", report)
    return report


def frequency_grid(args, case, pilot_report):
    frequencies = list(np.geomspace(100, CASES[case]["fmax"], args.broad_samples))
    windows = []
    for center, q in zip(
        pilot_report["reference_wet_Hz"], pilot_report["reference_Q"], strict=True
    ):
        half = min(0.55 * center, 1.4 * center / q)
        low, high = max(100, center - half), min(CASES[case]["fmax"], center + half)
        frequencies.extend(np.linspace(low, high, args.peak_samples))
        windows.append([low, high])
    return np.unique(frequencies), windows


def checkpoint(path, config, frequencies, completed, arrays, diagnostics):
    np.savez_compressed(
        path / "spectrum.npz", frequencies_Hz=frequencies, completed=completed, **arrays
    )
    write_json(
        path / "sweep.json",
        {
            "configuration": config,
            "source_hash": source_hash(),
            "diagnostics": diagnostics,
        },
    )


def sweep(args, case):
    out = args.output / case
    out.mkdir(parents=True, exist_ok=True)
    pilot_path = out / "pilot.json"
    report = (
        json.loads(pilot_path.read_text()) if pilot_path.exists() else pilot(args, case)
    )
    study = configuration(args, case)
    if report["source_hash"] != source_hash() or report["configuration"] != json.loads(
        json.dumps(study.config)
    ):
        raise ValueError("Pilot configuration/source differs; rerun the pilot.")
    f, windows = frequency_grid(args, case, report)
    models = list(MODELS) + (["KL_3D_odd", "KL_2D_odd"] if case == "wide" else [])
    for name in models:
        study.prepare(name)
    study.share_assembly()
    arrays = {f"{name}_displacement_m": np.zeros(len(f), complex) for name in models}
    completed = np.zeros(len(f), bool)
    diagnostics = []
    path = out / "spectrum.npz"
    if args.resume and path.exists():
        meta = json.loads((out / "sweep.json").read_text())
        with np.load(path) as saved:
            if (
                meta["source_hash"] != source_hash()
                or meta["configuration"] != json.loads(json.dumps(study.config))
                or not np.array_equal(saved["frequencies_Hz"], f)
            ):
                raise ValueError(
                    "Sweep checkpoint does not match current source/configuration."
                )
            completed = saved["completed"].copy()
            arrays = {key: saved[key].copy() for key in arrays}
        diagnostics = meta["diagnostics"]
    start = perf_counter()
    for i, hz in enumerate(f):
        if completed[i]:
            continue
        row = {"frequency_Hz": float(hz), "models": {}}
        for name in models:
            before = perf_counter()
            response, observable = study.solve(name, [hz])
            arrays[f"{name}_displacement_m"][i] = observable[0]
            energy = study.energy(name, response)
            diag = {
                "seconds": perf_counter() - before,
                "equilibrium_error": float(response.relative_errors.max()),
                "work_balance_error": float(energy.work_balance_errors.max()),
                "Q_energy": float(energy.q_factor[0]),
            }
            if name.removesuffix("_odd").endswith("3D"):
                diag.update(
                    collocation_error=float(response.fluid_errors.max()),
                    force_projection_error=response.force_projection_error,
                )
            row["models"][name] = diag
        diagnostics.append(row)
        completed[i] = True
        if i % 10 == 0 or i == len(f) - 1:
            checkpoint(out, study.config, f, completed, arrays, diagnostics)
            print(
                f"{case}: {completed.sum()}/{len(f)} frequencies; elapsed {perf_counter() - start:.1f}s",
                flush=True,
            )
    write_json(
        out / "windows.json",
        {
            "windows_Hz": windows,
            "models": models,
            "broad_samples": args.broad_samples,
            "peak_samples": args.peak_samples,
            "shared_assembly": "Identical EB even-basis integrals extracted from KL; coupled systems solved independently.",
        },
    )
    finish_q(args, case, study, f, windows, arrays, models)


def choose_fit(f, values, window):
    """Use the local peak, then a narrower resolved fit window to avoid other modes."""
    mask = (f >= window[0]) & (f <= window[1])
    local_f, local = f[mask], values[mask]
    if len(local_f) < 21:
        raise ValueError("Not enough samples in the resonance window.")
    peak = int(np.argmax(abs(local)))
    if peak in (0, len(local) - 1):
        raise ValueError("Peak is at the search-window boundary.")
    center = local_f[peak]
    above = np.flatnonzero(abs(local) >= abs(local[peak]) / np.sqrt(2))
    linewidth = (
        (local_f[above[-1]] - local_f[above[0]]) if len(above) > 1 else center / 10
    )
    half = max(0.8 * linewidth, 5 * np.median(np.diff(local_f)))
    selected = mask & (abs(f - center) <= half)
    if selected.sum() < 21:
        selected = mask
    fit = fit_sho(f[selected], values[selected], fit_background=True)
    return fit, selected


def finish_q(args, case, study, f, windows, arrays, models):
    results = []
    out = args.output / case
    fit_arrays = {}
    for name in MODELS:
        values = arrays[f"{name}_displacement_m"]
        for mode, window in enumerate(windows, 1):
            row = {"model": name, "resonance": mode, "search_window_Hz": window}
            try:
                fit, selected = choose_fit(f, values, window)
                response, _ = study.solve(name, [fit.resonance_frequency])
                energy = study.energy(name, response)
                row.update(
                    fitted_f0_Hz=fit.resonance_frequency,
                    Q_SHO=fit.q_factor,
                    Q_energy=float(energy.q_factor[0]),
                    fit_error=fit.relative_error,
                    peak_Hz=resonance_frequency(f[selected], values[selected]),
                    work_balance_error=float(energy.work_balance_errors.max()),
                    status="ok",
                )
                fit_arrays[f"{name}_{mode}_frequency_Hz"] = f[selected]
                fit_arrays[f"{name}_{mode}_fit_m"] = fit.fitted_amplitudes
            except (ValueError, RuntimeError) as error:
                row.update(status="unresolved", reason=str(error))
            results.append(row)
    write_json(
        out / "qfactor.json",
        {
            "results": results,
            "energy_definition": "Maximum structural bending plus kinetic energy divided by model-specific work; no stored-fluid energy term.",
        },
    )
    np.savez_compressed(out / "sho_fits.npz", **fit_arrays)
    fields = [
        "model",
        "resonance",
        "status",
        "fitted_f0_Hz",
        "peak_Hz",
        "Q_SHO",
        "Q_energy",
        "fit_error",
        "work_balance_error",
        "reason",
    ]
    with (out / "qfactor.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def independent_velocity_error(study, model, response):
    """Check no slip at 35 points excluded from the collocation grid."""
    from mufsi.hydrodynamics.weighted_pressure import PlateMobility, WeightedMobility

    base = model.removesuffix("_odd")
    hydro = study.solvers[base].problem.hydrodynamics
    g = study.geometry
    points = np.array(
        [
            (x * g.length, y * g.width / 2)
            for x in (0.013, 0.11, 0.29, 0.51, 0.73, 0.91, 0.987)
            for y in (-0.93, -0.47, 0.07, 0.53, 0.91)
        ]
    )
    cls = WeightedMobility if base.startswith("EB") else PlateMobility
    mobility = cls(hydro.basis, hydro.fluid, hydro.mobility.quadrature)
    H, _ = mobility.assemble(2 * np.pi * response.frequencies[0], points)
    velocity = (
        2j
        * np.pi
        * response.frequencies[0]
        * (surface_evaluation(study.structure(base), points) @ response.displacement[0])
    )
    predicted = H @ response.pressure_coefficients[0]
    return float(np.linalg.norm(predicted - velocity) / np.linalg.norm(velocity))


def validation(args, case):
    """One-variable-at-a-time refinement at three wet resonance estimates.

    These probes check complex displacement and energy Q, not extrapolated
    continuum limits or fitted-Q convergence of an entire refined sweep.
    """
    out = args.output / case
    report = json.loads((out / "pilot.json").read_text())
    base = configuration(args, case)
    variants = {
        "baseline": base,
        "pressure_degree": configuration(args, case, x_degree=20, y_degree=12),
        "collocation": configuration(args, case, nx=32, ny_half=10),
        "quadrature": configuration(args, case, tolerance=1e-6),
        "structural_mesh": configuration(args, case, mesh_scale=2),
        "section_grid": configuration(args, case, grid_scale=2),
    }
    models = list(MODELS)
    records = []
    for key, study in variants.items():
        for name in models:
            study.prepare(name)
        study.share_assembly()
        if key in ("structural_mesh", "section_grid"):
            # Reuse exactly the same mobility action; force projection and FE
            # evaluations still come from the independently refined structure.
            for name in ("EB_3D", "KL_3D"):
                study.solvers[
                    name
                ].problem.hydrodynamics.mobility.assemble = base.solvers[
                    name
                ].problem.hydrodynamics.mobility.assemble
    for mode, hz in enumerate(report["reference_wet_Hz"], 1):
        responses = {}
        for variant, study in variants.items():
            relevant = (
                MODELS
                if variant in ("baseline", "structural_mesh")
                else (
                    ("EB_2D", "KL_2D")
                    if variant == "section_grid"
                    else ("EB_3D", "KL_3D")
                )
            )
            for name in relevant:
                response, tip = study.solve(name, [hz])
                energy = study.energy(name, response)
                row = {
                    "variant": variant,
                    "model": name,
                    "resonance": mode,
                    "frequency_Hz": hz,
                    "displacement_real_m": float(tip[0].real),
                    "displacement_imag_m": float(tip[0].imag),
                    "Q_energy": float(energy.q_factor[0]),
                    "work_balance_error": float(energy.work_balance_errors.max()),
                }
                if name.endswith("3D"):
                    row.update(
                        collocation_error=float(response.fluid_errors.max()),
                        independent_velocity_error=independent_velocity_error(
                            study, name, response
                        ),
                    )
                if variant == "baseline":
                    responses[name] = (tip[0], energy.q_factor[0])
                    row.update(complex_response_change=0.0, energy_Q_change=0.0)
                else:
                    reference_tip, reference_q = responses[name]
                    row.update(
                        complex_response_change=float(
                            abs(tip[0] - reference_tip) / abs(reference_tip)
                        ),
                        energy_Q_change=float(
                            abs(energy.q_factor[0] - reference_q) / reference_q
                        ),
                    )
                records.append(row)
                print(
                    case,
                    variant,
                    name,
                    mode,
                    f"response change {row['complex_response_change']:.2%}",
                    flush=True,
                )
                write_json(
                    out / "validation.json",
                    {
                        "probe_definition": "Sader wet resonance estimates; one-variable-at-a-time changes, not full refined spectra.",
                        "configurations": {k: s.config for k, s in variants.items()},
                        "results": records,
                    },
                )


def timing_worker(args, case):
    """Fresh process, one thread, independent assembly without sweep sharing."""
    start = perf_counter()
    study = configuration(args, case)
    setup = perf_counter() - start
    start = perf_counter()
    study.prepare(args.model)
    coupling_setup = perf_counter() - start
    _, wet, _ = study.seeds()
    # Warm structural assembly and library routines, outside timing samples.
    study.solve(args.model, [wet[0] * 0.91])
    records = []
    for repeat in range(3):
        for mode, hz in enumerate(wet, 1):
            start = perf_counter()
            response, _ = study.solve(args.model, [hz])
            seconds = perf_counter() - start
            records.append(
                {
                    "repeat": repeat + 1,
                    "resonance": mode,
                    "seconds": seconds,
                    "equilibrium_error": float(response.relative_errors.max()),
                }
            )
    report = {
        "model": args.model,
        "case": case,
        "configuration": study.config,
        "source_hash": source_hash(),
        "setup_seconds": setup,
        "coupling_setup_seconds": coupling_setup,
        "samples": records,
        "peak_process_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "memory_scope": "Total process including both common study meshes and scientific runtime; not isolated solver allocation.",
        "thread_count": 1,
        "sharing": False,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "dolfinx": __import__("dolfinx").__version__,
        },
    }
    write_json(args.output / case / "timings" / f"{args.model}.json", report)
    print(
        case,
        args.model,
        f"median {np.median([r['seconds'] for r in records]):.4g}s",
        flush=True,
    )


def timings(args, case):
    for name in MODELS:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stage",
            "timing-worker",
            "--case",
            case,
            "--model",
            name,
            "--output",
            str(args.output),
            "--x-degree",
            str(args.x_degree),
            "--y-degree",
            str(args.y_degree),
            "--tolerance",
            str(args.tolerance),
        ]
        for option in ("section_nx", "section_ny"):
            if getattr(args, option, None) is not None:
                command.extend(
                    ["--" + option.replace("_", "-"), str(getattr(args, option))]
                )
        subprocess.run(command, check=True)


def odd_sweep(args, case):
    """Dense first torsional response; locate the peak in the computed sweep."""
    if case != "wide":
        raise ValueError("The antisymmetric example uses the wide plate.")
    out = args.output / case
    study = configuration(args, case)
    for name in ("KL_3D", "KL_2D"):
        study.prepare(name)
    with np.load(out / "spectrum.npz") as data:
        f = data["frequencies_Hz"]
        initial = {
            name: data[f"{name}_displacement_m"].copy()
            for name in ("KL_3D_odd", "KL_2D_odd")
        }
    windows = []
    for name, values in initial.items():
        peaks, _ = find_peaks(abs(values), prominence=0.15 * max(abs(values)))
        if not len(peaks):
            raise ValueError(f"No resolved antisymmetric peak for {name}.")
        center = f[peaks[0]]
        provisional, _ = study.solve(name, [center])
        q = float(study.energy(name, provisional).q_factor[0])
        half = min(0.45 * center, 1.7 * center / q)
        windows.append([max(100, center - half), center + half])
    frequencies = np.linspace(
        min(w[0] for w in windows), max(w[1] for w in windows), 161
    )
    arrays = {"frequencies_Hz": frequencies}
    rows = []
    for name in ("KL_3D_odd", "KL_2D_odd"):
        values = np.zeros(len(frequencies), complex)
        for i, hz in enumerate(frequencies):
            _, tip = study.solve(name, [hz])
            values[i] = tip[0]
            if i % 10 == 0:
                print(case, name, "odd", i + 1, len(frequencies), flush=True)
        arrays[f"{name}_displacement_m"] = values
        fit, selected = choose_fit(
            frequencies, values, [frequencies[0], frequencies[-1]]
        )
        at_f0, _ = study.solve(name, [fit.resonance_frequency])
        energy = study.energy(name, at_f0)
        rows.append(
            {
                "model": name,
                "fitted_f0_Hz": fit.resonance_frequency,
                "Q_SHO": fit.q_factor,
                "Q_energy": float(energy.q_factor[0]),
                "fit_error": fit.relative_error,
                "independent_velocity_error": independent_velocity_error(
                    study, name, at_f0
                )
                if "3D" in name
                else None,
            }
        )
        arrays[f"{name}_fit_frequency_Hz"] = frequencies[selected]
        arrays[f"{name}_fit_m"] = fit.fitted_amplitudes
        print(name, "odd", fit.resonance_frequency, fit.q_factor, flush=True)
        np.savez_compressed(out / "antisymmetric.npz", **arrays)
        write_json(
            out / "antisymmetric.json",
            {
                "results": rows,
                "load": "Opposite 1 nN forces near the two free corners; antisymmetric corner displacement.",
            },
        )


def adaptive(args, case):
    """Resolve peaks displaced from the initial reduced-model seed windows."""
    from scipy.signal import peak_widths

    out = args.output / case
    study = configuration(args, case)
    for name in MODELS:
        study.prepare(name)
    study.share_assembly()
    with np.load(out / "spectrum.npz") as saved:
        if not saved["completed"].all():
            raise ValueError("Complete the initial spectrum before adaptive windows.")
        f = saved["frequencies_Hz"].copy()
        values = {name: saved[f"{name}_displacement_m"].copy() for name in MODELS}
    extra = {}
    if case == "wide":
        # Finite-length 3D loading can move the third resonance past the old
        # 150 kHz plotting limit. Keep that range and add a search extension.
        extension = np.geomspace(CASES[case]["fmax"] * 1.01, 350e3, 41)
        extension_path = out / "extension.npz"
        if extension_path.exists():
            with np.load(extension_path) as saved:
                extended = {
                    name: saved[f"{name}_displacement_m"].copy() for name in MODELS
                }
        else:
            extended = {name: np.zeros(len(extension), complex) for name in MODELS}
            for i, hz in enumerate(extension):
                for name in MODELS:
                    _, tip = study.solve(name, [hz])
                    extended[name][i] = tip[0]
                print(case, "extension", i + 1, len(extension), flush=True)
            np.savez_compressed(
                extension_path,
                frequencies_Hz=extension,
                **{f"{name}_displacement_m": v for name, v in extended.items()},
            )
        f = np.concatenate((f, extension))
        values = {
            name: np.concatenate((v, extended[name])) for name, v in values.items()
        }
    windows = {}
    peaks_by_model = {}
    first = study.seeds()[1][0]
    for name in MODELS:
        peaks, _ = find_peaks(
            np.log(np.maximum(abs(values[name]), 1e-300)), prominence=0.4
        )
        peaks = peaks[f[peaks] > 0.6 * first][:3]
        if len(peaks) < 3:
            raise ValueError(
                f"{name}: fewer than three peaks found; extend the search."
            )
        widths = peak_widths(abs(values[name]), peaks, rel_height=1 - 1 / np.sqrt(2))
        sample_index = np.arange(len(f))
        width_hz = np.interp(widths[3], sample_index, f) - np.interp(
            widths[2], sample_index, f
        )
        peaks_by_model[name] = (f[peaks], width_hz)
        windows[name] = [
            [max(100, center - 1.3 * width), center + 1.3 * width]
            for center, width in zip(f[peaks], width_hz, strict=True)
        ]
    # Paired EB/KL 3D windows share physical kernel integrals at each frequency.
    needs_3d = case == "wide"
    prior = json.loads((out / "qfactor.json").read_text())["results"]
    if any(
        r["status"] != "ok" or r.get("fit_error", 1) > 0.025
        for r in prior
        if "3D" in r["model"]
    ):
        needs_3d = True
    for name in ("EB_3D", "KL_3D"):
        for center, width in zip(*peaks_by_model[name], strict=True):
            if np.count_nonzero(abs(f - center) < width / 2) < 12:
                needs_3d = True
    if needs_3d:
        paired = [
            [
                min(windows[m][mode][0] for m in ("EB_3D", "KL_3D")),
                max(windows[m][mode][1] for m in ("EB_3D", "KL_3D")),
            ]
            for mode in range(3)
        ]
        dense = np.unique(np.concatenate([np.linspace(*w, 81) for w in paired]))
        computed = {name: np.zeros(len(dense), complex) for name in ("EB_3D", "KL_3D")}
        for i, hz in enumerate(dense):
            for name, vector in computed.items():
                _, tip = study.solve(name, [hz])
                vector[i] = tip[0]
            if i % 10 == 0:
                print(case, "adaptive 3D", i + 1, len(dense), flush=True)
        for name, v in computed.items():
            extra[f"{name}_frequencies_Hz"] = dense
            extra[f"{name}_displacement_m"] = v
        np.savez_compressed(out / "adaptive.npz", **extra)
    for name in ("EB_2D", "KL_2D", "EB_Sader"):
        dense = np.unique(np.concatenate([np.linspace(*w, 81) for w in windows[name]]))
        _, v = study.solve(name, dense)
        extra[f"{name}_frequencies_Hz"] = dense
        extra[f"{name}_displacement_m"] = v
    np.savez_compressed(out / "adaptive.npz", **extra)
    rows, fits = [], {}
    for name in MODELS:
        all_f, all_v = f, values[name]
        if f"{name}_frequencies_Hz" in extra:
            combined_f = np.concatenate((f, extra[f"{name}_frequencies_Hz"]))
            combined_v = np.concatenate((values[name], extra[f"{name}_displacement_m"]))
            all_f, idx = np.unique(combined_f, return_index=True)
            all_v = combined_v[idx]
        for mode, window in enumerate(windows[name], 1):
            fit, selected = choose_fit(all_f, all_v, window)
            response, _ = study.solve(name, [fit.resonance_frequency])
            energy = study.energy(name, response)
            row = {
                "model": name,
                "resonance": mode,
                "status": "ok",
                "fitted_f0_Hz": fit.resonance_frequency,
                "Q_SHO": fit.q_factor,
                "Q_energy": float(energy.q_factor[0]),
                "fit_error": fit.relative_error,
                "peak_Hz": resonance_frequency(all_f[selected], all_v[selected]),
                "work_balance_error": float(energy.work_balance_errors.max()),
                "fit_sample_count": int(selected.sum()),
                "samples_in_linewidth": int(
                    np.count_nonzero(
                        abs(all_f - fit.resonance_frequency)
                        <= fit.resonance_frequency / fit.q_factor / 2
                    )
                ),
            }
            rows.append(row)
            fits[f"{name}_{mode}_frequency_Hz"] = all_f[selected]
            fits[f"{name}_{mode}_fit_m"] = fit.fitted_amplitudes
            print(
                case,
                name,
                mode,
                f"f0 {fit.resonance_frequency / 1e3:.3f} kHz Q {fit.q_factor:.3f}",
                flush=True,
            )
    write_json(
        out / "qfactor.json",
        {
            "results": rows,
            "energy_definition": "Maximum structural bending plus kinetic energy / model-specific dissipated work; no separate fluid stored energy.",
        },
    )
    write_json(
        out / "adaptive.json",
        {
            "windows_Hz": windows,
            "extra_3d_windows": needs_3d,
            "wide_search_extension_Hz": [151500, 350000] if case == "wide" else None,
        },
    )
    np.savez_compressed(out / "sho_fits.npz", **fits)
    with (out / "qfactor.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def section_validation(args, case):
    """Separate transverse refinement while holding the x grid fixed."""
    results = []
    for ny in (128, 256):
        study = configuration(args, case, section_nx=129, section_ny=ny)
        for mode, hz in enumerate(study.seeds()[1], 1):
            for model in ("EB_2D", "KL_2D"):
                start = perf_counter()
                response, tip = study.solve(model, [hz])
                energy = study.energy(model, response)
                row = {
                    "ny": ny,
                    "nx": 129,
                    "model": model,
                    "mode": mode,
                    "seconds": perf_counter() - start,
                    "amplitude_m": float(abs(tip[0])),
                    "phase_deg": float(np.angle(tip[0], deg=True)),
                    "Q_energy": float(energy.q_factor[0]),
                }
                results.append(row)
                write_json(args.output / case / "section_resolution.json", results)
                print(
                    case,
                    "section validation",
                    model,
                    ny,
                    mode,
                    row["Q_energy"],
                    flush=True,
                )


def refined_sections(args, case):
    """Replace underresolved wide-plate 2D curves with a finer section grid."""
    out = args.output / case
    study = configuration(args, case, section_nx=129, section_ny=256)
    prior = json.loads((out / "qfactor.json").read_text())["results"]
    models = ("EB_2D", "KL_2D")
    windows = {}
    for name in models:
        rows = [r for r in prior if r["model"] == name and r["status"] == "ok"]
        windows[name] = [
            [
                r["fitted_f0_Hz"] * (1 - 1.5 / r["Q_SHO"]),
                r["fitted_f0_Hz"] * (1 + 1.5 / r["Q_SHO"]),
            ]
            for r in rows
        ]
    paired = [
        [min(windows[m][i][0] for m in models), max(windows[m][i][1] for m in models)]
        for i in range(3)
    ]
    f = np.unique(
        np.concatenate(
            [np.geomspace(100, 350e3, 161), *[np.linspace(*w, 81) for w in paired]]
        )
    )
    arrays = {name: np.zeros(len(f), complex) for name in models}
    completed = np.zeros(len(f), bool)
    saved_path = out / "sections_refined.npz"
    if saved_path.exists():
        with np.load(saved_path) as saved:
            # Later adaptive refits can slightly change seed centers. Keep
            # the checkpoint's computed grid when continuing a saved run.
            f = saved["frequencies_Hz"].copy()
            completed = saved["completed"].copy()
            arrays = {name: saved[f"{name}_displacement_m"].copy() for name in models}
    metadata_path = out / "sections_refined_config.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["source_hash"] != source_hash() or metadata[
            "configuration"
        ] != json.loads(json.dumps(study.config)):
            raise ValueError(
                "Refined section checkpoint has different source/configuration."
            )
        windows = metadata["windows_Hz"]
    else:
        write_json(
            metadata_path,
            {
                "source_hash": source_hash(),
                "configuration": study.config,
                "windows_Hz": windows,
            },
        )
    for i, hz in enumerate(f):
        if completed[i]:
            continue
        for name in models:
            _, tip = study.solve(name, [hz])
            arrays[name][i] = tip[0]
        completed[i] = True
        if i % 10 == 0 or completed.all():
            np.savez_compressed(
                saved_path,
                frequencies_Hz=f,
                completed=completed,
                **{f"{name}_displacement_m": v for name, v in arrays.items()},
            )
            print(case, "refined 2D", int(completed.sum()), len(f), flush=True)
    rows = []
    for name in models:
        for mode, window in enumerate(windows[name], 1):
            fit, selected = choose_fit(f, arrays[name], window)
            response, _ = study.solve(name, [fit.resonance_frequency])
            energy = study.energy(name, response)
            rows.append(
                {
                    "model": name,
                    "resonance": mode,
                    "status": "ok",
                    "fitted_f0_Hz": fit.resonance_frequency,
                    "Q_SHO": fit.q_factor,
                    "Q_energy": float(energy.q_factor[0]),
                    "fit_error": fit.relative_error,
                    "peak_Hz": resonance_frequency(f[selected], arrays[name][selected]),
                    "work_balance_error": float(energy.work_balance_errors.max()),
                    "section_nx": 129,
                    "section_ny": 256,
                }
            )
    write_json(
        out / "section_refined_qfactor.json",
        {
            "configuration": study.config,
            "source_hash": source_hash(),
            "results": rows,
            "note": "Primary wide-plate 2D spectra use this finer grid; original 64-point curves remain archived.",
        },
    )


def refined_odd(args, case):
    out = args.output / case
    study = configuration(args, case, section_nx=129, section_ny=256)
    with np.load(out / "antisymmetric.npz") as saved:
        f = saved["frequencies_Hz"].copy()
    values, completed = np.zeros(len(f), complex), np.zeros(len(f), bool)
    path = out / "antisymmetric_refined.npz"
    if path.exists():
        with np.load(path) as saved:
            np.testing.assert_array_equal(f, saved["frequencies_Hz"])
            values, completed = (
                saved["displacement_m"].copy(),
                saved["completed"].copy(),
            )
    for i, hz in enumerate(f):
        if completed[i]:
            continue
        _, tip = study.solve("KL_2D_odd", [hz])
        values[i], completed[i] = tip[0], True
        if i % 10 == 0 or completed.all():
            np.savez_compressed(
                path, frequencies_Hz=f, displacement_m=values, completed=completed
            )
            print(case, "refined odd 2D", int(completed.sum()), len(f), flush=True)
    fit, _ = choose_fit(f, values, [f[0], f[-1]])
    response, _ = study.solve("KL_2D_odd", [fit.resonance_frequency])
    energy = study.energy("KL_2D_odd", response)
    write_json(
        out / "antisymmetric_refined.json",
        {
            "model": "KL_2D_odd",
            "fitted_f0_Hz": fit.resonance_frequency,
            "Q_SHO": fit.q_factor,
            "Q_energy": float(energy.q_factor[0]),
            "fit_error": fit.relative_error,
            "section_nx": 129,
            "section_ny": 256,
            "configuration": study.config,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "pilot",
            "sweep",
            "validation",
            "timings",
            "timing-worker",
            "odd",
            "adaptive",
            "refined-sections",
            "section-validation",
            "refined-odd",
        ),
        default="pilot",
    )
    parser.add_argument("--case", choices=tuple(CASES), default="slender")
    parser.add_argument("--x-degree", type=int, default=16)
    parser.add_argument("--y-degree", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=5e-6)
    parser.add_argument(
        "--broad-samples",
        type=int,
        default=161,
        help="Broadband count for the initial sweep.",
    )
    parser.add_argument(
        "--peak-samples",
        type=int,
        default=81,
        help="Per-resonance count for the initial sweep; adaptive windows use 81 samples.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", choices=MODELS, default="EB_3D")
    parser.add_argument("--section-nx", type=int)
    parser.add_argument("--section-ny", type=int)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/formulation_study"
    )
    args = parser.parse_args()
    stages = {
        "pilot": pilot,
        "sweep": sweep,
        "validation": validation,
        "timings": timings,
        "timing-worker": timing_worker,
        "odd": odd_sweep,
        "adaptive": adaptive,
        "refined-sections": refined_sections,
        "section-validation": section_validation,
        "refined-odd": refined_odd,
    }
    stages[args.stage](args, args.case)


if __name__ == "__main__":
    main()
