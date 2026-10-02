"""Plot the saved extensive formulation study without rerunning any solver.

Run benchmarks/formulation_study.py first. This script needs only NumPy and
Matplotlib, so the expensive scientific environment is unnecessary for plots.
All spectral lines join actual computed samples; no smoothing/interpolation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
CASES = ("beam", "slender", "wide")
TITLES = {
    "beam": "800 × 50 × 5 µm",
    "slender": "500 × 50 × 5 µm",
    "wide": "500 × 250 × 5 µm",
}
MODELS = ("EB_3D", "KL_3D", "EB_2D", "KL_2D", "EB_Sader")
LABELS = {
    "EB_3D": "EB · weighted 3D",
    "KL_3D": "KL · weighted 3D",
    "EB_2D": "EB · 2D",
    "KL_2D": "KL · 2D",
    "EB_Sader": "EB · Sader",
}
COLORS = dict(
    zip(MODELS, ("#2166ac", "#b2182b", "#67a9cf", "#ef8a62", "#343434"), strict=True)
)
STYLES = dict(zip(MODELS, ("-", "-", "--", "--", ":"), strict=True))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def legend_models(fig, *, energy=False):
    handles = [
        Line2D([], [], color=COLORS[m], linestyle=STYLES[m], label=LABELS[m])
        for m in MODELS
    ]
    if energy:
        handles += [
            Line2D([], [], color="black", marker="o", linestyle="", label="SHO fit"),
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                markerfacecolor="white",
                linestyle="",
                label="Energy/work",
            ),
        ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=4 if energy else 5,
        frameon=False,
        fontsize=9,
    )


def save(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(output / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def spectra(directory):
    with np.load(directory / "spectrum.npz") as data:
        if not data["completed"].all():
            raise ValueError(f"{directory.name}: frequency sweep is incomplete.")
        arrays = {key: data[key].copy() for key in data.files}
    # Optional model-specific targeted windows add actual solves where the
    # initial Sader-seeded grid missed a large finite-length frequency shift.
    for adaptive in (directory / "extension.npz", directory / "adaptive.npz"):
        if not adaptive.exists():
            continue
        with np.load(adaptive) as data:
            for m in MODELS:
                key = f"{m}_displacement_m"
                if key not in data:
                    continue
                f = np.concatenate(
                    (
                        arrays.get(f"{m}_frequencies_Hz", arrays["frequencies_Hz"]),
                        data[f"{m}_frequencies_Hz"]
                        if f"{m}_frequencies_Hz" in data
                        else data["frequencies_Hz"],
                    )
                )
                values = np.concatenate((arrays[key], data[key]))
                unique, indices = np.unique(f, return_index=True)
                arrays[f"{m}_frequencies_Hz"] = unique
                arrays[key] = values[indices]
    for m in MODELS:
        arrays.setdefault(f"{m}_frequencies_Hz", arrays["frequencies_Hz"])
    refined = directory / "sections_refined.npz"
    if (directory / "section_refined_qfactor.json").exists():
        with np.load(refined) as data:
            if not data["completed"].all():
                raise ValueError("Refined 2D spectrum is incomplete.")
            for m in ("EB_2D", "KL_2D"):
                arrays[f"{m}_frequencies_Hz"] = data["frequencies_Hz"].copy()
                arrays[f"{m}_displacement_m"] = data[f"{m}_displacement_m"].copy()
    return arrays


def plot_all(args):
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.17,
            "lines.linewidth": 1.7,
            "axes.titleweight": "semibold",
            "pdf.fonttype": 42,
        }
    )
    available = [c for c in CASES if (args.results / c / "qfactor.json").exists()]
    if not available:
        raise ValueError("No completed study cases found.")
    data = {c: spectra(args.results / c) for c in available}
    qrows = {
        c: load_json(args.results / c / "qfactor.json")["results"] for c in available
    }
    for c in available:
        refined = args.results / c / "section_refined_qfactor.json"
        if refined.exists():
            rows = load_json(refined)["results"]
            replaced = {(r["model"], r["resonance"]): r for r in rows}
            qrows[c] = [replaced.get((r["model"], r["resonance"]), r) for r in qrows[c]]
    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(5.1 * len(available), 4.3),
        squeeze=False,
        layout="constrained",
    )
    for case, ax in zip(available, axes[0], strict=True):
        for m in MODELS:
            ax.loglog(
                data[case][f"{m}_frequencies_Hz"] / 1e3,
                abs(data[case][f"{m}_displacement_m"]) * 1e9,
                color=COLORS[m],
                linestyle=STYLES[m],
            )
        ax.set(
            title=TITLES[case],
            xlabel="Drive frequency [kHz]",
            ylabel="Corner displacement [nm]",
        )
    fig.suptitle(
        "Broadband displacement · equal 1 nN forces at the free corners", fontsize=14
    )
    legend_models(fig)
    save(fig, args.output, "spectral_displacement_broadband")
    for case in available:
        rows = qrows[case]
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), layout="constrained")
        for mode in range(1, 4):
            selected = [
                r for r in rows if r["resonance"] == mode and r["status"] == "ok"
            ]
            if not selected:
                continue
            centers = [r["fitted_f0_Hz"] for r in selected]
            widths = [r["fitted_f0_Hz"] / r["Q_SHO"] for r in selected]
            low, high = (
                min(centers) - 1.25 * max(widths),
                max(centers) + 1.25 * max(widths),
            )
            for m in MODELS:
                f = data[case][f"{m}_frequencies_Hz"]
                values = data[case][f"{m}_displacement_m"]
                mask = (f >= low) & (f <= high)
                indices = np.flatnonzero(mask)
                if len(indices):
                    mask[max(0, indices[0] - 1) : min(len(f), indices[-1] + 2)] = True
                axes[0, mode - 1].plot(
                    f[mask] / 1e3,
                    abs(values[mask]) * 1e9,
                    color=COLORS[m],
                    linestyle=STYLES[m],
                )
                row = next((r for r in selected if r["model"] == m), None)
                if row is not None:
                    phase_mask = (
                        abs(f - row["fitted_f0_Hz"])
                        <= 1.25 * row["fitted_f0_Hz"] / row["Q_SHO"]
                    )
                    axes[1, mode - 1].plot(
                        f[phase_mask] / 1e3,
                        np.angle(values[phase_mask], deg=True),
                        color=COLORS[m],
                        linestyle=STYLES[m],
                    )
            axes[0, mode - 1].set(
                title=f"Symmetric bending resonance {mode}",
                ylabel="Displacement [nm]",
                xlim=(low / 1e3, high / 1e3),
            )
            axes[1, mode - 1].set_title("Phase near each model's resonance", fontsize=9)
            axes[1, mode - 1].set(
                xlabel="Drive frequency [kHz]",
                ylabel="Phase [degrees]",
                xlim=(low / 1e3, high / 1e3),
            )
        fig.suptitle(TITLES[case] + " · resonance detail", fontsize=14)
        legend_models(fig)
        save(fig, args.output, f"spectral_displacement_{case}")
    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(5.1 * len(available), 4.4),
        squeeze=False,
        layout="constrained",
    )
    for case, ax in zip(available, axes[0], strict=True):
        for m in MODELS:
            rows = [r for r in qrows[case] if r["model"] == m and r["status"] == "ok"]
            f = np.array([r["fitted_f0_Hz"] for r in rows]) / 1e3
            ax.plot(
                f,
                [r["Q_SHO"] for r in rows],
                "o",
                color=COLORS[m],
                linestyle=STYLES[m],
                markersize=5,
            )
            ax.plot(
                f,
                [r["Q_energy"] for r in rows],
                "o",
                color=COLORS[m],
                markerfacecolor="white",
                linestyle="none",
                markersize=5,
            )
        ax.set(
            title=TITLES[case],
            xlabel="Fitted resonance frequency [kHz]",
            ylabel="Q factor",
            ylim=(0, None),
        )
    fig.suptitle(
        "Q versus resonance frequency · first three symmetric bending resonances",
        fontsize=14,
    )
    legend_models(fig, energy=True)
    save(fig, args.output, "qfactor_vs_frequency")
    wide = args.results / "wide"
    if (wide / "antisymmetric.json").exists() and len(
        load_json(wide / "antisymmetric.json")["results"]
    ) == 2:
        with np.load(wide / "antisymmetric.npz") as odd:
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
            f = odd["frequencies_Hz"] / 1e3
            for m in ("KL_3D", "KL_2D"):
                values = odd[f"{m}_odd_displacement_m"]
                if m == "KL_2D" and (wide / "antisymmetric_refined.json").exists():
                    with np.load(wide / "antisymmetric_refined.npz") as refined:
                        values = refined["displacement_m"].copy()
                axes[0].plot(
                    f,
                    abs(values) * 1e9,
                    color=COLORS[m],
                    linestyle=STYLES[m],
                    label=LABELS[m],
                )
                axes[1].plot(
                    f, np.angle(values, deg=True), color=COLORS[m], linestyle=STYLES[m]
                )
            axes[0].set(
                xlabel="Drive frequency [kHz]", ylabel="Antisymmetric displacement [nm]"
            )
            axes[1].set(xlabel="Drive frequency [kHz]", ylabel="Phase [degrees]")
            axes[0].legend(frameon=False)
            fig.suptitle(
                "500 × 250 × 5 µm · first antisymmetric plate resonance", fontsize=14
            )
            save(fig, args.output, "antisymmetric_wide")
    plot_accuracy(args, available)
    plot_timings(args, available)
    make_report(args, available, data, qrows)
    print(f"Plots and report: {args.output.resolve()}")


def plot_accuracy(args, available):
    keys = (
        "pressure_degree",
        "collocation",
        "quadrature",
        "structural_mesh",
        "section_grid",
    )
    labels = (
        "Pressure degree",
        "Collocation",
        "Quadrature",
        "Structural mesh",
        "2D section grid",
    )
    records = {
        c: load_json(args.results / c / "validation.json")["results"]
        for c in available
        if (args.results / c / "validation.json").exists()
    }
    if not records:
        return
    fig, axes = plt.subplots(
        2,
        len(records),
        figsize=(5.3 * len(records), 8.1),
        squeeze=False,
        layout="constrained",
    )
    for column, (case, rows) in enumerate(records.items()):
        for row, (metric, ylabel) in enumerate(
            (
                ("complex_response_change", "Maximum displacement change [%]"),
                ("energy_Q_change", "Maximum energy-Q change [%]"),
            )
        ):
            ax = axes[row, column]
            for mi, m in enumerate(MODELS):
                y = [
                    max(
                        (
                            r[metric] * 100
                            for r in rows
                            if r["variant"] == k and r["model"] == m
                        ),
                        default=np.nan,
                    )
                    for k in keys
                ]
                ax.semilogy(
                    np.arange(len(keys)) + 0.04 * (mi - 2),
                    np.maximum(y, 1e-6),
                    "o",
                    color=COLORS[m],
                    markersize=5,
                )
            ax.set(
                title=TITLES[case] if row == 0 else "",
                ylabel=ylabel,
                xticks=np.arange(len(keys)),
                xticklabels=labels if row else [],
            )
            ax.tick_params(axis="x", labelrotation=35)
    fig.suptitle(
        "Numerical refinement · probes at three wet resonance estimates", fontsize=14
    )
    legend_models(fig)
    save(fig, args.output, "numerical_refinement")


def plot_timings(args, available):
    ready = [
        c
        for c in available
        if all((args.results / c / "timings" / f"{m}.json").exists() for m in MODELS)
    ]
    if not ready:
        return
    fig, axes = plt.subplots(
        1,
        len(ready),
        figsize=(5.1 * len(ready), 4.5),
        squeeze=False,
        layout="constrained",
    )
    for case, ax in zip(ready, axes[0], strict=True):
        for i, m in enumerate(MODELS):
            rows = load_json(args.results / case / "timings" / f"{m}.json")["samples"]
            samples = np.array([r["seconds"] for r in rows])
            ax.bar(i, np.median(samples), color=COLORS[m], width=0.65)
            ax.plot(np.full(len(samples), i), samples, ".", color="black", markersize=3)
        ax.set(
            yscale="log",
            title=TITLES[case],
            ylabel="Seconds per frequency",
            xticks=range(5),
            xticklabels=("EB 3D", "KL 3D", "EB 2D", "KL 2D", "Sader"),
        )
        ax.tick_params(axis="x", labelrotation=25)
    fig.suptitle(
        "Independent single-thread solves · median and nine timing samples", fontsize=14
    )
    save(fig, args.output, "runtime_comparison")


def make_report(args, available, data, qrows):
    lines = [
        "# Formulation comparison",
        "",
        "All results use silicon (E=169 GPa, ρ=2330 kg/m³, ν=0.3), water (ρ=997 kg/m³, μ=0.89 mPa s), and 5 µm thickness. Symmetric forcing applies 1 nN near each free corner; EB receives the same total force.",
        "",
        "The weighted pressure has degrees 16 in x and 8 in y, with 24 × 6 independent EB collocation points and 24 × 12 KL points. The kernel integration tolerance is 5×10⁻⁶. EB uses 48 cubic elements; KL meshes are 40×6 (800 µm beam), 32×6 (slender), and 32×16 (wide). The initial 2D section uses 64 transverse points; KL has 65 longitudinal samples. The primary wide-plate 2D spectra use 256 transverse and 129 longitudinal points after the refinement checks identified sensitivity at higher frequencies; original spectra are retained.",
        "",
        "Spectral lines connect actual solves. Each initial grid has 161 logarithmic broadband samples plus 81 linear samples around each of three Sader-seeded windows. Additional targeted windows, when present, resolve peaks shifted outside those windows. Phase panels focus on ±1.25 fitted linewidths around each model's resonance. SHO fits use displacement magnitudes and a coherent background; f₀ differs from the amplitude maximum. Energy Q uses structural stored energy and model-specific dissipated work, without a separate stored-fluid-energy term.",
        "",
        "EB and Sader curves for the 500×250 µm case are comparisons outside a slender-beam geometry. KL retains transverse deformation and supports antisymmetric loading. Agreement of two reduced models does not establish validity for a wide plate.",
        "",
    ]
    lines += [
        "## Effect of replacing local 2D loading with weighted 3D",
        "",
        "Positive shifts mean the weighted-3D result is larger. Each comparison keeps the structural formulation and forcing fixed; the wide-plate 2D reference uses its refined grid.",
        "",
        "| Geometry [µm] | Structure | f₀ shift: resonance 1 | f₀ shift: resonance 3 | SHO-Q shift: resonance 3 |",
        "|---|---|---:|---:|---:|",
    ]
    for case in available:
        lookup = {
            (r["model"], r["resonance"]): r for r in qrows[case] if r["status"] == "ok"
        }
        for structural in ("EB", "KL"):
            required = [
                (f"{structural}_{fluid}", mode)
                for fluid in ("3D", "2D")
                for mode in (1, 3)
            ]
            if not all(key in lookup for key in required):
                continue
            changes = [
                lookup[(f"{structural}_3D", mode)]["fitted_f0_Hz"]
                / lookup[(f"{structural}_2D", mode)]["fitted_f0_Hz"]
                - 1
                for mode in (1, 3)
            ]
            qchange = (
                lookup[(f"{structural}_3D", 3)]["Q_SHO"]
                / lookup[(f"{structural}_2D", 3)]["Q_SHO"]
                - 1
            )
            lines.append(
                f"| {TITLES[case]} | {structural} | {changes[0]:+.2%} | {changes[1]:+.2%} | {qchange:+.2%} |"
            )
    lines.append("")
    csvrows = []
    for case in available:
        rows = qrows[case]
        lines += [
            f"## {TITLES[case]}",
            "",
            f"Initial frequency grid: {len(data[case]['frequencies_Hz'])} completed samples.",
            "",
            "| Formulation | Resonance | f₀ [kHz] | Q SHO | Q energy | Fit error | Points / linewidth |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            if r["status"] == "ok":
                model_f = data[case][f"{r['model']}_frequencies_Hz"]
                count = int(
                    np.count_nonzero(
                        abs(model_f - r["fitted_f0_Hz"])
                        <= r["fitted_f0_Hz"] / r["Q_SHO"] / 2
                    )
                )
                csvrows.append(
                    {
                        "case": case,
                        **r,
                        "points_in_linewidth": count,
                        "model_frequency_samples": len(model_f),
                    }
                )
                lines.append(
                    f"| {LABELS[r['model']]} | {r['resonance']} | {r['fitted_f0_Hz'] / 1e3:.4f} | {r['Q_SHO']:.3f} | {r['Q_energy']:.3f} | {r['fit_error']:.2%} | {count} |"
                )
            else:
                csvrows.append({"case": case, **r})
                lines.append(
                    f"| {LABELS[r['model']]} | {r['resonance']} | unresolved: {r['reason']} | | | | |"
                )
        path = args.results / case / "validation.json"
        if path.exists():
            validation = load_json(path)
            lines += [
                "",
                "One-variable-at-a-time refinement of the initial grids at three wet resonance estimates (these are response probes, not full refined-spectrum convergence):",
                "",
                "| Refinement | Maximum complex displacement change | Maximum energy-Q change |",
                "|---|---:|---:|",
            ]
            for variant in (
                "pressure_degree",
                "collocation",
                "quadrature",
                "structural_mesh",
                "section_grid",
            ):
                selected = [r for r in validation["results"] if r["variant"] == variant]
                lines.append(
                    f"| {variant.replace('_', ' ')} | {max(r['complex_response_change'] for r in selected):.3%} | {max(r['energy_Q_change'] for r in selected):.3%} |"
                )
            baseline = [
                r
                for r in validation["results"]
                if r["variant"] == "baseline" and "independent_velocity_error" in r
            ]
            lines += [
                "",
                f"Maximum baseline no-slip error at 35 independent off-grid surface points: {max(r['independent_velocity_error'] for r in baseline):.2%}.",
            ]
        meta = load_json(args.results / case / "sweep.json")
        diagnostics = [v for r in meta["diagnostics"] for v in r["models"].values()]
        lines += [
            "",
            f"Maximum equilibrium residual across the sweep: {max(r['equilibrium_error'] for r in diagnostics):.3g}; maximum input/fluid-work discrepancy: {max(r['work_balance_error'] for r in diagnostics):.3g}.",
        ]
        section_path = args.results / case / "section_resolution.json"
        if section_path.exists():
            probes = load_json(section_path)
            lines += [
                "",
                "Additional wide-plate section refinement fixes 129 longitudinal points and increases the transverse count from 128 to 256:",
                "",
                "| Model | Resonance estimate | Q energy at 128 | Q energy at 256 | Change |",
                "|---|---:|---:|---:|---:|",
            ]
            for m in ("EB_2D", "KL_2D"):
                for mode in (1, 2, 3):
                    a, b = [
                        next(
                            r
                            for r in probes
                            if r["model"] == m and r["mode"] == mode and r["ny"] == n
                        )
                        for n in (128, 256)
                    ]
                    lines.append(
                        f"| {LABELS[m]} | {mode} | {a['Q_energy']:.3f} | {b['Q_energy']:.3f} | {abs(b['Q_energy'] / a['Q_energy'] - 1):.2%} |"
                    )
            lines += [
                "",
                "The refined 2D curves improve the comparison; the last grid change still gives a few-percent Q variation in the highest KL mode. Structural-mesh and section-grid sensitivity should be considered when interpreting small model differences.",
            ]
        if all(
            (args.results / case / "timings" / f"{m}.json").exists() for m in MODELS
        ):
            lines += [
                "",
                "Independent timings use fresh processes and one BLAS/OpenMP thread, after warm-up. Mobility is assembled independently for each formulation. Setup is separate. RSS includes the scientific runtime and both common study meshes, so it is not isolated solver memory.",
                "",
                "| Model | Median solve [s] | Min–max [s] | Coupling setup [s] | Peak process RSS [MiB] |",
                "|---|---:|---:|---:|---:|",
            ]
            for m in MODELS:
                timing = load_json(args.results / case / "timings" / f"{m}.json")
                samples = [r["seconds"] for r in timing["samples"]]
                lines.append(
                    f"| {LABELS[m]} | {np.median(samples):.4g} | {min(samples):.4g}–{max(samples):.4g} | {timing['coupling_setup_seconds']:.3g} | {timing['peak_process_RSS_MiB']:.1f} |"
                )
        lines.append("")
    odd = args.results / "wide/antisymmetric.json"
    if odd.exists():
        lines += [
            "## Wide plate: antisymmetric load",
            "",
            "Opposite corner forces isolate the first antisymmetric response, which EB does not represent. The dense window contains 161 frequency samples.",
            "",
            "| Model | f₀ [kHz] | Q SHO | Q energy | Fit error |",
            "|---|---:|---:|---:|---:|",
        ]
        odd_rows = load_json(odd)["results"]
        refined = args.results / "wide/antisymmetric_refined.json"
        if refined.exists():
            refined_row = load_json(refined)
            odd_rows = [
                refined_row if r["model"] == "KL_2D_odd" else r for r in odd_rows
            ]
        for r in odd_rows:
            lines.append(
                f"| {LABELS[r['model'].removesuffix('_odd')]} | {r['fitted_f0_Hz'] / 1e3:.4f} | {r['Q_SHO']:.3f} | {r['Q_energy']:.3f} | {r['fit_error']:.2%} |"
            )
    flow_metadata = args.results / "flow/parameters.json"
    if flow_metadata.exists():
        field = load_json(flow_metadata)
        lines += [
            "",
            "## Antisymmetric fluid-field example",
            "",
            f"The flow example uses the weighted-3D KL displacement at {field['frequency_Hz'] / 1e3:.3f} kHz and reconstructs a fresh local 2D field near x={field['section_x_m'] * 1e6:.2f} µm. In-phase and quadrature snapshots, complex field data, and PNG/PDF figures are stored under flow/. The displayed loss density belongs to this 2D approximation and does not replace the weighted-3D Q denominator.",
            "",
        ]
    lines += [
        "",
        "## Files and reproducibility",
        "",
        "Per-case NPZ files contain complex corner displacement in metres and frequency in Hz. JSON files retain configuration, source SHA256, integration/solve diagnostics, Q definitions, refinement probes, and independent timing samples. The combined qfactor_all.csv uses the primary refined wide-plate 2D results; original per-case coarse-grid results remain available. The shared mobility used to accelerate spectra was checked against a separate EB assembly; no coupled response is shared. Sweep timing is not used to rank costs.",
        "",
        "Generate a case with `PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/formulation_study.py --stage sweep --case wide --resume`. Run the `validation`, `timings`, `adaptive`, and (wide only) `odd` stages for supplementary checks. Regenerate figures with `python examples/formulation_spectra.py`.",
        "",
    ]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    with (args.output / "qfactor_all.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = list(dict.fromkeys(k for r in csvrows for k in r))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csvrows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results/formulation_study"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/formulation_study/figures"
    )
    plot_all(parser.parse_args())


if __name__ == "__main__":
    main()
