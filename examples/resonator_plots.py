"""Render saved resonator spectra, resonance/Q estimates and 2D fluid fields.

Only NumPy and Matplotlib are needed. No solver or FEM module is imported.
Run: python examples/resonator_plots.py --results results/slender_3d
The companion showcase driver writes spectrum.npz, report.json, resonance
windows and complex fields; figures can be regenerated without another solve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D


MODEL_LABELS = {
    "EB_2D": "Euler–Bernoulli · 2D Stokes",
    "KL_2D": "Kirchhoff–Love · 2D Stokes",
    "EB_3D": "Euler–Bernoulli · weighted 3D Stokes",
    "KL_3D": "Kirchhoff–Love · weighted 3D Stokes",
    "EB_Sader": "Euler–Bernoulli · Sader",
}
STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#68737d",
    "axes.linewidth": 0.7,
    "grid.color": "#dce2e7",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
BLUE = "#176b91"
ORANGE = "#d56a2a"
GRAY = "#606b75"


def _arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _response(arrays, observable="displacement_m"):
    frequency = np.asarray(arrays["frequency_Hz"], dtype=float)
    displacement = np.asarray(arrays[observable], dtype=complex)
    if (
        frequency.ndim != 1
        or len(frequency) < 2
        or displacement.shape != frequency.shape
        or not np.isfinite(frequency).all()
        or np.any(frequency <= 0)
        or np.any(np.diff(frequency) <= 0)
        or not np.isfinite(displacement).all()
    ):
        raise ValueError("Saved response needs increasing positive Hz and finite complex displacement.")
    return frequency, displacement


def _title(report):
    parameters = report["parameters"]
    geometry = parameters["geometry"]
    dimensions = " × ".join(
        f"{geometry[name] * 1e6:g}" for name in ("length", "width", "thickness")
    )
    formulation = str(parameters["formulation"]).upper()
    model = str(parameters["model"]).lower()
    suffix = "Sader" if model == "sader" else model.upper()
    name = MODEL_LABELS.get(f"{formulation}_{suffix}", f"{formulation} · {model}")
    return f"{dimensions} µm cantilever\n{name}"


def _save(figure, output, stem):
    figure.savefig(output / f"{stem}.png", dpi=220)
    figure.savefig(output / f"{stem}.pdf")
    plt.close(figure)
    return stem


def _segments(frequency):
    """Avoid joining a large unsampled gap inside an otherwise dense window."""
    steps = np.diff(frequency)
    split = np.flatnonzero(steps > 6 * np.median(steps)) + 1
    return np.split(np.arange(len(frequency)), split)


def _window_line(axis, frequency, values, **kwargs):
    for indices in _segments(frequency):
        axis.plot(frequency[indices] / 1e3, values[indices], **kwargs)


def _phase(displacement):
    return np.rad2deg(np.unwrap(np.angle(displacement)))


def _windows(directory, rows):
    return [(row, _arrays(directory / row["fit_file"])) for row in rows]


def plot_spectra(directory, output, report, rows):
    """Draw actual spectrum samples and three independently resolved windows."""
    arrays = _arrays(directory / "spectrum.npz")
    frequency, displacement = _response(arrays, "observable_displacement_m")
    windows = _windows(directory, rows)
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 6.6), sharex=True, layout="constrained")
    amplitude = np.abs(displacement) * 1e9
    axes[0].scatter(frequency / 1e3, np.where(amplitude > 0, amplitude, np.nan), s=7, color=BLUE)
    # Global phase stays wrapped: no invented unwrapping across sparse gaps.
    axes[1].scatter(frequency / 1e3, np.rad2deg(np.angle(displacement)), s=7, color=BLUE)
    for row, window in windows:
        local_f, local_u = _response(window)
        _window_line(axes[0], local_f, np.abs(local_u) * 1e9, color=BLUE, lw=1.5)
        local_phase = np.rad2deg(np.angle(local_u))
        # Wrapped phase must not get a connecting stroke across the branch cut.
        cuts = np.flatnonzero(np.abs(np.diff(local_phase)) > 180) + 1
        for indices in np.split(np.arange(len(local_f)), cuts):
            if len(indices):
                _window_line(axes[1], local_f[indices], local_phase[indices], color=BLUE, lw=1.5)
        for axis in axes:
            axis.axvline(row["fitted_f0_Hz"] / 1e3, color=GRAY, lw=0.8, ls=":", alpha=0.7)
        axes[0].annotate(
            f"{row['resonance']}",
            (row["amplitude_peak_Hz"] / 1e3, np.max(np.abs(local_u)) * 1e9),
            xytext=(0, 7), textcoords="offset points", ha="center", color=GRAY,
        )
    axes[0].set(yscale="log", ylabel="Corner displacement amplitude [nm]")
    axes[1].set(xlabel="Drive frequency [kHz]", ylabel="Displacement phase [°]", ylim=(-190, 190))
    axes[1].set_yticks([-180, -90, 0, 90, 180])
    for axis in axes:
        axis.grid(alpha=0.8)
    axes[0].legend(handles=[
        Line2D([], [], marker="o", markersize=4, color=BLUE, lw=0, label="Computed samples"),
        Line2D([], [], color=BLUE, lw=1.5, label="Resolved resonance windows"),
        Line2D([], [], color=GRAY, lw=0.8, ls=":", label="Fitted undamped f₀"),
    ], loc="lower right")
    figure.suptitle(_title(report), fontsize=14)
    names = [_save(figure, output, "spectrum_amplitude_phase")]

    figure, axes = plt.subplots(
        2, len(windows), figsize=(4.3 * len(windows), 6.0),
        squeeze=False, sharex="col", layout="constrained",
    )
    for column, (row, window) in enumerate(windows):
        local_f, local_u = _response(window)
        amplitude_axis, phase_axis = axes[:, column]
        _window_line(amplitude_axis, local_f, np.abs(local_u) * 1e9, color=BLUE, lw=1.8)
        amplitude_axis.scatter(local_f / 1e3, np.abs(local_u) * 1e9, s=8, color=BLUE, alpha=0.7)
        fitted = np.asarray(window["fit_m"], dtype=float)
        if fitted.shape != local_f.shape or not np.isfinite(fitted).all():
            raise ValueError("Saved SHO amplitude must match its resonance frequency array.")
        _window_line(amplitude_axis, local_f, fitted * 1e9, color=ORANGE, lw=1.4, ls="--")
        _window_line(phase_axis, local_f, _phase(local_u), color=BLUE, lw=1.6)
        for axis in (amplitude_axis, phase_axis):
            axis.axvline(row["fitted_f0_Hz"] / 1e3, color=GRAY, ls=":", lw=0.8)
            axis.grid(alpha=0.8)
        amplitude_axis.set_title(f"Resonance {row['resonance']} · Q = {row['Q_SHO']:.2f}")
        phase_axis.set_xlabel("Drive frequency [kHz]")
    axes[0, 0].set_ylabel("Displacement amplitude [nm]")
    axes[1, 0].set_ylabel("Unwrapped local phase [°]")
    axes[0, 0].legend(handles=[
        Line2D([], [], color=BLUE, lw=1.8, label="Complex response"),
        Line2D([], [], color=ORANGE, lw=1.4, ls="--", label="SHO amplitude fit"),
    ])
    figure.suptitle("Resolved bending resonances\n" + _title(report), fontsize=13)
    names.append(_save(figure, output, "resolved_resonances"))
    return names


def plot_resonance_q(output, report, rows):
    """Keep undamped f0, amplitude peak and the two Q definitions distinct."""
    resonance = np.array([row["resonance"] for row in rows])
    fitted = np.array([row["fitted_f0_Hz"] for row in rows]) / 1e3
    peak = np.array([row["amplitude_peak_Hz"] for row in rows]) / 1e3
    sho_q = np.array([row["Q_SHO"] for row in rows])
    energy_q = np.array([row["Q_energy"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), layout="constrained")
    axes[0].plot(resonance, fitted, "o-", color=BLUE, lw=1.2, label="Fitted undamped f₀")
    axes[0].plot(resonance, peak, "D--", color=ORANGE, markerfacecolor="white", lw=1.2, label="Amplitude peak")
    axes[0].set(xlabel="Bending resonance", ylabel="Frequency [kHz]", title="Resonance frequencies")
    axes[0].set_xticks(resonance)
    axes[1].plot(fitted, sho_q, "o-", color=BLUE, lw=1.2, label="SHO amplitude fit")
    axes[1].plot(fitted, energy_q, "s--", color=ORANGE, markerfacecolor="white", lw=1.2, label="Energy / fluid work")
    for i, x, y in zip(resonance, fitted, sho_q, strict=True):
        axes[1].annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points", color=GRAY)
    axes[1].set(xlabel="Fitted undamped f₀ [kHz]", ylabel="Quality factor Q [–]", title="SHO and energy Q")
    for axis in axes:
        axis.grid(alpha=0.8)
        axis.legend(loc="upper left")
        axis.margins(x=0.1, y=0.12)
    figure.suptitle(_title(report), fontsize=14)
    return _save(figure, output, "resonance_qfactor")


def _section_field(arrays):
    points = np.asarray(arrays["points"], dtype=float)
    velocity = np.asarray(arrays["velocity"], dtype=complex)
    dissipation = np.asarray(arrays["mean_dissipation"], dtype=float)
    if (
        points.ndim != 3 or points.shape[-1] != 2
        or min(points.shape[:2]) < 2 or velocity.shape != points.shape
        or dissipation.shape != points.shape[:2] or not np.isfinite(points).all()
    ):
        raise ValueError("Saved field needs rectilinear (z,y,2) points/velocity and (z,y) dissipation.")
    y, z = points[0, :, 0], points[:, 0, 1]
    yy, zz = np.meshgrid(y, z)
    if not np.allclose(points, np.stack((yy, zz), axis=-1), rtol=0, atol=1e-14):
        raise ValueError("Saved field coordinates must form a rectilinear section.")
    for axis in (y, z):
        if np.any(np.diff(axis) <= 0) or not np.allclose(np.diff(axis), np.diff(axis)[0]):
            raise ValueError("Streamline coordinates must be increasing and uniformly spaced.")
    mask = np.asarray(arrays.get("singular_points", np.zeros(dissipation.shape)), dtype=bool)
    if mask.shape != dissipation.shape:
        raise ValueError("Saved singular-point mask must match the section.")
    mask = mask | ~np.isfinite(velocity).all(axis=-1)
    return y * 1e6, z * 1e6, velocity, dissipation, mask


def plot_field(directory, output, report, field_record):
    """Plot complex phase snapshots and mean viscous loss in an unbounded section."""
    arrays = _arrays(directory / field_record["file"])
    y, z, velocity, dissipation, mask = _section_field(arrays)
    reference = float(arrays["phase_reference_rad"])
    phases = (-reference, -reference + np.pi / 2)
    snapshots = [np.real(velocity * np.exp(1j * phase)) for phase in phases]
    speeds = [np.linalg.norm(snapshot, axis=-1) * 1e3 for snapshot in snapshots]
    finite_speeds = np.concatenate([speed[~mask & np.isfinite(speed)] for speed in speeds])
    vmax = max(float(np.max(finite_speeds)) if finite_speeds.size else 0.0, np.finfo(float).tiny)
    speed_norm = Normalize(vmin=0, vmax=vmax)
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.6), layout="constrained")
    for axis, snapshot, speed, label in zip(
        axes[:2], snapshots, speeds, ("In-phase velocity", "Quarter-cycle velocity"), strict=True
    ):
        image = axis.pcolormesh(
            y, z, np.ma.array(speed, mask=mask | ~np.isfinite(speed)),
            cmap="viridis", norm=speed_norm, shading="auto", rasterized=True,
        )
        axis.streamplot(
            y, z,
            np.ma.array(snapshot[..., 0], mask=mask),
            np.ma.array(snapshot[..., 1], mask=mask),
            color="white", density=1.1, linewidth=0.55, arrowsize=0.65,
        )
        axis.set_title(label)
        figure.colorbar(image, ax=axis, location="bottom", pad=0.05, shrink=0.9, label="Instantaneous speed [mm/s]")
    positive = ~mask & np.isfinite(dissipation) & (dissipation > 0)
    loss_axis = axes[2]
    if np.any(positive):
        values = dissipation[positive]
        low, high = float(np.min(values)), float(np.max(values))
        if high == low:
            low, high = low / 2, high * 2
        image = loss_axis.pcolormesh(
            y, z, np.ma.array(dissipation, mask=~positive), cmap="magma",
            norm=LogNorm(vmin=low, vmax=high), shading="auto", rasterized=True,
        )
        figure.colorbar(image, ax=loss_axis, location="bottom", pad=0.05, shrink=0.9, label="Cycle-averaged dissipation [W/m³]")
    else:
        loss_axis.text(0.5, 0.5, "No positive finite dissipation samples", transform=loss_axis.transAxes, ha="center", va="center")
    loss_axis.set_title("Mean viscous dissipation")
    edges = np.asarray(arrays["plate_edges"], dtype=float) * 1e6
    if edges.shape != (2,) or not np.isfinite(edges).all() or edges[1] <= edges[0]:
        raise ValueError("Saved field must identify its two physical plate edges.")
    for axis in axes:
        (surface,) = axis.plot(edges, [0, 0], color="#17232b", lw=2.2, solid_capstyle="butt", zorder=5)
        surface.set_path_effects([path_effects.Stroke(linewidth=4, foreground="white"), path_effects.Normal()])
        axis.set(xlabel="Transverse y [µm]", ylabel="Normal z [µm]", xlim=(y[0], y[-1]), ylim=(z[0], z[-1]))
        axis.set_aspect("equal", adjustable="box")
    frequency = float(arrays["frequency_Hz"])
    section_x = float(arrays["section_x_m"])
    figure.suptitle(
        f"Resonance {field_record['resonance']} · {frequency / 1e3:.3f} kHz · section x = {section_x * 1e6:.1f} µm\n"
        "Local 2D fluid approximation · surface marked at z = 0",
        fontsize=13,
    )
    figure.supxlabel(
        "Finite view of unbounded fluid · quarter-cycle phase = in-phase + π/2 · field loss belongs to the 2D approximation",
        fontsize=8, color=GRAY,
    )
    return _save(figure, output, f"flow_peak_{int(field_record['resonance']):02d}")


def _linked_report(directory, output, report, rows, names):
    """Save a concise table with definitions, units and directly linked figures."""
    lines = [
        "# Resonator showcase figures", "", _title(report).replace("\n", " · ") + ".", "",
        "All spectra use computed complex displacement samples. Local lines stay inside resolved resonance windows. "
        "The fitted undamped frequency f₀ and the amplitude maximum are reported separately.", "",
        "| Resonance | Fitted f₀ [kHz] | Amplitude peak [kHz] | SHO Q | Energy Q | Fit residual | Work discrepancy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['resonance']} | {row['fitted_f0_Hz'] / 1e3:.4f} | {row['amplitude_peak_Hz'] / 1e3:.4f} "
            f"| {row['Q_SHO']:.4f} | {row['Q_energy']:.4f} | {row['fit_error']:.2%} | {row['work_balance_error']:.2e} |"
        )
    lines.extend([
        "", "Energy Q uses the maximum stored structural energy and the response model's fluid work per cycle. "
        "The plotted fluid field is an independent local 2D reconstruction of the structural motion. "
        "Its cycle-averaged dissipation [W/m³] is not substituted into a 3D or Sader Q calculation.", "",
        "Flow arrows show real velocity at phases −arg(v_tip) and −arg(v_tip) + π/2. "
        "The latter is the negative imaginary component in the tip-velocity reference. "
        "Black lines mark the infinitesimally thin resonator. Singular/nonfinite samples and nonpositive logarithmic densities are masked.", "",
    ])
    for name in names:
        label = name.replace("_", " ").capitalize()
        lines.append(f"- {label}: [PNG]({name}.png) · [PDF]({name}.pdf)")
    lines.extend(["", f"Saved source data: `{directory.resolve()}`.", ""])
    (output / "plots.md").write_text("\n".join(lines), encoding="utf-8")


def plot_showcase(results_directory, output_directory=None):
    """Regenerate all figures from a showcase directory without any FEM imports."""
    directory = Path(results_directory)
    output = Path(output_directory) if output_directory is not None else directory / "figures"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    rows = sorted(report["results"], key=lambda row: row["resonance"])
    if not rows:
        raise ValueError("A showcase needs at least one resolved resonance.")
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        names = plot_spectra(directory, output, report, rows)
        names.append(plot_resonance_q(output, report, rows))
        for field in report.get("fields", []):
            names.append(plot_field(directory, output, report, field))
    _linked_report(directory, output, report, rows, names)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Saved showcase directory.")
    parser.add_argument("--output", type=Path, help="Figure directory; default RESULTS/figures.")
    args = parser.parse_args(argv)
    output = plot_showcase(args.results, args.output)
    print(f"Saved PNG/PDF figures and plots.md to {output.resolve()}")


if __name__ == "__main__":
    main()
