"""Prescribed first-mode pressure and weighted polynomial diagnostics.

Run from anywhere with NumPy, SciPy and Matplotlib; production code is read only.
    python research/pressure_polynomial.py --quick
    python research/pressure_polynomial.py

All pressure fields are divided by a real tip-velocity amplitude (SI units).
Polynomial basis functions are averaged over the actual rectangular panels.
This is a postprocessing experiment, not a new weighted Stokeslet discretization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mufsi.hydrodynamics.legacy.stokes_3d_analytic import (
    Stokes3DAnalytic,
    analytic_fluid_grid,
)
from mufsi.models.fluid import Fluid
from mufsi.models.geometry import PlateGeometry

FAMILIES = ("ordinary", "transverse", "both")
LABELS = {
    "ordinary": "Ordinary polynomial",
    "transverse": "Transverse edge weight",
    "both": "Both edge weights",
}
CORE_FILES = (
    "hydrodynamics/legacy/stokes_3d_analytic.py",
    "hydrodynamics/legacy/panel_analytic.py",
    "hydrodynamics/legacy/grid_analytic.py",
    "hydrodynamics/stokeslet.py",
)


def core_hashes():
    return {
        name: hashlib.sha256((ROOT / "src/mufsi" / name).read_bytes()).hexdigest()
        for name in CORE_FILES
    }


def first_mode(x, length):
    """Exact clamped/free first EB shape, with unit tip displacement/velocity."""
    beta = 1.875104068711961
    sigma = (np.cosh(beta) + np.cos(beta)) / (np.sinh(beta) + np.sin(beta))
    z = beta * np.asarray(x) / length
    value = np.cosh(z) - np.cos(z) - sigma * (np.sinh(z) - np.sin(z))
    tip = np.cosh(beta) - np.cos(beta) - sigma * (np.sinh(beta) - np.sin(beta))
    return value / tip


def panel_basis(edges, degree, weighted):
    """Exact means of T_n(t), or T_n(t)/sqrt(1-t^2), on normalized panels."""
    edges = np.asarray(edges)
    if np.any(np.diff(edges) <= 0) or np.any(np.abs(edges) > 1 + 1e-12):
        raise ValueError("Basis panel edges must increase within [-1,1].")
    edges = np.clip(edges, -1, 1)
    widths = np.diff(edges)
    result = np.empty((len(widths), degree + 1))
    if weighted:
        angles = np.arccos(edges)
        result[:, 0] = (angles[:-1] - angles[1:]) / widths
        for n in range(1, degree + 1):
            # cos(n*theta) integrated over the reversed arccos limits.
            result[:, n] = (np.sin(n * angles[:-1]) - np.sin(n * angles[1:])) / (
                n * widths
            )
    else:
        for n in range(degree + 1):
            unit = np.zeros(n + 1)
            unit[n] = 1
            primitive = cheb.chebint(unit)
            result[:, n] = np.diff(cheb.chebval(edges, primitive)) / widths
    return result


def design(grid, geometry, family, m, n):
    xe = 2 * grid.x_panel_edges / geometry.length - 1
    ye = 2 * grid.panel_edges / geometry.width
    bx = panel_basis(xe, m, family == "both")
    by = panel_basis(ye, n, family != "ordinary")[:, ::2]
    return np.einsum("im,jn->ijmn", bx, by).reshape(grid.nx * grid.ny, -1)


def point_fit(points, geometry, fit, remainder=False):
    """Continuous polynomial reconstruction, strictly inside the sheet."""
    points = np.asarray(points)
    xi, eta = 2 * points[:, 0] / geometry.length - 1, 2 * points[:, 1] / geometry.width
    if np.any(np.abs(xi) >= 1) or np.any(np.abs(eta) >= 1):
        raise ValueError("Pointwise pressure is only evaluated inside the sheet.")
    bx = cheb.chebvander(xi, fit["m"])
    by = cheb.chebvander(eta, fit["n"])[:, ::2]
    value = np.einsum("im,mn,in->i", bx, fit["coefficients"], by)
    if remainder:
        return value
    if fit["family"] != "ordinary":
        value /= np.sqrt(1 - eta**2)
    if fit["family"] == "both":
        value /= np.sqrt(1 - xi**2)
    return value


def areas(grid):
    return np.outer(np.diff(grid.x_panel_edges), np.diff(grid.panel_edges)).ravel()


def smoother_weight(grid, geometry):
    xi = 2 * grid.points[:, 0] / geometry.length - 1
    eta = 2 * grid.points[:, 1] / geometry.width
    return np.sqrt((1 - xi**2) * (1 - eta**2))


def norm_error(value, reference, weight, mask=None):
    if mask is not None:
        value, reference, weight = value[mask], reference[mask], weight[mask]
    denominator = np.sum(weight * np.abs(reference) ** 2)
    if denominator == 0 or len(reference) == 0:
        return None
    return float(np.sqrt(np.sum(weight * np.abs(value - reference) ** 2) / denominator))


def relative_complex(value, reference):
    return float(abs(value - reference) / max(abs(reference), np.finfo(float).tiny))


def forces(grid, pressure, geometry):
    field = pressure.reshape(grid.nx, grid.ny)
    q = field @ np.diff(grid.panel_edges)
    force = np.diff(grid.x_panel_edges) @ q
    modal = np.sum(
        areas(grid) * first_mode(grid.points[:, 0], geometry.length) * pressure
    )
    return q, force, modal


def corner_mask(grid, geometry, fraction):
    x, y = grid.points.T
    return (np.minimum(x, geometry.length - x) < fraction * geometry.width) & (
        geometry.width / 2 - np.abs(y) < fraction * geometry.width
    )


def fit_pressure(grid, geometry, pressure, family, m, n, scope, fraction):
    matrix = design(grid, geometry, family, m, n)
    ix, iy = np.indices((grid.nx, grid.ny))
    holdout = ((ix + 2 * iy) % 5 == 0).ravel()
    corners = corner_mask(grid, geometry, fraction)
    domain = np.ones(len(pressure), dtype=bool)
    if scope == "omit_corners":
        domain &= ~corners
    train = domain & ~holdout
    # All families use the SAME weighted objective. No singular raw L2 claim.
    weight = areas(grid) * smoother_weight(grid, geometry) ** 2
    scale = np.sqrt(weight[train] / np.mean(weight[train]))
    if train.sum() < matrix.shape[1]:
        raise ValueError("Too few training panels for the requested polynomial order.")
    coefficients, _, rank, singular = np.linalg.lstsq(
        scale[:, None] * matrix[train],
        scale * pressure[train],
        rcond=None,
    )
    if rank != matrix.shape[1]:
        raise ValueError("Rank-deficient polynomial fit; reduce the orders or refine.")
    prediction = matrix @ coefficients
    row = {
        "family": family,
        "scope": scope,
        "m": m,
        "n": n,
        "coefficients_count": matrix.shape[1],
        "training_panels": int(train.sum()),
        "rank": int(rank),
        "condition": float(singular[0] / singular[-1]),
        "weighted_error_all": norm_error(prediction, pressure, weight),
        "weighted_error_real": norm_error(prediction.real, pressure.real, weight),
        "weighted_error_imag": norm_error(prediction.imag, pressure.imag, weight),
        "weighted_error_interior": norm_error(prediction, pressure, weight, ~corners),
        "weighted_error_corners": norm_error(prediction, pressure, weight, corners),
        "weighted_error_holdout": norm_error(
            prediction, pressure, weight, domain & holdout
        ),
        "area_l1_error": float(
            np.sum(areas(grid) * abs(prediction - pressure))
            / np.sum(areas(grid) * abs(pressure))
        ),
    }
    return {
        **row,
        "coefficients": coefficients.reshape(m + 1, n // 2 + 1),
        "prediction": prediction,
    }


def overlap_average(target_edges, source_edges):
    overlap = np.maximum(
        0,
        np.minimum(target_edges[1:, None], source_edges[None, 1:])
        - np.maximum(target_edges[:-1, None], source_edges[None, :-1]),
    )
    return overlap / np.diff(target_edges)[:, None]


def restrict_pressure(reference, target):
    gx, gy = reference["grid"], target["grid"]
    rx = overlap_average(gy.x_panel_edges, gx.x_panel_edges)
    ry = overlap_average(gy.panel_edges, gx.panel_edges)
    return (rx @ reference["pressure"].reshape(gx.nx, gx.ny) @ ry.T).ravel()


def solve_field(geometry, fluid, nx, ny, omega, tolerance):
    grid = analytic_fluid_grid(geometry, nx=nx, ny=ny)
    model = Stokes3DAnalytic(
        fluid,
        grid,
        tolerance=tolerance,
        absolute_tolerance=1e-18,
        max_refinements=7,
    )
    velocity = first_mode(grid.points[:, 0], geometry.length).astype(complex)
    start = perf_counter()
    pressure = model.pressure_from_velocity(omega, velocity)
    residual = norm_error(model.apply_mobility(omega, pressure), velocity, areas(grid))
    if not np.isfinite(pressure).all() or residual > 1e-9:
        raise RuntimeError(f"Invalid reference pressure: no-slip residual {residual}.")
    q, force, modal = forces(grid, pressure, geometry)
    summary = {
        "nx": nx,
        "ny": ny,
        "solve_seconds": perf_counter() - start,
        "reference_no_slip": residual,
        "pressure_y_symmetry": float(
            np.linalg.norm(pressure.reshape(nx, ny) - pressure.reshape(nx, ny)[:, ::-1])
            / np.linalg.norm(pressure)
        ),
        "force_real": float(force.real),
        "force_imag": float(force.imag),
        "modal_real": float(modal.real),
        "modal_imag": float(modal.imag),
        "integration": asdict(model.integration_report),
    }
    return {
        "grid": grid,
        "model": model,
        "pressure": pressure,
        "velocity": velocity,
        "q": q,
        "force": force,
        "modal": modal,
        "summary": summary,
    }


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def corner_probes(fields, geometry):
    """Bilinear panel-value probes inside the node hull, without extrapolation."""
    rows = []
    for field in fields:
        grid = field["grid"]
        interp = RegularGridInterpolator(
            (grid.x, grid.y),
            field["pressure"].reshape(grid.nx, grid.ny),
            bounds_error=True,
        )
        for end in ("root", "tip"):
            dx_min = grid.x[0] if end == "root" else geometry.length - grid.x[-1]
            dy_min = geometry.width / 2 - grid.y[-1]
            for ratio in (0.25, 0.5, 1.0):
                low = max(dx_min, dy_min / ratio) * (1 + 1e-10)
                high = min(2 * geometry.width, 0.9 * geometry.width / (2 * ratio))
                if high <= 1.2 * low:
                    continue
                dx = np.geomspace(low, high, 50)
                dy = ratio * dx
                x = dx if end == "root" else geometry.length - dx
                value = interp(np.column_stack((x, geometry.width / 2 - dy)))
                radius = np.hypot(dx, dy)
                exponent = -np.gradient(
                    np.log(np.maximum(abs(value), 1e-300)), np.log(radius)
                )
                for r, px, py, p, alpha in zip(radius, dx, dy, value, exponent):
                    rows.append(
                        {
                            "nx": grid.nx,
                            "ny": grid.ny,
                            "end": end,
                            "dy_over_dx": ratio,
                            "radius_over_width": float(r / geometry.width),
                            "dx_over_width": float(px / geometry.width),
                            "dy_over_width": float(py / geometry.width),
                            "pressure_real": float(p.real),
                            "pressure_imag": float(p.imag),
                            "pressure_abs": float(abs(p)),
                            "finite_resolution_exponent": float(alpha),
                        }
                    )
    return rows


def make_plots(output, fields, fits, geometry, frequency, probes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2})
    reference = fields[-1]
    grid, p = (
        reference["grid"],
        reference["pressure"].reshape(
            reference["grid"].nx,
            reference["grid"].ny,
        ),
    )
    colors = dict(zip(FAMILIES, ("#c66d25", "#2278b0", "#278556")))
    title = f"First EB mode, real unit tip velocity; f = {frequency:.3g} Hz"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, values, name in zip(
        axes, (p.real, p.imag, abs(p)), ("Real", "Imaginary", "Magnitude")
    ):
        if name == "Magnitude":
            mesh = ax.pcolormesh(
                grid.x_panel_edges * 1e6,
                grid.panel_edges * 1e6,
                values.T,
                shading="flat",
            )
        else:
            bound = np.max(abs(values))
            mesh = ax.pcolormesh(
                grid.x_panel_edges * 1e6,
                grid.panel_edges * 1e6,
                values.T,
                shading="flat",
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
            )
        ax.set(title=name, xlabel="Length x [µm]", ylabel="Width y [µm]")
        fig.colorbar(mesh, ax=ax, label="Pressure / tip velocity [Pa s/m]")
    fig.suptitle(title)
    fig.savefig(output / "pressure_field.png", dpi=170)
    plt.close(fig)

    # Select an equal-order comparison (highest requested order, full domain).
    best_order = max((f["m"], f["n"]) for f in fits)
    selected = [
        f for f in fits if (f["m"], f["n"]) == best_order and f["scope"] == "all"
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
    for column, position in enumerate((0.25, 0.6, 0.9, 1.0)):
        ix = np.argmin(abs(grid.x / geometry.length - position))
        points = np.column_stack((np.full(grid.ny, grid.x[ix]), grid.y))
        for row, component in enumerate((np.real, np.imag)):
            ax = axes[row, column]
            ax.plot(
                grid.y / (geometry.width / 2),
                component(p[ix]),
                "k.",
                label="Reference panel coefficients",
            )
            for fit in selected:
                ax.plot(
                    grid.y / (geometry.width / 2),
                    component(point_fit(points, geometry, fit)),
                    color=colors[fit["family"]],
                    label=LABELS[fit["family"]] + " (continuous)",
                )
                ax.plot(
                    grid.y / (geometry.width / 2),
                    component(fit["prediction"].reshape(grid.nx, grid.ny)[ix]),
                    "+",
                    color=colors[fit["family"]],
                    markersize=3,
                )
            ax.set(
                title=f"x/L = {grid.x[ix] / geometry.length:.3f}",
                xlabel="y / half-width",
            )
            if column == 0:
                ax.set_ylabel(("Real" if row == 0 else "Imaginary") + " p/V [Pa s/m]")
            if row == 0 and column == 0:
                ax.legend(fontsize=7)
    fig.suptitle(title + f"; orders {best_order}; + = fitted panel averages")
    fig.savefig(output / "transverse_fits.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for column, family in enumerate(FAMILIES):
        group = sorted(
            (f for f in fits if f["family"] == family and f["scope"] == "all"),
            key=lambda f: f["coefficients_count"],
        )
        for row, component in enumerate((np.real, np.imag)):
            ax = axes[row, column]
            ax.plot(
                grid.y / (geometry.width / 2),
                component(p[-1]),
                "k.",
                label="Reference panels",
                zorder=10,
            )
            for fit in group:
                values = fit["prediction"].reshape(grid.nx, grid.ny)[-1]
                ax.plot(
                    grid.y / (geometry.width / 2),
                    component(values),
                    label=f"M={fit['m']}, N={fit['n']} ({fit['coefficients_count']} coeff.)",
                )
            ax.set(
                title=LABELS[family],
                xlabel="y / half-width",
                ylabel=("Real" if row == 0 else "Imaginary") + " p/V [Pa s/m]",
            )
            if row == 0:
                ax.legend(fontsize=7)
    fig.suptitle(
        f"Polynomial order comparison near tip, x/L={grid.x[-1] / geometry.length:.4f}; panel means"
    )
    fig.savefig(output / "order_fits.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for column, eta_target in enumerate((0, 0.9)):
        iy = np.argmin(abs(grid.y / (geometry.width / 2) - eta_target))
        points = np.column_stack((grid.x, np.full(grid.nx, grid.y[iy])))
        for row, component in enumerate((np.real, np.imag)):
            ax = axes[row, column]
            ax.plot(
                grid.x / geometry.length,
                component(p[:, iy]),
                "k.",
                label="Reference panels",
            )
            for fit in selected:
                ax.plot(
                    grid.x / geometry.length,
                    component(point_fit(points, geometry, fit)),
                    color=colors[fit["family"]],
                    label=LABELS[fit["family"]],
                )
            ax.set(
                title=f"y / half-width = {grid.y[iy] / (geometry.width / 2):.3f}",
                xlabel="x/L",
                ylabel=("Real" if row == 0 else "Imaginary") + " p/V [Pa s/m]",
            )
            if row == 0 and column == 0:
                ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(output / "longitudinal_fits.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    xi = 2 * grid.points[:, 0] / geometry.length - 1
    eta = 2 * grid.points[:, 1] / geometry.width
    for column, family in enumerate(FAMILIES):
        weight = np.ones_like(xi)
        if family != "ordinary":
            weight *= np.sqrt(1 - eta**2)
        if family == "both":
            weight *= np.sqrt(1 - xi**2)
        g = (weight * reference["pressure"]).reshape(grid.nx, grid.ny)
        for row, component in enumerate((np.real, np.imag)):
            ax = axes[row, column]
            values = component(g)
            bound = np.max(abs(values))
            mesh = ax.pcolormesh(
                grid.x_panel_edges / geometry.length,
                grid.panel_edges / (geometry.width / 2),
                values.T,
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
                shading="flat",
            )
            ax.set(
                title=LABELS[family] + (": real" if row == 0 else ": imaginary"),
                xlabel="x/L",
                ylabel="y / half-width",
            )
            fig.colorbar(mesh, ax=ax)
    fig.suptitle(
        "Weighted panel proxies (node weight × panel coefficient), not exact pointwise remainders"
    )
    fig.savefig(output / "weighted_remainders.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    metrics = (
        ("weighted_error_holdout", "Weighted held-out pressure error"),
        ("no_slip_error", "Prescribed velocity error"),
        ("modal_force_error", "Modal force error"),
    )
    for row, scope in enumerate(("all", "omit_corners")):
        for column, (metric, name) in enumerate(metrics):
            ax = axes[row, column]
            for family in FAMILIES:
                group = sorted(
                    (f for f in fits if f["family"] == family and f["scope"] == scope),
                    key=lambda f: f["coefficients_count"],
                )
                ax.loglog(
                    [f["coefficients_count"] for f in group],
                    [max(f[metric], 1e-16) for f in group],
                    "o-",
                    color=colors[family],
                    label=LABELS[family],
                )
            ax.set(
                title=name + "\n" + scope.replace("_", " "),
                xlabel="Complex coefficients",
                ylabel="Relative error",
            )
            if column == 0:
                ax.legend(fontsize=8)
    fig.suptitle(
        title + "; every family uses the same weighted least-squares objective"
    )
    fig.savefig(output / "fit_convergence.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for field in fields:
        g = field["grid"]
        for column, target in enumerate((0.6, 1.0)):
            ix = np.argmin(abs(g.x / geometry.length - target))
            weighted = (
                np.sqrt(1 - (g.y / (geometry.width / 2)) ** 2)
                * field["pressure"].reshape(g.nx, g.ny)[ix]
            )
            for row, component in enumerate((np.real, np.imag)):
                axes[row, column].plot(
                    g.y / (geometry.width / 2),
                    component(weighted),
                    ".-",
                    label=f"{g.nx}×{g.ny}; x/L={g.x[ix] / geometry.length:.3f}",
                )
                axes[row, column].set(
                    title="Mid-beam" if column == 0 else "Last longitudinal section",
                    xlabel="y / half-width",
                    ylabel=("Real" if row == 0 else "Imaginary")
                    + " weighted panel proxy",
                )
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(
        "Grid refinement; last-section positions change with nx (not a fixed-location comparison)"
    )
    fig.savefig(output / "grid_refinement.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, end in enumerate(("root", "tip")):
        for index, field in enumerate(fields):
            g = field["grid"]
            for ratio, linestyle in zip((0.25, 0.5, 1.0), ("-", "--", ":")):
                group = [
                    r
                    for r in probes
                    if r["nx"] == g.nx
                    and r["ny"] == g.ny
                    and r["end"] == end
                    and r["dy_over_dx"] == ratio
                ]
                if not group:
                    continue
                label = f"{g.nx}×{g.ny}, dy/dx={ratio}"
                color = f"C{index}"
                axes[0, column].loglog(
                    [r["radius_over_width"] for r in group],
                    [r["pressure_abs"] for r in group],
                    linestyle,
                    color=color,
                    label=label,
                )
                axes[1, column].semilogx(
                    [r["radius_over_width"] for r in group],
                    [r["finite_resolution_exponent"] for r in group],
                    linestyle,
                    color=color,
                )
        axes[0, column].set(
            title=end.title() + " corner: magnitude",
            xlabel="r/W",
            ylabel="|p/V| [Pa s/m]",
        )
        axes[0, column].legend(fontsize=6, ncol=2)
        axes[1, column].axhline(
            0.5, color="gray", linestyle="--", label="Single-edge candidate 0.5"
        )
        axes[1, column].axhline(
            1.0, color="gray", linestyle=":", label="Product-weight candidate 1"
        )
        axes[1, column].set(xlabel="r/W", ylabel="−d log|p| / d log r", ylim=(-1, 2))
        axes[1, column].legend(fontsize=7)
    fig.suptitle(
        "Exploratory corner rays: interpolated panel data within node hull; no asymptotic exponent certified"
    )
    fig.savefig(output / "corner_diagnostics.png", dpi=170)
    plt.close(fig)


def run_frequency(args, geometry, fluid, frequency, output):
    output.mkdir(parents=True, exist_ok=True)
    omega = 2 * np.pi * frequency
    fields = []
    for nx, ny in args.grids:
        print(f"Solving {nx}x{ny} at {frequency:.6g} Hz...", flush=True)
        if fields:
            fields[-1]["model"].clear_cache()
        field = solve_field(geometry, fluid, nx, ny, omega, args.tolerance)
        g = field["grid"]
        np.savez_compressed(
            output / f"pressure_{nx}x{ny}.npz",
            x=g.x,
            y=g.y,
            x_edges=g.x_panel_edges,
            y_edges=g.panel_edges,
            pressure=field["pressure"].reshape(nx, ny),
            velocity=field["velocity"].reshape(nx, ny),
            line_load=field["q"],
            physical_panel_areas=areas(g).reshape(nx, ny),
        )
        # Keep only the final model's dense mobility and LU, avoiding cumulative memory.
        fields.append(field)
    reference = fields[-1]
    grid = reference["grid"]
    fits, rows = [], []
    for family in FAMILIES:
        for scope in ("all", "omit_corners"):
            for m, n in args.orders:
                fit = fit_pressure(
                    grid,
                    geometry,
                    reference["pressure"],
                    family,
                    m,
                    n,
                    scope,
                    args.corner_fraction,
                )
                prediction = fit["prediction"]
                fitted_velocity = reference["model"].apply_mobility(omega, prediction)
                q, force, modal = forces(grid, prediction, geometry)
                fit.update(
                    {
                        "no_slip_error": norm_error(
                            fitted_velocity, reference["velocity"], areas(grid)
                        ),
                        "line_load_error": norm_error(
                            q, reference["q"], np.diff(grid.x_panel_edges)
                        ),
                        "net_force_error": relative_complex(force, reference["force"]),
                        "modal_force_error": relative_complex(
                            modal, reference["modal"]
                        ),
                    }
                )
                rows.append(
                    {
                        k: v
                        for k, v in fit.items()
                        if k not in ("coefficients", "prediction")
                    }
                )
                fits.append(fit)
                np.savez_compressed(
                    output / f"fit_{family}_{scope}_{m}x{n}.npz",
                    coefficients=fit["coefficients"],
                    pressure_panels=prediction.reshape(grid.nx, grid.ny),
                )
    write_csv(output / "fit_metrics.csv", rows)
    grid_rows = []
    for field in fields:
        g = field["grid"]
        restricted = restrict_pressure(reference, field)
        rq, _, _ = forces(g, restricted, geometry)
        grid_rows.append(
            {
                **{k: v for k, v in field["summary"].items() if k != "integration"},
                "weighted_pressure_difference_to_finest": norm_error(
                    field["pressure"],
                    restricted,
                    areas(g) * smoother_weight(g, geometry) ** 2,
                ),
                "line_load_difference_to_finest": norm_error(
                    field["q"], rq, np.diff(g.x_panel_edges)
                ),
                "net_force_difference_to_finest": relative_complex(
                    field["force"], reference["force"]
                ),
                "modal_force_difference_to_finest": relative_complex(
                    field["modal"], reference["modal"]
                ),
            }
        )
    write_csv(output / "grid_metrics.csv", grid_rows)
    probes = corner_probes(fields, geometry)
    write_csv(output / "corner_probes.csv", probes)
    make_plots(output, fields, fits, geometry, frequency, probes)
    best = min(rows, key=lambda row: row["no_slip_error"])
    # Numerical panel integration sensitivity is distinct from spatial refinement.
    tight = solve_field(geometry, fluid, *args.grids[0], omega, args.tolerance / 10)
    baseline = fields[0]
    integration_check = {
        "grid": list(args.grids[0]),
        "tighter_tolerance": args.tolerance / 10,
        "weighted_pressure_change": norm_error(
            tight["pressure"],
            baseline["pressure"],
            areas(baseline["grid"]) * smoother_weight(baseline["grid"], geometry) ** 2,
        ),
        "modal_force_change": relative_complex(tight["modal"], baseline["modal"]),
        "integration": tight["summary"]["integration"],
    }
    tight["model"].clear_cache()
    result = {
        "frequency_hz": frequency,
        "omega_b_squared_over_nu": omega
        * (geometry.width / 2) ** 2
        / fluid.kinematic_viscosity,
        "reference_grid": list(args.grids[-1]),
        "grids": grid_rows,
        "integration_reports": [f["summary"]["integration"] for f in fields],
        "integration_sensitivity": integration_check,
        "fits": rows,
        "lowest_velocity_error_fit": best,
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    table = "\n".join(
        f"| {r['family']} | {r['scope']} | {r['m']}×{r['n']} | {r['coefficients_count']} | "
        f"{r['weighted_error_holdout']:.2%} | {r['no_slip_error']:.2%} | {r['modal_force_error']:.2%} |"
        for r in rows
    )
    summary = f"""# First-mode pressure polynomial experiment

Frequency: {frequency:.6g} Hz. L/W = {geometry.length / geometry.width:g}.
Reference grid: {grid.nx}×{grid.ny}. Real unit tip velocity; p/V in Pa s/m.
Euler–Bernoulli first mode is prescribed, independent of resonance.

## What this run establishes

The lowest prescribed-velocity error among the tested fits is
**{best["family"]}, {best["scope"]}, orders {best["m"]}×{best["n"]}**, with
{best["coefficients_count"]} complex coefficients: velocity error
{best["no_slip_error"]:.2%}, held-out weighted pressure error
{best["weighted_error_holdout"]:.2%}, modal-force error {best["modal_force_error"]:.2%}.
This is a finite-grid ranking, not a generally validated basis recommendation.
The maximum reference no-slip residual is {max(r["reference_no_slip"] for r in grid_rows):.3g}.
Tightening panel integration tolerance by 10 changes the base-grid weighted
pressure by {integration_check["weighted_pressure_change"]:.3g}.

Grid comparisons use exact area-overlap restriction of the finest panel field.
Compare separate x and y refinements in `grid_metrics.csv`; the finest grid is
a comparison baseline, not an exact pressure solution. Corner-ray slopes are
finite-resolution diagnostics of interpolated panel coefficients. Uniform x
resolution limits the approach to the corner. No corner exponent is certified.

## Fits

Every model uses exact panel means of its basis and the same weighted objective:
sum(area × (1−xi²) × (1−eta²) × |p_fit−p_reference|²).
20% of panels are held out from fitting. `omit_corners` additionally excludes
four corner squares of side {args.corner_fraction:g}W. Its held-out error is
evaluated only in that retained domain; velocity/force errors always use the
whole sheet. All expansions use even transverse Chebyshev terms.

| Family | Fit domain | Orders M×N | Coefficients | Held-out weighted pressure | Velocity | Modal force |
| --- | --- | --- | ---: | ---: | ---: | ---: |
{table}

## Plots

![Real, imaginary and magnitude pressure](pressure_field.png)
![Fits across the width](transverse_fits.png)
![Different polynomial orders near the tip](order_fits.png)
![Fits along the beam](longitudinal_fits.png)
![Weighted panel proxies](weighted_remainders.png)
![Fit errors versus coefficient count](fit_convergence.png)
![Grid refinement](grid_refinement.png)
![Corner rays and exploratory slopes](corner_diagnostics.png)

Physical panel areas are used for all force projections, rather than the
legacy ordinary-integral Chebyshev weights. q = integral p dy is resisting
load per tip velocity; the actual load on the beam has the opposite sign.
Modal resistance = integral phi_1 p dA for real unit tip velocity.
Continuous fitted curves and fitted panel means differ near singular edges;
the transverse plots show both. Weighted remainder images are panel proxies.
See `report.json`, CSV tables and NPZ arrays for all numerical data.
"""
    (output / "report.md").write_text(summary, encoding="utf-8")
    for field in fields:
        field["model"].clear_cache()
    return result


def pair(value):
    try:
        a, b = map(int, value.lower().split("x"))
        return a, b
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected two integers like 48x24.") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small diagnostic run, not corner convergence.",
    )
    parser.add_argument("--length", type=float, default=500e-6)
    parser.add_argument("--width", type=float, default=25e-6)
    parser.add_argument("--density", type=float, default=997.0)
    parser.add_argument("--viscosity", type=float, default=890e-6)
    parser.add_argument(
        "--frequencies", type=float, nargs="+", help="Hz; default omega*b^2/nu=1."
    )
    parser.add_argument(
        "--grids",
        type=pair,
        nargs="+",
        help="Finest grid must be last and dominate the others.",
    )
    parser.add_argument(
        "--orders",
        type=pair,
        nargs="+",
        default=[(2, 2), (4, 4), (8, 6), (12, 8), (16, 10)],
    )
    parser.add_argument("--tolerance", type=float, default=2e-6)
    parser.add_argument("--corner-fraction", type=float, default=0.25)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/results/pressure_polynomial"
    )
    args = parser.parse_args()
    if not all(
        np.isfinite(v) and v > 0
        for v in (args.length, args.width, args.density, args.viscosity, args.tolerance)
    ):
        parser.error(
            "Dimensions, fluid properties and tolerance must be finite and positive."
        )
    if not 0 < args.corner_fraction < 0.5:
        parser.error("Corner fraction must be between zero and 0.5.")
    args.grids = args.grids or (
        [(24, 12), (48, 12), (24, 24), (48, 24)]
        if args.quick
        else [(48, 24), (96, 24), (48, 48), (96, 48)]
    )
    if len(set(args.grids)) != len(args.grids) or any(min(g) < 3 for g in args.grids):
        parser.error(
            "Grids must be distinct and have at least three panels in each direction."
        )
    if any(nx > args.grids[-1][0] or ny > args.grids[-1][1] for nx, ny in args.grids):
        parser.error(
            "The final reference grid must dominate every earlier grid in both counts."
        )
    if any(
        m < 0 or n < 0 or n % 2 or m >= args.grids[-1][0] or n >= args.grids[-1][1]
        for m, n in args.orders
    ):
        parser.error(
            "Orders must be nonnegative, N even, and below the reference grid counts."
        )
    geometry = PlateGeometry(args.length, args.width, args.width / 100)
    fluid = Fluid(args.density, args.viscosity)
    frequencies = args.frequencies or [
        fluid.kinematic_viscosity / (2 * np.pi * (args.width / 2) ** 2)
    ]
    if any(not np.isfinite(f) or f <= 0 for f in frequencies) or len(
        set(frequencies)
    ) != len(frequencies):
        parser.error("Frequencies must be distinct, finite and positive.")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = core_hashes()
    for index, frequency in enumerate(frequencies):
        folder = args.output / f"frequency_{index + 1:02d}"
        run_frequency(args, geometry, fluid, frequency, folder)
    after = core_hashes()
    if hashes != after:
        raise RuntimeError(
            "Production Stokeslet source changed during this experiment."
        )
    import matplotlib
    import scipy

    manifest = {
        "geometry": asdict(geometry),
        "fluid": asdict(fluid),
        "grids": args.grids,
        "orders": args.orders,
        "tolerance": args.tolerance,
        "corner_fraction": args.corner_fraction,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "core_source_sha256": hashes,
        "core_source_unchanged": True,
        "experiment_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "frequencies_hz": frequencies,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    links = "\n".join(
        f"- [{f:.6g} Hz](frequency_{i + 1:02d}/report.md)"
        for i, f in enumerate(frequencies)
    )
    (args.output / "README.md").write_text(
        "# Pressure polynomial analysis results\n\n"
        + links
        + "\n\nProduction Stokeslet source hashes were unchanged. See manifest.json for settings.\n",
        encoding="utf-8",
    )
    print(f"Saved reports, plots and pressure arrays in {args.output}", flush=True)


if __name__ == "__main__":
    main()
