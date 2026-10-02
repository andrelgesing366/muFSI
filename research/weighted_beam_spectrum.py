"""Modal Euler-Bernoulli displacement spectrum with continuous weighted pressure.

Experimental clamped-free beam, exact dry EB modes, tip point-force excitation.
No DOLFINx required. Fluid couples ALL retained structural modes nonlocally.
exp(+i omega t): (K - omega² Mass + i omega Z) d = F; fluid load = -C a.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from weighted_pressure_mobility import solve_coefficients
from weighted_pressure_projection import (
    ROOT,
    add_arguments,
    build_model,
    modal_projection,
    source_hashes,
)


class CantileverModes:
    """Exact homogeneous clamped/free modes, unit tip value; dimensions in SI."""

    def __init__(self, geometry, young_modulus=169e9, density=2330, count=2):
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or not 1 <= count <= 6
        ):
            raise ValueError("Retain 1..6 dry modes in this experiment.")
        if (
            not np.isfinite([young_modulus, density]).all()
            or min(young_modulus, density) <= 0
        ):
            raise ValueError("Young modulus and solid density must be positive.")
        self.geometry, self.count = geometry, count
        self.EI = young_modulus * geometry.width * geometry.thickness**3 / 12
        self.line_mass = density * geometry.width * geometry.thickness
        roots = []
        for n in range(1, count + 1):
            center = (n - 0.5) * np.pi
            roots.append(
                brentq(lambda z: np.cos(z) + 1 / np.cosh(z), center - 0.5, center + 0.5)
            )
        self.beta = np.array(roots)
        self.sigma = (np.cosh(self.beta) + np.cos(self.beta)) / (
            np.sinh(self.beta) + np.sin(self.beta)
        )
        self.tip_values = self._raw(np.array([geometry.length]))[0]
        nodes, weights = leggauss(120)
        x = geometry.length * (nodes + 1) / 2
        shapes = self.values(x)
        self.mass = (
            self.line_mass
            * geometry.length
            / 2
            * np.sum(weights[:, None] * shapes**2, axis=0)
        )
        self.dry_omega = (
            self.beta**2 / geometry.length**2 * np.sqrt(self.EI / self.line_mass)
        )
        self.stiffness = self.mass * self.dry_omega**2

    def _raw(self, x):
        z = np.asarray(x)[:, None] * self.beta / self.geometry.length
        return np.cosh(z) - np.cos(z) - self.sigma * (np.sinh(z) - np.sin(z))

    def values(self, x):
        return self._raw(np.atleast_1d(x)) / self.tip_values


def solve_modal_response(beam, omega, impedance, force):
    """Impedance maps unit modal velocity to generalized resisting force."""
    Z = np.asarray(impedance, complex)
    force = np.asarray(force, complex)
    if Z.shape != (beam.count, beam.count) or force.shape != (beam.count,):
        raise ValueError("Impedance/load dimensions must match the mode count.")
    if (
        not np.isfinite(omega)
        or omega < 0
        or not np.isfinite(Z).all()
        or not np.isfinite(force).all()
    ):
        raise ValueError("Frequency, impedance and force must be finite.")
    D = np.diag(beam.stiffness - omega**2 * beam.mass) + 1j * omega * Z
    displacement = np.linalg.solve(D, force)
    residual = np.linalg.norm(D @ displacement - force) / max(
        np.linalg.norm(force), 1e-300
    )
    return displacement, float(residual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--modes", type=int, default=2)
    parser.add_argument("--young-modulus", type=float, default=169e9)
    parser.add_argument("--solid-density", type=float, default=2330)
    parser.add_argument("--tip-force", type=float, default=1e-9, help="N, real phase")
    parser.add_argument(
        "--frequencies", type=float, nargs="+", help="Hz, explicit list"
    )
    parser.add_argument(
        "--start", type=float, help="Hz; default 0.02*first dry frequency"
    )
    parser.add_argument(
        "--stop", type=float, help="Hz; default 1.2*first dry frequency"
    )
    parser.add_argument("--frequency-count", type=int, default=31)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/results/weighted_beam_spectrum"
    )
    args = parser.parse_args()
    solver = build_model(args)
    basis, geometry = solver.basis, solver.basis.geometry
    if not np.isfinite(args.tip_force) or args.tip_force == 0:
        parser.error("tip-force must be finite and nonzero.")
    if args.modes > basis.M + 1:
        parser.error(
            "The pressure x basis must have at least as many terms as beam modes."
        )
    beam = CantileverModes(geometry, args.young_modulus, args.solid_density, args.modes)
    first_hz = beam.dry_omega[0] / (2 * np.pi)
    if args.frequencies is None:
        start = first_hz * 0.02 if args.start is None else args.start
        stop = first_hz * 1.2 if args.stop is None else args.stop
        if args.frequency_count < 2 or start < 0 or stop <= start:
            parser.error("Use frequency-count>=2 and 0<=start<stop.")
        frequencies = np.linspace(start, stop, args.frequency_count)
    else:
        frequencies = np.asarray(args.frequencies)
    if (
        not np.isfinite(frequencies).all()
        or np.any(frequencies < 0)
        or np.any(np.diff(frequencies) <= 0)
    ):
        parser.error("Frequencies must be finite, nonnegative and strictly increasing.")
    before = source_hashes()
    points = basis.collocation(args.nx, args.ny_half)
    E = beam.values(points[:, 0])
    C = modal_projection(basis, beam.values)
    Cfine = modal_projection(basis, beam.values, order=160)
    projection_error = float(np.linalg.norm(Cfine - C) / np.linalg.norm(Cfine))
    nx = args.nx or 2 * (basis.M + 1)
    ny = args.ny_half or 2 * (basis.K + 1)
    check_points = basis.collocation(nx + 1, ny + 1)[::3]
    Echeck = beam.values(check_points[:, 0])
    force = args.tip_force * np.ones(beam.count)
    x = np.linspace(0, geometry.length, 301)
    shapes = beam.values(x)
    modal = np.empty((len(frequencies), beam.count), complex)
    dry = np.empty_like(modal)
    coefficients = np.empty((len(frequencies), basis.count), complex)
    impedance = np.empty((len(frequencies), beam.count, beam.count), complex)
    diagnostics = []
    saved_matrices = []
    started = perf_counter()
    for i, hz in enumerate(frequencies):
        omega = 2 * np.pi * hz
        H, integration = solver.assemble(omega, points)
        A, projection = solve_coefficients(H, E)
        Z = C @ A
        impedance[i] = Z
        d, equilibrium = solve_modal_response(beam, omega, Z, force)
        modal[i] = d
        coefficients[i] = 1j * omega * (A @ d)
        dry[i], _ = solve_modal_response(beam, omega, np.zeros_like(Z), force)
        Hcheck, _ = solver.assemble(omega, check_points)
        desired = 1j * omega * (Echeck @ d)
        actual = Hcheck @ coefficients[i]
        no_slip = (
            float(np.linalg.norm(actual - desired) / np.linalg.norm(desired))
            if omega > 0
            else 0.0
        )
        mode_errors = np.linalg.norm(Hcheck @ A - Echeck, axis=0) / np.linalg.norm(
            Echeck, axis=0
        )
        resistance = (Z + Z.conj().T) / 2
        minimum_resistance = float(np.linalg.eigvalsh(resistance).min())
        reciprocity = float(np.linalg.norm(Z - Z.T) / max(np.linalg.norm(Z), 1e-300))
        diagnostics.append(
            {
                "frequency_Hz": float(hz),
                "assembly_seconds": integration["seconds"],
                "max_quadrature_order": integration["max_order"],
                "quadrature_error_ratio": integration["max_estimated_error_ratio"],
                "rank": projection["rank"],
                "scaled_condition": projection["scaled_condition"],
                "modal_collocation_errors": projection["relative_residual"],
                "modal_heldout_errors": mode_errors.tolist(),
                "response_heldout_no_slip_error": no_slip,
                "relative_equilibrium_error": equilibrium,
                "relative_reciprocity_error": reciprocity,
                "minimum_resistance_eigenvalue_Ns_per_m": minimum_resistance,
            }
        )
        if i in {0, len(frequencies) // 2, len(frequencies) - 1}:
            saved_matrices.append((i, H))
        print(
            f"{i + 1}/{len(frequencies)}: {hz:.1f} Hz, "
            f"|tip|={abs(d.sum()):.4g} m, heldout no-slip={no_slip:.2%}",
            flush=True,
        )
    if before != source_hashes():
        raise RuntimeError("Production source changed during the experiment.")
    displacement = modal @ shapes.T
    tip = modal.sum(axis=1)
    peak = int(np.argmax(abs(tip)))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "spectrum.npz",
        frequencies=frequencies,
        x=x,
        displacement=displacement,
        modal_displacement=modal,
        dry_modal_displacement=dry,
        pressure_coefficients=coefficients,
        impedance=impedance,
        modal_projection=C,
        collocation=points,
        check_points=check_points,
        mass=beam.mass,
        stiffness=beam.stiffness,
        dry_frequencies=beam.dry_omega / (2 * np.pi),
        saved_H_indices=np.array([i for i, _ in saved_matrices]),
        saved_H=np.array([h for _, h in saved_matrices]),
    )
    with (args.output / "tip_spectrum.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["frequency_Hz", "tip_real_m", "tip_imag_m", "tip_abs_m", "tip_phase_rad"]
        )
        writer.writerows(zip(frequencies, tip.real, tip.imag, abs(tip), np.angle(tip)))
    report = {
        "configuration": vars(args) | {"output": str(args.output)},
        "coefficient_count": basis.count,
        "matrix_shape": [len(points), basis.count],
        "dry_frequencies_Hz": (beam.dry_omega / (2 * np.pi)).tolist(),
        "projection_quadrature_relative_difference": projection_error,
        "total_seconds": perf_counter() - started,
        "sampled_peak_frequency_Hz": float(frequencies[peak]),
        "sampled_peak_tip_m": float(abs(tip[peak])),
        "diagnostics": diagnostics,
        "source_hashes": before,
        "research_source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("weighted*.py")
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    maximum_no_slip = max(d["response_heldout_no_slip_error"] for d in diagnostics)
    maximum_equilibrium = max(d["relative_equilibrium_error"] for d in diagnostics)
    maximum_reciprocity = max(d["relative_reciprocity_error"] for d in diagnostics)
    minimum_resistance = min(
        d["minimum_resistance_eigenvalue_Ns_per_m"] for d in diagnostics
    )
    summary = (
        "# Weighted-pressure beam displacement spectrum\n\n"
        f"M={basis.M}, K={basis.K}; {basis.count} pressure coefficients; "
        f"{beam.count} exact dry cantilever modes; {len(frequencies)} frequencies.\n\n"
        f"Tip force: {args.tip_force:g} N. Time convention: exp(+i omega t).\n\n"
        f"- Total elapsed: {report['total_seconds']:.3f} s.\n"
        f"- First dry frequency: {first_hz:.6g} Hz.\n"
        f"- Largest sampled displacement: {abs(tip[peak]):.6g} m at {frequencies[peak]:.6g} Hz.\n"
        f"- Maximum independent-point no-slip error: {maximum_no_slip:.3%}.\n"
        f"- Maximum modal equilibrium error: {maximum_equilibrium:.3g}.\n"
        f"- Maximum impedance reciprocity defect: {maximum_reciprocity:.3%}.\n"
        f"- Minimum resistance eigenvalue: {minimum_resistance:.6g} N s/m.\n\n"
        "The sampled peak is not a fitted resonance. The structural modes and pressure "
        "orders are truncated; finite no-slip and reciprocity errors must be checked "
        "before interpreting precision. A negative resistance eigenvalue indicates "
        "nonphysical discrete fluid loading. Production source hashes were unchanged.\n"
    )
    (args.output / "report.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2, 1, figsize=(8, 7), sharex=True, constrained_layout=True
        )
        axes[0].semilogy(frequencies, abs(tip) * 1e9, "o-", label="Weighted 3D fluid")
        axes[0].semilogy(
            frequencies, abs(dry.sum(axis=1)) * 1e9, "--", label="Dry beam"
        )
        axes[0].set(ylabel="Tip displacement (nm)")
        axes[0].legend()
        axes[1].plot(frequencies, np.unwrap(np.angle(tip)), "o-")
        axes[1].set(xlabel="Frequency (Hz)", ylabel="Tip phase (rad)")
        fig.savefig(args.output / "tip_spectrum.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(x * 1e6, displacement[peak].real * 1e9, label="Real")
        ax.plot(x * 1e6, displacement[peak].imag * 1e9, label="Imaginary")
        ax.set(
            xlabel="x (µm)",
            ylabel="Displacement (nm)",
            title=f"At largest sampled response: {frequencies[peak]:.1f} Hz",
        )
        ax.legend()
        fig.savefig(args.output / "beam_displacement.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
