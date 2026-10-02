# Q factors from corner-driven displacement

`mufsi.postprocessing.qfactor` implements an SHO amplitude fit and the legacy
energy definition for steady harmonic response. The end-to-end workflow
supports Kirchhoff-Love plates with `Stokes2D`, and Euler-Bernoulli beams with
`SectionForce2D` (Tuck or Sader). It is serial. The same workflow now supports weighted F3D for EB and KL;
see [the 3D/Q guide](f3d_spectrum.md). The existing 2D example is preserved.

## Excitation and observable

`corner_loads(structure, symmetry=..., amplitude=..., inset=...)` supplies
two `PointLoad`s inside a `PointLoads` container. The amplitude is in newtons
**per corner**. The positions are x=(1-inset)L, y=+W/2 and -W/2. The default
inset is 0.001 of the length. Equal force signs select symmetric displacement;
opposite signs select antisymmetric displacement. The observed response is
respectively `(u_plus + u_minus)/2` or `(u_plus - u_minus)/2` at the drive points.

This selects parity, not a single longitudinal resonance: choose an isolated
frequency window as well. The 1D Euler-Bernoulli beam has no width-dependent
or torsional displacement. Its symmetric pair becomes two coincident forces
with total amplitude 2F; antisymmetric excitation raises an error. Cantilever
supports are required because the requested x=L edge must be free.

Point forces use the shared basis evaluator, `F = E.T @ point_amplitudes`,
so interpolation and load projection obey virtual work. Both structural
assemblers clear constrained DOFs after summing the point forces.

## SHO fit

`fit_sho(frequencies, amplitudes, fit_background=False)` fits displacement
amplitudes, accepting magnitudes or complex displacement samples:

```text
|a f0^2 / (f0^2 - f^2 + i f f0 / Q) + b|
```

The numerator is constant with drive frequency for these fixed point forces.
An optional coherent **complex** constant background b accounts approximately
for neighbouring modes. It is not an incoherent additive amplitude floor.
The example enables this background and uses a local frequency window.

Frequency and amplitude are scaled internally, and Q is optimized
logarithmically. The fit requires an interior peak, increasing positive
frequencies, at least nine samples, and at least five samples across the
fitted linewidth f0/Q. Failed or bounded fits raise an error. The result
reports the L2 amplitude-fit residual / L2 data norm so users can judge
whether a single SHO is adequate.

`q_factor(...)` returns only fitted Q. `resonance_frequency(...)` interpolates
the amplitude maximum, which generally differs from the fitted undamped f0.
The fit is for displacement, not a thermal-noise spectrum or displacement PSD.

## Energy definition

`energy_q_factor(frequencies, displacement, stiffness, mass, force)` operates
on numerical arrays without requiring DOLFINx:

```text
u(t) = Re(u_hat exp(i omega t))
v_hat = i omega u_hat
E(t) = 1/2 u(t).T K u(t) + 1/2 v(t).T M v(t)
DeltaW = pi/omega Re(F_hat.H v_hat)
Q_energy = 2 pi max_t(E(t)) / DeltaW
```

The force may be complex and may vary between frequency samples. The maximum
is evaluated analytically from a real 2x2 quadratic form, retaining spatial
phase differences without a time-grid approximation. Full displacement arrays
must have constrained entries zero. K and M are the real symmetric operators
used by the structural solver, including the interior-penalty and boundary
terms. This differs from integrating element curvature alone on a finite mesh.

As in `energy_qfac.py`, only structural bending and kinetic stored energies
are included. Explicit stored fluid energy is not added. In steady response
the input work equals dissipated energy per cycle; positive work is required.
`energy_from_response` assembles the operators and input force, then also
computes fluid work from pressure with the original coupling weights, or from
the beam's resisting line force with consistent mass projection. It reports
the relative work discrepancy and rejects errors above 1e-5.

## Combined workflow

```python
import numpy as np
from mufsi import analyze_q_factor

# solver = FrequencyResponseSolver(CoupledProblem(plate, Stokes2D(fluid, grid)))
# or BeamFrequencyResponseSolver(beam, SectionForce2D(...))
frequencies = np.linspace(3e3, 5.2e3, 81)  # Choose for this resonator/fluid.
result = analyze_q_factor(
    solver, frequencies, symmetry="symmetric", amplitude=1e-9,
    inset=1e-3, fit_background=True,
)
print(result.sho.q_factor, result.energy.q_factor[0])
print(result.sho.relative_error, result.energy.work_balance_errors[0])
```

The helper solves the full response, fits the corner observable, then performs
one additional solve at fitted f0 for energy. It never truncates displacement
into vacuum modes. For a plate use `symmetry="antisymmetric"` and a frequency
window containing the desired width-antisymmetric resonance.

## Slender silicon cantilever in water

Run from the repository root in the scientific environment:

```console
PYTHONPATH=src python examples/qfactor_2d.py --quick --output results/qfactor_2d_quick
PYTHONPATH=src python examples/qfactor_2d.py
PYTHONPATH=src python examples/qfactor_2d.py --no-fit-background --window-linewidths 3
```

The default compares the first three flexural resonances of an
800 x 50 x 5 micrometre beam/plate, E=169 GPa, rho=2330 kg/m3, nu=0.3, in
water with rho=997 kg/m3 and mu=0.00089 Pa s. Each corner receives 1 nN at
x=0.999 L. It uses a 48-element cubic beam, a 48x6 quadratic plate mesh,
a 49x64 F2D grid and 81 samples per resonance. The sweep width is 1.5 times
the Sader linewidth f0/Q. `--modes`, `--samples` and `--output` are adjustable.

Beam + Sader section force provides a control for the fluid model comparison.
The independent reference solves the implicit loaded-frequency equation and
evaluates Q from [Sader (1998), Eqs. (33) and (35)](https://sadermethod.org/Overview_files/docs/JAP_1998.pdf).
Analytical beam vacuum frequencies only locate the sweep windows; they are
not used to construct point forces or truncate the response.

Recorded on 1 October 2026 with DOLFINx 0.10.0.post5, NumPy 2.3.5 and SciPy 1.16.3:

| Flexural resonance | Beam/Tuck Q SHO | Beam/Tuck Q energy | Plate/F2D Q SHO | Plate/F2D Q energy | Sader Eq. (35) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.6615 | 2.9107 | 2.6822 | 2.9315 | 2.8303 |
| 2 | 6.1927 | 5.9602 | 6.2294 | 6.0061 | 6.0563 |
| 3 | 9.6204 | 9.1532 | 9.6362 | 9.2012 | 9.3329 |

The largest fit residual was 0.43%; input/fluid work discrepancies were below
7e-8 and equilibrium residuals below 9e-7. The estimates need not be equal:
the fluid impedance varies with frequency, the background is a local
approximation, and energy is evaluated at fitted f0 whereas Sader uses its
implicit loaded frequency. The paper's SHO analogy and Eq. (35) are most
accurate for Q much greater than one. A small first-resonance fit residual
does not make fitted Q identical to the energy ratio. Finite plate effects
and discretization also contribute; these meshes are not a convergence study.

Outputs are `comparison.csv`, `response.npz`, `parameters.json` and
`comparison.png` in `results/qfactor_2d`. They include complex fields, fit
curves, energies, work per cycle, physical parameters and numerical settings.
`--quick` is a reduced-discretization smoke example.

## Verification

```console
PYTHONPATH=src python -m unittest tests.postprocessing.test_qfactor tests.postprocessing.test_qfactor_fem -v
```

Tests use independently generated damped-oscillator data, phase/scaling
changes, a time-domain work integral and a densely sampled energy maximum.
FEM tests cover point-load virtual work, symmetric and antisymmetric pressure
responses, both complete plate Q workflows, the beam pair's total force, and
energy Q against the Sader control at the implicit loaded frequency.
