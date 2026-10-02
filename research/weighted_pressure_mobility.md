# Continuous weighted-pressure mobility and beam response

The current research defaults are **M=16, K=4** (85 coefficients; H=340 x 85).
They replace the initial M=4, K=6 defaults after a pressure-order study of the
matched first-peak beam comparison. Explicit M=4, K=6 commands and the initial
results below remain available for reproducing the original exploratory run.
See [the three-model comparison](weighted_beam_comparison.md) for the selection
evidence and an example with Sader and 2D Tuck fluid loading.

These scripts are isolated research experiments. They import the existing
point Stokeslet and parameter containers without modifying `src/`. The optional
panel comparison imports the existing analytic panel solver. The new mobility
assembly itself is entirely numerical.

## Basis and matrix

The convention in these scripts is

```text
xi = 2*x/L - 1; eta = 2*y/W
p(x,y) = sum_{m=0..M,k=0..K} a[m,k]*T_m(xi)*T_(2*k)(eta)
         / sqrt((1-xi^2)*(1-eta^2))
```

Thus `M=4, K=6` means **35 coefficients**, with transverse maximum degree 12.
This differs from the older postprocessing script's `M x N` notation, in which
N is the transverse degree. Coefficient flattening is m-major, k-minor.

The matrix `H[i,j] = integral_surface Gzz(target_i-source,omega)*psi_j(source) dA`
maps pressure coefficients [Pa] to velocity [m/s]. H has units m/(Pa s).
Here H is a small dense mobility matrix, not a hierarchical matrix format.
The frequency convention is exp(+i omega t); pressure is normal traction
applied to the fluid. The force applied to the beam has the opposite sign.

Default collocation uses `2*(M+1)` Chebyshev interior x locations and
`2*(K+1)` positive-y interior locations: a **140 x 35 matrix** at the requested
orders. Mirrored negative-y rows would duplicate the equations. Source
integration always includes the full width. The least-squares objective has
equal row weights on this angularly distributed grid; it is not an area norm.
Columns are scaled before the solve. Rank and scaled condition are reported.

## Numerical singular integration

1. Substitute x=L/2*(1+cos(alpha)), y=W/2*cos(beta), with alpha,beta in [0,pi].
   The Jacobian cancels the pressure denominator, giving L*W/4 d(alpha)d(beta).
2. Split this angular rectangle at the target coordinates into four rectangles.
3. Split each rectangle into two triangles sharing the target vertex. Map each
   triangle to u,v in [0,1] with Duffy coordinates `(A*u, B*u*v)` or
   `(A*u*v, B*u)`. The Jacobian contributes A*B*u, cancelling the interior
   kernel's 1/r divergence. Open quadrature nodes never sample r=0.
4. Grade the v intervals using local physical length/width scales. This resolves
   the narrow directional transition in a slender beam. Stable cosine
   differences avoid cancellation close to the target.
5. Use tensor products of **Quadpy-generated Gauss-Legendre line rules** on each
   transformed subregion, evaluating all basis functions together. Increase
   order until successive estimates agree. This is not a plain rectangle rule
   applied to an untreated singular integrand.

No kernel cutoff, omitted self term, epsilon regularization, or analytical
radial integration is used. Quadrature points depend on each target and are
separate from collocation points and pressure unknowns. The installed scientific
environment uses `legacy-quadpy==0.16.10`; explicitly select `--backend gauss`
for NumPy-generated equivalent rules without Quadpy. There is no silent fallback.

The stopping estimate is max_j abs(H_new[i,j]-H_old[i,j]), compared with
`atol + rtol*max_j abs(H_new[i,j])`. atol has mobility units after division by
viscosity. This controls a row-scaled error estimate, not relative accuracy of
every tiny cancelling entry. Exhausting the permitted orders raises an error.
Successive-order agreement is not a rigorous error bound.

## Run

From `muFSI/` in the existing scientific Python environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/weighted_pressure_projection.py --M 4 --K 6 --reference
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/weighted_beam_spectrum.py --M 4 --K 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/test_weighted_pressure.py -v
```

The first script prescribes unit real first-mode tip velocity. Its default
frequency satisfies omega*(W/2)^2/nu=1 (909.274 Hz for the default geometry and
fluid). It saves H, coefficients, collocation and separate check-point results,
pressure plots, JSON diagnostics and a Markdown report. It also tightens
quadrature tenfold on five representative rows using different orders.
`--reference` compares modal force with 48 x 32 constant-pressure analytic panels;
this finite panel solution is an independent comparison, not exact truth.

For a cheaper square solve, choose `--nx 5 --ny-half 7`. Check-point errors still
need attention even when its algebraic residual is zero. Defaults use an
overdetermined system. Increase collocation density and pressure degree
independently, for example:

```sh
.venv/bin/python research/weighted_pressure_projection.py --M 4 --K 6 --nx 15 --ny-half 21 --output research/results/weighted_pressure_more_collocation
.venv/bin/python research/weighted_pressure_projection.py --M 8 --K 6 --output research/results/weighted_pressure_projection_M8
.venv/bin/python research/weighted_beam_spectrum.py --frequencies 0 909.274 10000 15000 --output research/results/weighted_beam_selected
```

All dimensions, fluid parameters, tolerances, orders and output directories are
CLI options; `--help` lists them. Results remain in the ignored `research/results/`
folder. No full fluid pressure grid or dense panel LU is needed for the new solve.

## Beam displacement spectrum

`weighted_beam_spectrum.py` uses exact homogeneous dry clamped-free
Euler-Bernoulli mode shapes, normalized to unit tip displacement. Defaults:
two modes, L=500 um, W=25 um, t=5 um, E=169 GPa, solid density=2330 kg/m3,
fluid density=997 kg/m3, viscosity=890 uPa s, real 1 nN tip point force.
There is no separate support surface, finite-thickness fluid surface, or
structural material damping. No DOLFINx installation is needed for this
truncated modal research model. The exact-mode implementation lives in research.

At each frequency, solve `H*A=E` in least squares for all unit modal velocities
together. Let `C[n,j]=integral phi_n(x)*psi_j(x,y) dA`. The transverse Chebyshev
orthogonality makes C zero except for k=0. Compute Z=C*A and solve

```text
(K_struct - omega^2*Mass_struct + i*omega*Z) d = F
pressure_coefficients = i*omega*A*d
w(x) = sum_n phi_n(x)*d_n
```

The structural mass/stiffness are diagonal for these exact dry modes; Z generally
is dense and includes nonlocal fluid coupling. The script retains the full
computed Z without imposing symmetry. Outputs include complex spatial
displacement, modal displacement, pressure coefficients, impedances, selected
H matrices, CSV tip amplitudes/phases, plots and diagnostics at every frequency.
The default 31-point sweep covers 0.02 to 1.2 times the first dry frequency.
Its largest sampled response is not a refined resonance estimate.

Diagnostics include independent-point no-slip error, errors for individual
unit mode velocities, structural equilibrium, reciprocity Z=Z.T and the
eigenvalues of the Hermitian resistance (Z+Z.conj().T)/2. Oversampled collocation
does not guarantee exact reciprocity or passivity; inspect these quantities.
Increasing the structural mode count alone cannot repair an insufficient
longitudinal pressure basis.

## Results from the first runs

- Requested M=4, K=6: H assembly about 1.3 s; coefficient solve about 0.0004 s;
  rank 35, scaled condition 6.72. At 909.274 Hz, check-point velocity error is
  6.24%. Tenfold quadrature tightening changes the sampled H rows by 4.2e-11
  in relative Frobenius norm. Modal force differs by 1.56% from 48 x 32 panels.
- M=8, K=6: check-point velocity error falls to 0.94%, with assembly about 2.9 s.
  This is evidence of a longitudinal-basis limitation at the original orders.
- Requested-order, two-mode, 31-frequency spectrum: about 108 s including
  check-point assembly; first dry frequency 27.515 kHz; sampled wet peak at
  13.538 kHz, amplitude 2.843 nm for 1 nN. Maximum check-point no-slip error is
  12.02%; maximum reciprocity defect 0.504%; resistance eigenvalues remain
  positive. The spectrum is exploratory at this pressure degree.
- Five-frequency check at M=8, K=6 (63 coefficients): maximum independent-point
  no-slip error 3.42%, reciprocity defect 0.004%, and positive resistance
  eigenvalues. At 13.538 kHz the tip amplitude is 2.862 nm, about 0.7% above
  the default-order value. This is a selected-frequency pressure-order check,
  not a converged spectrum or structural-mode convergence test. Results are
  in `results/weighted_beam_selected_M8/`. Reproduce with:

  ```sh
  .venv/bin/python research/weighted_beam_spectrum.py --M 8 --K 6 --frequencies 909.274 10000 13537.6 18000 30000 --output research/results/weighted_beam_selected_M8
  ```

## Notebook fit checks

The existing executed notebook outputs are preserved. The added comparison cell
evaluates the 36-, 65-, and 102-coefficient **previous panel-pressure fits** as
continuous weighted densities at four independent points. It compares polar
and cosine/Duffy integrations at orders 48 and 96. These fits are distinct from
the new direct coefficient solve above. The notebook uses the older maximum
transverse degree N notation for filenames.

At integration order 96, the complex velocity error relative to the prescribed
local first-mode velocity is:

| Coefficients | x/L=0.25, eta=0 | x/L=0.6, eta=0 | x/L=0.9, eta=0 | x/L=0.99, eta=0.8 |
|---|---:|---:|---:|---:|
| 36 | 1.005% | 0.359% | 0.560% | 5.665% |
| 65 | 0.371% | 0.182% | 0.247% | 5.831% |
| 102 | 0.234% | 0.232% | 0.265% | 5.182% |

Increasing fit degree improves the interior overall, but does not resolve the
tip/side-edge discrepancy. The cosine/Duffy order comparison is stable;
agreement with the independent polar integration is within about 1e-5 m/s
for a unit tip velocity. This motivates solving continuous coefficients
directly, rather than increasing only the degree of the old panel fit.

Run `research/run_weighted_fit_checks.py` to rerun just the numerical comparison
and save its output into the notebook without repeating the symbolic attempts.
The machine-readable comparison is
`results/weighted_stokeslet_symbolic/fit_order_comparison.json`.

The product endpoint weight is still an empirical candidate. These runs do not
establish the rectangular corner exponent, pressure-order convergence across
the spectrum, or structural-mode convergence. Tight quadrature and small
equilibrium residuals must not be interpreted as a converged fluid solution.

The independent checks include an elliptic-integral steady center value,
reflection symmetry, backend agreement, known force moments, manufactured
complex coefficient recovery, explicit convergence failures, static EB
compliance with its known truncation tail, and fluid added-mass/damping signs.
Every executed workflow records production hashes and verifies they are unchanged.

See generated [projection report](results/weighted_pressure_projection/report.md)
and [beam spectrum report](results/weighted_beam_spectrum/report.md).
