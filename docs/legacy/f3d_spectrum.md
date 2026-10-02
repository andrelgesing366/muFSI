# Legacy panel F3D reference

The following records the published constant-panel implementation. Import it
from `mufsi.hydrodynamics.legacy.stokes_3d`; the active `Stokes3D` now uses
weighted polynomial coefficients. The old example is `examples/legacy/plate_3d.py`.

## Adaptive F3D frequency response

`Stokes3D` ports the rectangle-panel method in the original `Fluid/F3D.py`.
It includes interactions between all longitudinal sections of a plate.
`examples/legacy/plate_3d.py` compares this model with F2D and Sader for slender
and wide cantilevers, using the isotropic DOLFINx Kirchhoff–Love plate.

The formulation follows Gesing, Platz and Schmid, *On the 3D Stokes flow
around non-slender MEMS resonators*, *Computers and Fluids* **299** (2025),
106677, [doi:10.1016/j.compfluid.2025.106677](https://doi.org/10.1016/j.compfluid.2025.106677),
particularly Eqs. (4)–(7), (13)–(18), and (21).

## Formulation and signs

Coordinates and material/fluid properties use SI units. The zero-thickness
plate occupies `[0,L] x [-W/2,W/2]`; the fluid is Newtonian, incompressible,
unbounded and linearized about rest. This implementation handles transverse
motion on a planar plate. It does not include walls or finite-thickness flow.

The library uses `exp(+i omega t)`, with angular frequency in rad/s and
`lambda = sqrt(+i omega/nu)`, where nu = mu/rho. Define

```text
z = lambda*r
A(z) = 2 exp(-z) (1 + 1/z + 1/z^2) - 2/z^2
B(z) = -2 exp(-z) (1 + 3/z + 3/z^2) + 6/z^2
Kzz = [A(z)/r + B(z) dz^2/r^3] / (8*pi*mu)
mobility[i,j] = integral over panel j of Kzz(observation i - source) dA
velocity = mobility @ pressure
```

Positive pressure is resisting traction; the fluid force on the structure is
its negative. The kernel is conjugated relative to the original code's
`sqrt(-i omega/nu)`. The positive steady limit,
`Kzz = (1 + dz^2/r^2)/(8*pi*mu*r)`, fixes the normalization and sign.
There is no extra factor of two for the two plate faces. Small-argument
Taylor expansions remove cancellation in A, B and the radial primitive.

`unsteady_stokeslet_zz` accepts 2D or 3D separation vectors, including
omega=0 for point-kernel verification. `Stokes3D` requires omega>0.

## Panels and grids

`FluidGrid.cantilever(geometry, nx=12, ny=24)` preserves the legacy F3D grid:

- Half Chebyshev–Gauss nodes in x, clustered at the free tip:
  `x_i = L sin((2i-1) pi/(4nx))`, i=1,…,nx.
- Full ascending Chebyshev–Gauss nodes in y, clustered at both edges.
- Panel boundaries halfway between adjacent nodes, with the physical outer
  boundaries at 0, L, and ±W/2.
- Pressure projection weights
  `wx = pi/(2nx) sqrt(L^2-x^2)` and
  `wy = pi/ny sqrt((W/2)^2-y^2)`, without renormalization.
- x-major ordering: `ix*ny + iy`. Even and odd counts are both supported.

`x_uniform=True` and `y_uniform=True` select midpoint panels and ordinary
uniform weights over the full physical dimensions. `FluidGrid.midpoint`
also defines valid 3D rectangles. The F2D Simpson grid has endpoint x nodes
and does not define 3D panels.

`grid.panel_bounds` returns `(nx*ny,4)` rectangle bounds in grid order;
`x_panel_edges` and `panel_edges` store the longitudinal/transverse boundaries.
Mobility uses actual panel integrals; structural force projection uses the
grid's separate area quadrature weights, as in the original method.

## Tolerance-controlled integration

The default backend uses Quadpy C2 rules of degree 2, 4, 6 and 8. Successive
complex integral estimates are compared for each regular panel. Unresolved
panels undergo local longest-side bisection; accurate leaves are retained.
The summed local error estimates and successive total integrals must satisfy

```text
estimated error <= absolute_tolerance + tolerance*abs(integral)
```

The defaults are relative tolerance 2e-3, absolute tolerance 1e-15 m
(before division by viscosity), at most 12 refinement rounds and 16384
subpanels per regular entry. `QuadratureConvergenceError` reports failure;
the implementation never silently accepts a panel that exceeded its limits.

For the singular self panel, an inscribed disk is integrated analytically.
The stable planar radial primitive, excluding viscosity, is

```text
H(R) = integral_0^R Kzz(r)*mu*r dr
     = [1-(1+lambda*R) exp(-lambda*R)] / (4*pi*lambda^2*R)
disk integral = 2*pi*H(R)
H(R) -> R/(8*pi) as lambda -> 0
```

The rectangle remainder uses this exact radial primitive and angular
Gauss–Legendre integration, split at each corner direction. Angular order
starts at 15, is checked against 30, and doubles if needed, up to 960.
This is the special diagonal treatment of the adaptive panel method.
General regular entries still use 2D cubature.

`quadrature_backend="gauss"` uses NumPy tensor Gauss–Legendre rules with the
same tolerance/refinement algorithm and no Quadpy dependency. Tests compare
the two backends. Error estimates control numerical integration of a fixed
panel discretization; they are not rigorous bounds or a fluid-grid
convergence study.

After assembly, `hydro.integration_report` records the backend, frequency,
number of evaluated rows, locally refined entries, maximum estimated
relative error, maximum error/target ratio, and maximum refinement depth.
Counts refer to actually evaluated entries; symmetry copies are excluded.
Reports are saved at every frequency by the example.

## Assembly and coupled solve

`assemble_matrix(omega)` returns read-only complex128 mobility in m/(Pa s).
Transverse reflection is reused for symmetric grids. Uniform x panels also
reuse longitudinal translation and reflection. Only the most recent
frequency is cached. `pressure_from_velocity` accepts a vector or a batch
of velocity columns and solves with reusable LU factors. No inverse is formed.
`clear_cache()` releases the cached matrix and default LU factors.

After eliminating fixed structural DOFs, both fluid methods solve

```text
D u + G p = F,          D = K - omega^2 M, G = E.T Q
-i omega E u + B p = 0
```

For dense F3D mobility, the solver forms the exact fluid Schur system

```text
(B + i omega E D^-1 G) p = i omega E D^-1 F
u = D^-1 (F-Gp)
```

Sparse structural LU solves use batches of 64 RHS columns. The full FEM
displacement is recovered, without truncating structural modes or forming a
dense structural hydrodynamic impedance. A scaled joint solve is used if D
cannot be factored at an in-vacuo pole. Two residual corrections reuse the
factors. Coupled response currently requires one MPI rank.

The dense fluid matrices still cost O((nx*ny)^2) storage and their LU costs
O((nx*ny)^3). At 32x64, each complex128 fluid matrix occupies 64 MiB;
assembly, Schur work and LU require additional memory. Start with coarse grids.

## Environment and execution

Use the matched DOLFINx/PETSc environment described in
[the plate guide](../plate_eigenproblem.md). In Linux/WSL, an isolated environment
can reuse those system packages:

```console
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e ".[quadpy,plot]"
.venv/bin/python examples/legacy/plate_3d.py --quick
.venv/bin/python examples/legacy/plate_3d.py
```

The `quadpy` extra selects `legacy-quadpy==0.16.10`, which provides the
original C1/C2 APIs and is distributed under GPL-3.0-or-later
([package metadata](https://pypi.org/project/legacy-quadpy/)).
Current [official Quadpy](https://github.com/sigma-py/quadpy) requires its
own license. No Quadpy implementation is copied into µFSI.
To use only NumPy/SciPy quadrature, install the `plot` extra and run

```console
PYTHONPATH=src python3 examples/legacy/plate_3d.py --quick --quadrature gauss
```

Useful options include `--case slender`, `--case wide`, `--nx`, `--ny`,
`--samples`, `--tolerance`, `--uniform-x`, `--f-min`, `--f-max` and
`--output`. The comparison script enforces 3<=nx<=32 and 1<=ny<=64.
F2D uses the previous odd x count when nx is even, so its actual count also
respects the limit.

| Setting | Default | Quick |
| --- | ---: | ---: |
| F3D fluid points | 12x24 | 6x12 |
| F2D fluid points | 11x24 | 5x12 |
| Frequency samples | 72 | 24 |
| Slender crossed P2 mesh | 40x4 | 24x4 |
| Wide crossed P2 mesh | 40x20 | 24x12 |

Both geometries have L=500 um and t=5 um; W=50 um for the slender case and
250 um for the wide case. Material: E=169 GPa, rho_s=2330 kg/m^3, nu_s=0.3.
Water: rho=997 kg/m^3, mu=890e-6 Pa s. The load is uniform pressure 1 Pa,
measured at (L,W/2). Logarithmic frequencies span 1–400 kHz for the slender
case and 1–150 kHz for the wide case.

Sader is a slender-beam reference and is an extrapolation for the wide case.
The plots show driven displacement amplitude in nm/Pa, not a thermal PSD.

## Outputs and verification

Each case writes `spectrum.png`, `spectrum.csv`, `response.npz` and
`parameters.json` under `results/plate_3d/{slender,wide}/`.
A combined `comparison.png` is written at the parent level.
CSV contains complex tip responses and force/no-slip residuals; NPZ contains
full displacement and pressure arrays, coordinates, weights and 3D panels.
JSON records physical parameters, discretization, versions, runtime,
conventions and per-frequency integration reports.

Independent tests cover the original point-kernel formula with harmonic
conversion, the exact steady singular rectangle, unsteady self panels against
Duffy-triangle integration, near-panel refinement and failure limits,
Quadpy/Gauss agreement, symmetry, odd counts, pressure batches, positive rigid
dissipation, the slender F2D limit, coupled FEM response against a separate
joint dense solve, and the singular-D fallback. Existing F2D and plate/eigen
tests remain in the suite.

```console
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -v
```

On DOLFINx 0.10.0.post5, PETSc 3.24.4, NumPy 2.3.5 and SciPy 1.16.3, the
default Quadpy examples gave the following first sampled amplitude maxima:

| Geometry | F3D [kHz] | F2D [kHz] | Sader [kHz] |
| --- | ---: | ---: | ---: |
| Slender, L/W=10 | 11.556 | 11.556 | 11.556 |
| Wide, L/W=2 | 7.742 | 6.264 | 6.264 |

The slender spectra closely overlap around the first response peak; finite
end effects and discretization cause differences at higher modes. The wide
F3D response shifts relative to the section-based models. These values are
maxima on a 72-point frequency grid, not fitted resonance frequencies.
The two F3D sweeps took approximately 47 and 48 seconds after imports in the
test environment. Maximum force-balance residuals were 1.1e-6 and 1.8e-6;
no-slip residuals were below 6.1e-15.

All 36 serial tests passed in this environment, including the Quadpy backend
comparison and an elongated self panel from a 32x64 grid. A complete coarse
Gauss example also ran in the system environment without Quadpy. Ruff checks
passed for the library, examples, benchmarks and tests.

These deliberately small examples demonstrate the workflow. Publication
results need separate fluid-grid, structural-mesh, integration-tolerance
and frequency-resolution convergence studies.
