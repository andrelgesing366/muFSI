# Weighted polynomial 3D fluid loading

The canonical `mufsi.Stokes3D` integrates the unsteady Stokeslet against a
continuous Chebyshev pressure expansion with inverse-square-root weights at
all four rectangle edges. It supports Euler-Bernoulli (EB) and Kirchhoff-Love
(KL) finite-element structures. The rectangle is x in [0,L], y in [-W/2,W/2].

```text
xi = 2*x/L-1, eta = 2*y/W
p(x,y) = sum a[m,n]*T_m(xi)*T_n(eta) / sqrt((1-xi^2)*(1-eta^2))
H[i,(m,n)] = integral Gzz(target_i-source,omega)*psi[m,n](source) dA
H a ~= v       (column-scaled, oversampled complex least squares)
```

`x_degree` and `y_degree` always mean maximum polynomial degree. EB retains
only even transverse degrees up to `y_degree`; KL retains all degrees,
including the odd terms needed for torsional/antisymmetric motion. In the older
research EB API, K meant an even-degree index, so y_degree=2*K. The KL research
API used K as its actual transverse maximum degree. Production avoids this
ambiguity. Defaults are x_degree=16 and y_degree=8; both remain adjustable.

```python
from mufsi import CoupledProblem, FrequencyResponseSolver, Stokes3D
hydro = Stokes3D(fluid, beam.geometry, formulation="EB",
                 x_degree=16, y_degree=8)
solver = FrequencyResponseSolver(CoupledProblem(beam, hydro))
response = solver.solve(frequencies_Hz, load)
# For a plate: Stokes3D(fluid, plate.geometry, formulation="KL", ...)
```

## Singular integration and force projection

Cosine coordinates cancel the two pressure edge weights. Duffy triangle
quadrature removes the coincident-point 1/r Stokeslet singularity. The stable
unsteady kernel is shared with the legacy reference. Numerical Gauss rules are
the default and require no Quadpy; quadrature_backend="quadpy" remains an
option. Successive rules must meet tolerance, otherwise assembly raises an
IntegrationConvergenceError. No artificial kernel regularization is applied.
The product edge weight is a useful representation, not a verified exact
rectangular-corner exponent.

`assemble_matrix(omega)` returns a rectangular mobility, not the square
panel-pressure matrix of the previous API. `coefficients_from_velocity` solves
for pressure coefficients; `pressure_from_velocity` evaluates their continuous
pressure at the collocation points. `pressure_from_coefficients(a, points)`
accepts arbitrary strictly interior evaluation points. Edge values are excluded
because the representation is singular there. Angular frequency is rad/s.

EB collocation uses independent positive-y points. KL uses the full width;
reflected mobility rows reuse integration with the correct sign for each
transverse degree. `nx` and `ny` control collocation independently of polynomial
orders; EB ny counts half-width points, KL ny counts full-width points.
Only the most recent frequency is cached.

The FE-to-fluid evaluation E and force projection C are different operators:

```text
E[i,j] = phi_j(collocation_i)
C[j,k] = integral phi_j(x,y)*psi_k(x,y) dA
L = least-squares action of H
D u + C a = F,       a = i*omega*L*E*u
(I + i*omega*L*E*D^-1*C) a = i*omega*L*E*D^-1*F
```

The coefficient Schur solve retains every free structural DOF, with a scaled
joint-system fallback at dry poles. It does not truncate structural modes or
construct a dense structural impedance. For EB, transverse orthogonality gives
an exact zero projection for every nonconstant transverse term. For KL,
force quadrature works cell by cell in cosine-space vertical slices, so it does
not cross FE derivative discontinuities. Successive orders check the force
projection, independently of fluid quadrature (default relative norm tolerance
1e-3). A custom `WeightedCouplingOperator.from_structure(..., tolerance=...,
orders=...)` can tighten this check before constructing the problem.

## Responses, Q and field recovery

The time convention is exp(+i omega t). Pressure is traction applied to the
fluid; the actual fluid force on the structure is its negative. Resisting
forces enter the left-hand side of structural equilibrium.

`FrequencyResponseResult` keeps full-DOF displacement, sampled pressure,
pressure_coefficients, pressure_points, full-DOF resisting fluid_force,
equilibrium residuals, collocation no-slip residuals and force_projection_error.
The no-slip residual includes finite-basis approximation error; it need not
approach the linear solver's equilibrium residual. Refine pressure degrees,
collocation and FE mesh independently. Use velocity_from_coefficients at
independent points to investigate pressure approximation further.

`analyze_q_factor` supports EB/KL with weighted 3D, the existing 2D models,
and EB/Sader section forces. It fits a resolved isolated resonance with the
SHO method and evaluates energy Q at its fitted frequency. Energy Q uses the
maximum structural bending-plus-kinetic energy and model-specific work per
cycle; there is no explicit stored-fluid-energy term. Weighted pressure work
uses C*a, never collocation areas times sampled singular pressure.

`reconstruct_flow_from_response` evaluates structural velocity on a separate
2D grid, solves Stokes2D, and recovers the existing streamfunction, velocity,
strain and dissipation. This is explicitly a 2D field approximation, even when
the displacement comes from 3D or Sader loading. Its dissipation does not
replace model-specific work for Q. Full 3D field recovery remains deferred.

## Small runnable examples

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python examples/formulation_comparison.py
PYTHONPATH=src .venv/bin/python examples/formulation_comparison.py --resonances 1
PYTHONPATH=src .venv/bin/python examples/plate_3d.py --resonances 1
PYTHONPATH=src .venv/bin/python examples/formulation_comparison.py --plots-only
```

The first example compares EB/3D, KL/3D, EB/2D, KL/2D and EB/Sader on the same
800 x 50 x 5 micrometre silicon cantilever in water, under the same two-corner
forcing. Defaults cover two bending resonances with 25 samples each. It saves
spectral_displacement.png, qfactor_vs_frequency.png, qfactor.csv, spectra.npz,
report.json and one flow_2d_approximation.png. The report retains SHO fit,
no-slip, projection and work-balance diagnostics. Modest orders/meshes make this
a demonstration, not a continuum-convergence benchmark. `--models` selects
individual formulations; `--x-degree`, `--y-degree` and mesh options allow
later refinement.

An antisymmetric plate example uses `--symmetry antisymmetric --resonances 1`
with an explicit `--f-min`/`--f-max` window containing its odd resonance.
EB cannot represent odd width motion.

## Legacy implementations

The former Quadpy panel solver, fully radial-analytic solver, multigrid and
hybrid/multigrid classes are retained under `hydrodynamics/legacy/` with their
panel integration and grid helpers. Existing comparison tests and benchmarks
still target those methods. Research drivers now import promoted library code;
production modules never import research. See [the legacy panel guide](legacy/f3d_spectrum.md)
and the preserved [analytic](legacy/stokes_3d_analytic.md),
[multigrid](legacy/stokeslet_multigrid.md), and
[hybrid](../research/stokeslet_hybrid_results.md) reports for their original
formulations and measurements. The [formulation study](formulation_study.md)
documents the extensive comparisons of the active methods.


## Compact example validation

The five-formulation example completed two bending-resonance windows with
25 samples each. All 101 repository tests and the 16 existing independent
weighted-pressure checks passed. No new runtime/memory convergence benchmark
was run. The 3D collocation velocity residuals in this demonstration reached
0.28% in the first resonance window and 1.26% in the second; SHO amplitude-fit
residuals remained below 0.8%. Work-balance errors were below 2e-9 across the
five models. The stored report and CSV include the mesh, polynomial orders,
fit residuals, projection errors and work balance.

First-window fitted f0 was 4.174 kHz (EB/3D), 4.208 kHz (KL/3D), 4.079 kHz
(EB/2D), 4.114 kHz (KL/2D) and 4.074 kHz (EB/Sader). These are finite-
discretization comparisons, not a validation of exact corner weights or a
continuum accuracy bound. `--plots-only` regenerates figures and the 2D field
from saved full-DOF spectra without another frequency sweep.
