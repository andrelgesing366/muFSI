# Analytic 3D Stokeslet implementation

`Stokes3DAnalytic` is a separate implementation of the legacy
`Fluid/F3D_1D.py` and `h_matrix_app.get_a_matrix_uniform_mid` method used in
`Example_3_F3D_spectrum_Uniform.ipynb`. The adaptive `Stokes3D` implementation,
its examples, shared grid code, and package exports are left untouched so the
two versions can be developed in parallel.

## Grid and use

Import this model directly from its module for now:

```python
import numpy as np
from mufsi import Fluid, PlateGeometry
from mufsi.hydrodynamics.stokes_3d_analytic import (
    Stokes3DAnalytic, analytic_fluid_grid,
)

geometry = PlateGeometry(500e-6, 50e-6, 5e-6)
fluid = Fluid(density=997, dynamic_viscosity=890e-6)
grid = analytic_fluid_grid(geometry, nx=24, ny=36)
hydro = Stokes3DAnalytic(fluid, grid)

# An illustration of the API, not an example run performed for this change:
omega = 2 * np.pi * 10e3
B = hydro.assemble_matrix(omega)             # v = B p
p = hydro.pressure_from_velocity(omega, np.ones(24 * 36))
```

The grid uses uniform midpoint x-panels over `[0,L]`. In y it uses the original
Chebyshev--Gauss nodes over `[-W/2,W/2]`, midpoint boundaries, and ordinary
integral weights `pi/ny * sqrt((W/2)**2-y**2)`. The x weights are `L/nx`.
There is no `y_uniform` option. The model also accepts a custom `FluidGrid`
with uniform midpoint x-panels and arbitrary transverse panels; it never
assumes translation invariance in y. Odd counts and single-panel directions
are supported.

Points, matrix rows/columns, velocities, pressures, and weights all use x-major
ordering: `(x0,y0), (x0,y1), ..., (x1,y0), ...`. Force projection uses the grid
weights, which differ from panel areas on the Chebyshev grid. This distinction
preserves the notebook's force discretization.

## Analytic reduction

On the planar plate `z=0`, the kernel excluding viscosity is

```text
Kzz(r) = A(lambda*r)/(8*pi*r)
A(q) = 2*exp(-q)*(1 + 1/q + 1/q**2) - 2/q**2
lambda = sqrt(+i*omega/nu)
```

The polar Jacobian cancels `1/r`. The exact radial primitive is

```text
F(R) = [1 - (1+lambda*R)*exp(-lambda*R)]/(4*pi*lambda**2*R)
F(0) = 0
integral = integral_theta [F(r_out(theta)) - F(r_in(theta))] dtheta
```

The exponential-integral terms in the legacy expressions cancel exactly,
leaving this primitive. The new code uses a small-argument series, a stable
radial interval expression for thin/distant panels, and `complex128` throughout.
`omega=0` uses the steady limit. The harmonic convention is `exp(+i*omega*t)`,
matching the rewritten coupled solver; the legacy `sqrt(-i*omega/nu)` results
must be conjugated when comparing conventions.

Only the radial integral is analytic. Angular Gauss--Legendre quadrature
remains, starting from order 8 and checking against order 16. Unconverged panels
alone receive higher orders. `tolerance`, `absolute_tolerance` (before division
by viscosity), `angular_order`, `max_refinements`, and `batch_size` control this
step. `AnalyticConvergenceError` reports exhaustion rather than returning an
unchecked panel value. Successive-order differences are error estimates, not
rigorous bounds. Singular panels are split into reflected quadrants and
integrated using the same primitive, without evaluating the point singularity.

## Assembly and coupling

Uniform x-midpoints give `B[ix,jx] = blocks[abs(ix-jx)]`. Only `nx` transverse
interaction blocks are integrated. Reflection in y further reduces evaluations
when both transverse nodes and boundaries are symmetric; an odd middle row is
computed once. Blocks are copied without transposition across x, since distinct
y-panel widths do not make individual transverse blocks symmetric matrices.
`use_symmetry=False` independently integrates all rows for small validation runs.

`assemble_blocks(omega)` stores only `(nx,ny,ny)` values. `apply_mobility(omega,p)`
applies those blocks without allocating the full matrix. `assemble_matrix` and
the pressure LU still need dense `(nx*ny,nx*ny)` storage; this first implementation
does not eliminate the dense solve cost. One frequency and its LU are cached;
`clear_cache()` releases them. `integration_report` records the rows/panels
actually integrated, angular refinement, and maximum estimated error/target.
An optional `progress(done,total)` callback reports independently integrated rows.

For a coupled plate problem, use the new entry point explicitly:

```python
from mufsi import CoupledProblem
from mufsi.solvers.frequency_response_analytic import AnalyticFrequencyResponseSolver

problem = CoupledProblem(plate, hydro)  # an existing KirchhoffPlate instance
result = AnalyticFrequencyResponseSolver(problem).solve(frequencies, load)
```

This reuses the existing scaled fluid Schur solve, with structural RHS batches,
without a dense structural impedance or explicit inverse. It returns the same
result type, including equilibrium and no-slip residuals, and retains the joint
solve fallback at a dry resonance. It currently requires one MPI rank.
The generic `CoupledProblem.frequency_response` also accepts this hydrodynamic
interface, but currently selects its iterative path; the explicit analytic
solver selects the efficient direct path without changing the shared solver.

## Checks and deferred work

Small tests check the exact steady singular integral, unsteady Duffy integration,
independent Cartesian panel quadrature in all geometric regions, small arguments
and thin radial intervals, nonuniform/asymmetric y, odd counts, block reuse,
pressure residuals, cache invalidation, and tiny coupled systems.

Run only these checks with NumPy/SciPy; DOLFINx and Quadpy are not required:

```console
PYTHONPATH=src .venv/bin/python -m unittest tests.hydrodynamics.test_stokes_3d_analytic tests.solvers.test_frequency_response_analytic -v
```

A cross-method comparison example and isolated time/memory benchmark are now
available; see [the comparison guide](stokes_3d_comparison.md). Full coupled
notebook/spectrum runs and comparisons against saved legacy results remain
deferred. These unit checks alone do not establish full-spectrum agreement.
