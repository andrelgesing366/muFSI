> Legacy reference: these panel implementations are now under
> `mufsi.hydrodynamics.legacy`. The active `mufsi.Stokes3D` is documented in
> [the weighted-pressure guide](../f3d_spectrum.md).

# Padded multigrid Stokeslet assembly

`hydrodynamics/legacy/stokeslet_multigrid.py` implements Arthur Vernydub's bachelor
thesis method: a uniform lattice of reusable Stokeslet integrals and a smaller
pressure grid with hierarchical refinement near the moving edges. It is an
additional F3D model; existing analytic and Quadpy models remain available.

## Use and force weights

```python
from mufsi import CoupledProblem
from mufsi.hydrodynamics.legacy.stokeslet_multigrid import (
    Stokes3DMultigrid, multigrid_fluid_grid,
)

grid = multigrid_fluid_grid(
    geometry, x_partitions=(3, 3), y_partitions=(3, 3)
)  # 5 x 7 pressure panels; 9 x 9 fine integration lattice
hydro = Stokes3DMultigrid(fluid, grid, tolerance=1e-6)
result = CoupledProblem(plate, hydro).frequency_response(frequencies, load)
```

Each initial partition is split into the next odd number of subdivisions at
the free tip (x) or both width edges (y). Single entries give uniform midpoint
panels. Odd integer widths ensure each pressure-panel centroid coincides with
a fine-cell centroid. The default safety limit is one million fine cells.
Custom grids must satisfy the same midpoint/lattice requirements.

Points, matrix rows/columns and pressure use x-major order. Coordinates are
metres, angular frequency is rad/s, velocity is m/s and pressure is resisting
traction in Pa, with `exp(+i omega t)` and `lambda=sqrt(+i omega/nu)`.
Viscosity and `1/(8*pi)` are included exactly once in mobility `v=Bp`.

Pressure is constant on each nonuniform rectangle. Its force weight is

```text
Q[ix*ny+iy] = (x_edges[ix+1]-x_edges[ix]) * (y_edges[iy+1]-y_edges[iy])
G = E.T @ diag(Q)
(K-omega^2*M) u + G p = F
B p = i*omega*E u
```

These weights are actual panel areas, not uniform weights or Chebyshev weights.
Their sum is the plate area up to roundoff. The mobility already integrates
each source cell: do not multiply its columns by Q a second time.
The grid builder supplies Q, and the existing coupling operator consumes it.

`Stokes3DMultigrid` inherits the legacy panel Stokes3D's pressure LU/cache lifecycle but replaces
matrix assembly. This also selects the existing direct fluid Schur solve in
`FrequencyResponseSolver`, recovering all structural DOFs without modal
truncation. The dry-pole joint fallback is retained and tested.

## Integral reuse and differences from the thesis code

Only the first quadrant of unit-cell offsets is integrated (`mx*my` cells).
For each pressure centroid, integer absolute offsets select a translated,
reflected cutout. Two `np.add.reduceat` operations sum unit integrals into
each source pressure panel. A full padded array is not allocated.

The existing stable complex128 kernel and adaptive rectangle integration are
used. The singular unit cell uses an exact radial primitive and converged
angular quadrature, replacing the thesis's circle-minus-fixed-polygon segments.
This avoids fixed arc-discretization error and additional MeshPy dependencies.
Quadpy is the default unit-cell backend; `quadrature_backend="gauss"` needs
only NumPy/SciPy. Local quadrature error estimates are summed over each coarse
source panel and checked against its tolerance. Failure raises an exception.
These estimates do not establish pressure-grid convergence.

Dense matrix storage and pressure LU still cost O(N^2) and O(N^3). Fine-grid
integration and cutout summation can become expensive with excessive refinement.
Only the latest frequency is cached; `clear_cache()` releases matrix/LU memory.

## Reproducible pressure and spectrum checks

Run in the existing WSL scientific environment:

```console
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python examples/legacy/pressure_multigrid.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python examples/legacy/plate_multigrid.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python examples/legacy/plate_multigrid.py --x-partitions 5,3 --y-partitions 5,5 --output results/multigrid_spectrum_7x13
```

Each script saves a plot, arrays and JSON report. The spectrum script also
saves CSV with complex tip displacement and per-frequency differences.

The pressure example uses L=500 um, W=50 um, thickness=5 um, water
(rho=997 kg/m^3, mu=890e-6 Pa s), frequency=1 kHz and constant displacement
1 nm. All three methods share a 5x9 grid (uniform x, refined y), which is
compatible with the analytic model's uniform-x requirement.

| Multigrid difference | Analytic | Quadpy |
| --- | ---: | ---: |
| Matrix relative Frobenius norm | 3.77e-12 | 2.87e-10 |
| Pressure relative L2 norm | 8.27e-12 | 4.13e-10 |
| Integrated force relative difference | 7.25e-12 | 2.07e-10 |

The full spectrum uses a 24x4 structural mesh, 825 structural DOFs, E=169 GPa,
rho_s=2330 kg/m^3, nu_s=0.3, a uniform 1 Pa driving load, and 32 logarithmic
samples from 1 to 400 kHz. Integration tolerance is 1e-6. The Quadpy reference
needs up to 20 refinement rounds on these long coarse panels at high frequency.

| Identical-panel spectrum comparison | 5x7 grid | 7x13 grid |
| --- | ---: | ---: |
| Maximum full-displacement relative L2 difference | 4.33e-8 | 1.70e-7 |
| Maximum pressure relative L2 difference | 3.84e-8 | 1.77e-7 |
| Maximum pointwise tip relative difference | 1.12e-7 | 1.81e-7 |
| Multigrid total sweep time | 0.54 s | 0.93 s |
| Quadpy total sweep time | 6.07 s | 22.86 s |

Times are single local runs with one BLAS/OpenMP thread, not repeated benchmark
medians. Maximum no-slip residuals are about 2e-15. Load-relative equilibrium
residuals reach 4.9e-7 for both methods; cancellation in the stiff FE equations
limits this measure. The script explicitly accepts equilibrium residuals below
1e-6 and no-slip residuals below 1e-8, separately from the 1e-4 method-agreement
criterion. An initial run with a 1e-8 equilibrium criterion failed this check
despite close method agreement; these residuals are retained in the reports.

The script also runs Quadpy on its native Chebyshev grid with the same point
counts. Relative to its maximum tip amplitude, the maximum complex tip
difference is 34% at 5x7 and 26% at 7x13. These runs change both the
pressure discretization and its force quadrature. They demonstrate that the
very coarse grids are not interchangeable or established as converged.
The close identical-panel comparisons validate the new assembly and coupling;
they do not establish that either coarse grid predicts a converged spectrum.

Tests cover exact area weights and virtual work, single/uniform/refined and
asymmetric panels, direct panel integration, the actual Quadpy and analytic
references, multiple RHSs, cache behavior, invalid lattices, and the full
coupled joint system including a dry pole.

```console
PYTHONPATH=src .venv/bin/python -m unittest tests.hydrodynamics.legacy.test_stokeslet_multigrid tests.solvers.legacy.test_frequency_response_multigrid -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t . -q
```
