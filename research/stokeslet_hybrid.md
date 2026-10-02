# Hybrid Stokeslet experiment

The preserved implementation is
[the legacy hybrid module](../src/mufsi/hydrodynamics/legacy/stokeslet_hybrid.py).
[stokeslet_hybrid.py](stokeslet_hybrid.py) retains compatibility imports and
forwards its command-line entry point to
[the legacy benchmark](../benchmarks/legacy/benchmark_stokeslet_hybrid.py). It keeps the existing
piecewise-constant pressure representation, stable unsteady kernel, SI units,
`exp(+i omega t)` convention, dense pressure solve, and full structural Schur
solve. No weighted screen formulation is used.

See [measured results and recommended settings](stokeslet_hybrid_results.md).

## Integration and grids

`Stokes3DHybrid` selects radial integration when the distance from the observation
to the source rectangle is at most `near_ratio` times the rectangle diagonal.
The default ratio is 0.5. Self panels always use the radial primitive. Far panels
compare Quadpy degrees 2 and 4, then 6 and 8 if needed. Panels that still fail the
complex-integral tolerance switch to radial integration instead of subdividing.
The radial primitive is exact; its remaining angular integral is numerical and
uses order doubling. These error estimates are not rigorous bounds.

`edge_clustered_grid` puts panel boundaries on a cosine distribution and uses
midpoint collocation. The default clusters at both x ends and both y edges;
`x_clustering="tip"` and `"uniform"` are comparison options. Clustering improves
resolution of edge traction but does not change the pressure basis.

`hierarchical_grid` creates odd integer widths on a smallest-cell lattice. It
can refine both x ends or just the tip. `Stokes3DHybridMultigrid` integrates an
offset table with the same hybrid rule and sums its translated unit integrals
into source panels. Signed-offset prefix sums avoid constructing a fine-grid
cutout for every observation. A roundoff allowance is included in the summed
error check. Prefix construction/queries and storage scale as O(M + N²), where
M is the fine offset table size and N the pressure count. Dense fluid/Schur
matrices and LU remain.

`lattice_edge_grid` rounds cosine-shaped panel widths to odd lattice widths,
preserving the requested pressure counts and symmetry. It avoids the accuracy
loss from a hierarchy whose middle panels remain large. Its offset table grows
approximately as N², so it is capped by `max_unit_cells`. Prefix-sum entries
whose roundoff allowance exceeds the target are recomputed with direct cell
sums; the report counts these recomputations and their extra work. The error
target is never silently relaxed.

For these midpoint grids the force weight is exactly

```python
Q = np.outer(np.diff(grid.x_panel_edges), np.diff(grid.panel_edges)).ravel()
G = E.T @ scipy.sparse.diags(Q)
```

`B` already integrates the source area. Applying `Q` to its columns a second
time would be incorrect. The legacy native Chebyshev grid is included with its
original nodes and weights as a separate discretization in the benchmark.

For full spectral displacement, use either experimental model through the
existing solver:

```python
from mufsi import CoupledProblem, FrequencyResponseSolver
from mufsi.hydrodynamics.legacy.stokeslet_hybrid import (
    lattice_edge_grid, Stokes3DHybridMultigrid,
)

grid = lattice_edge_grid(geometry, nx=32, ny=48)
hydro = Stokes3DHybridMultigrid(
    fluid, grid, tolerance=1e-5, absolute_tolerance=1e-16, batch_size=128
)
result = FrequencyResponseSolver(CoupledProblem(plate, hydro)).solve(frequencies, load)
```

This is the accuracy setting selected for the measured slender cantilever.
Use 24×36 for faster exploratory sweeps with about 2% difference from the finer
discrete reference. Other geometries and loss-sensitive frequency bands require
their own grid checks.

## Reproducing the benchmark

Use the repository's matched WSL/FEniCSx environment on one MPI rank:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python benchmarks/legacy/benchmark_stokeslet_hybrid.py \
  --output results/hybrid --repeats 3 --resume
```

Individual stages use `--stage integration`, `--stage grid`, or `--stage spectrum`.
`--stage lattice` extends an existing grid study with lattice-aligned cosine
panels without repeating its prior cases.
`--stage refine` extends the spectrum study with 16×48, 24×48 and 32×48 reuse
grids and a tighter 40×60 fluid-reference check. The default `all` runs these
stages in dependency order.
`--stage plot` redraws existing results. Every timing repeat and memory worker
runs in a fresh process, warms its own backend, and uses one BLAS thread.
JSON configurations, source hashes, raw NPZ data, repeated times, integration
path counts, and separate peak-RSS measurements are retained under the output.
`--resume` only reuses a worker result when its entire configuration and source
hash match. RSS includes imports and FEM; growth above a warmed baseline is
reported separately. Timings exclude process startup, backend warmup and export.
The driver freezes source files into a hash-named output subfolder before
launching workers, so source edits cannot change a running study.

1. Fixed panels: two aspect ratios, edge/translation/hierarchical grids,
   1 Hz to 400 kHz, rigid/bending/torsional velocity fields, three tolerances,
   and a crossover sweep. Tight radial results are checked against independent
   tensor-Gauss cubature. Matrix, pressure and generalized resistance errors are
   compared on identical panels and weights.
2. Grid refinement: uniform, native Chebyshev, cosine boundaries at all edges,
   tip-only cosine, uniform x/cosine y, and hierarchical grids. Compare integrated
   forces, moments, loads on common physical subregions, generalized resistance
   and dissipated power. Pressure L² across grids is deliberately omitted because
   edge-singular pressure need not have a finite continuum L² norm.
3. Spectra: fixed full Kirchhoff FE model with a 1 Pa distributed load. All
   methods call the same direct fluid Schur algebra. A pilot locates significant
   wet resonances, then adds a dense window around the first dominant peak to a
   broad frequency sweep. Higher peaks are not certified by linewidth checks. Compare
   complex displacement at 165 common physical probes, tip response, force, power,
   peak amplitude, peak shift, half-power linewidth and Q. Previous-fluid-grid,
   alternate-grid, finer-FE-mesh and frequency-sampling checks quantify reference
   sensitivity. Q here is peak frequency divided by half-power linewidth, not
   a fitted SHO or structural-energy Q.

The provisional target is 1% scaled complex displacement and peak/linewidth/Q
errors, with peak shift below 5% of the reference linewidth. Reference sensitivity
should be about ten times smaller before calling this a verified 1% continuum
result. Passing relative to one discrete reference alone is insufficient.

## Validation

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest tests.hydrodynamics.legacy.test_stokeslet_hybrid -v
```

The tests independently check integration with tensor-Gauss cubature, exact
weights and virtual work, symmetry, direct versus summed-lattice assembly,
pressure solves, invalid lattices, and full coupling against a joint block solve.
An exact SHO solution checks the peak and linewidth measurement, and a tight
relative-only target exercises the prefix precision fallback.
