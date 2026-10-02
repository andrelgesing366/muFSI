# Quick analytic/Quadpy comparison results

Measured on 1 October 2026 using the existing Ubuntu WSL environment, Python
3.14.4, NumPy 2.3.5, SciPy 1.16.3, and legacy-quadpy 0.16.10. Workers used one
BLAS/OpenMP thread, identical grids and symmetry, and relative panel tolerance
`2e-3`. Timing is the median of three trials' mean times per frequency, excluding
warmup and file output. See [the comparison guide](stokes_3d_comparison.md) for
the measurement method and commands.

## Runtime

| Plate and fluid grid | Analytic assembly | Quadpy assembly | Analytic assembly + first pressure | Quadpy assembly + first pressure | Faster total |
| --- | ---: | ---: | ---: | ---: | --- |
| Slender 6x12 | 8.76 ms | 18.36 ms | 8.90 ms | 18.51 ms | Analytic, 2.08x |
| Slender 12x24 | 39.91 ms | 44.88 ms | 41.68 ms | 46.60 ms | Analytic, 1.12x |
| Wide 6x12 | 7.30 ms | 8.59 ms | 7.41 ms | 8.73 ms | Analytic, 1.18x |
| Wide 12x24 | 37.44 ms | 22.07 ms | 39.07 ms | 23.89 ms | Quadpy, 1.64x |
| Notebook slender 24x36 | 128.29 ms | 75.48 ms | 158.18 ms | 104.53 ms | Quadpy, 1.51x |

The first four cases use 1, 10 and 100 kHz. The notebook-grid case also includes
400 kHz. The slender plate is 500x50 micrometres; the wide plate is 500x250
micrometres. All pressure solves use three RHSs: rigid, bending and torsional
velocity fields.

For the 24x36 grid, mean LU times are 29.38 ms for analytic and 27.65 ms for
Quadpy; cached three-RHS solves take about 0.73 ms for both. Both still use the
same dense LU method. The larger performance difference is in panel assembly.

## Memory

| Plate and fluid grid | Analytic process peak RSS | Quadpy process peak RSS | Analytic traced peak | Quadpy traced peak |
| --- | ---: | ---: | ---: | ---: |
| Slender 6x12 | 79.80 MiB | 115.50 MiB | 0.74 MiB | 0.31 MiB |
| Slender 12x24 | 83.04 MiB | 119.29 MiB | 2.80 MiB | 2.69 MiB |
| Wide 6x12 | 79.42 MiB | 115.34 MiB | 0.74 MiB | 0.31 MiB |
| Wide 12x24 | 82.70 MiB | 119.11 MiB | 2.80 MiB | 2.69 MiB |
| Notebook slender 24x36 | 105.91 MiB | 142.13 MiB | 23.45 MiB | 22.97 MiB |

RSS includes Python and loaded native libraries. Quadpy's post-warmup baseline
is higher, accounting for most of the approximately 36 MiB process difference.
On the notebook grid, computation raises the process high-water mark by
27.50 MiB for analytic and 27.07 MiB for Quadpy. The analytic model retains
0.475 MiB of x-separation blocks in addition to the common 11.391 MiB dense
matrix. Thus analytic uses less total process memory in these isolated workers,
but has slightly higher computation allocations. RSS workers have tracing
disabled; allocation peaks come from separate tracemalloc workers.

## Agreement and selection

All 16 frequency/grid/geometry comparisons passed the example's 1% agreement
threshold. Across the full set, maximum differences are:

- Mobility relative Frobenius norm: 0.0123%.
- Significant mobility entry: 0.3215%.
- Pressure relative L2 norm across the three velocity fields: 0.0606%.
- Generalized resistance relative Frobenius norm: 0.0583%.

For the notebook's 24x36 grid specifically, maximum mobility norm and pressure
differences are 0.0058% and 0.0272%. Both models pass the no-slip residual check
at every tested frequency. The automated cross-method test also passes at
relative panel tolerance `2e-5` on tiny slender and wide grids.

For the current 24x36 notebook configuration, prefer **Quadpy when runtime is
the priority**. Analytic is useful for smaller slender grids and for avoiding
the Quadpy dependency and its library-memory overhead. These measurements do
not support a universal analytic speedup. Both implementations already reuse
uniform-x symmetry; this comparison measures their integration strategies under
the same assembly conditions. No implementation or model default was changed
based on the benchmark.

This is a quick fluid-model study, not a coupled FEM spectrum or a grid-convergence
study. Larger grids, other tolerances, and different frequency ranges can change
the ranking. Full trial data and per-frequency timings are saved in:

- `results/stokes_3d_comparison_quick/` for the four small/medium scenarios.
- `results/stokes_3d_comparison_notebook_grid/` for the original notebook grid.

Each directory contains `report.json`, `benchmark.csv`, `comparison.csv`,
`frequency_timings.csv`, `comparison.png`, and raw numerical arrays. The reports
confirm that implementation source hashes stayed unchanged throughout each run.
