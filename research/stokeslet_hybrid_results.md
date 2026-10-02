# Hybrid Stokeslet benchmark results

Hybrid integration with reuse on a lattice-aligned cosine grid is the best
compromise in this study. The 32×48 setting completes the full 98-frequency FE
sweep in **105 s**, about **5.1 times faster** than direct hybrid integration
on a 32×48 cosine grid (540 s). Its complex tip displacement differs by
**0.60%** from the finer 40×60 calculation, and its first-resonance linewidth Q
differs by **0.061%**. Peak process memory remains about 430 MiB.

These are differences between discrete calculations, not certified continuum
errors. Fluid-grid refinement still changes the response by 0.60%, exceeding
the desired 0.1% reference sensitivity. The structural mesh check changes it
by 0.16%. The numerical agreement supports using the method, while leaving a
strict 1% continuum claim unverified.

![Response, runtime and memory comparison](../results/hybrid/compromise.png)

## Recommended settings

Use `lattice_edge_grid` and `Stokes3DHybridMultigrid` from
[stokeslet_hybrid.py](stokeslet_hybrid.py), with relative integration tolerance
`1e-5`, absolute tolerance `1e-16`, `near_ratio=0.5` and `batch_size=128`.

| Setting | Pressure grid | Median sweep | Peak RSS | Complex tip difference | First-peak Q difference |
|---|---:|---:|---:|---:|---:|
| Exploratory | 24×36 | 37.0 s | 326 MiB | 1.732% | 0.633% |
| Accuracy | 32×48 | 105.4 s | 430 MiB | 0.598% | 0.061% |

Differences use the tighter 40×60 lattice-cosine calculation at tolerance
`1e-7`. Tip difference means `max_f |u-u_ref| / max_f |u_ref|`, including phase;
it is not a pointwise relative error near zeros. Common-probe displacement
differences are 1.730% and 0.598%, respectively. For the accuracy setting,
first-peak amplitude differs by 0.065%, linewidth by 0.020%, and peak frequency
by 0.0031 reference linewidths. The reference peak is near 11.647 kHz with
18.588 nm/Pa displacement and linewidth Q = 3.8077.

## What was implemented

The pressure remains piecewise constant on rectangular panels. Near and self
interactions use the analytic radial primitive with converged numerical angular
integration. Far interactions first try low-order Quadpy cubature and increase
the degree; failures fall back to radial integration. Thus “analytic” here still
has a numerical angular integral.

The direct model supports cosine boundary clustering at all four edges.
The reuse model quantizes the cosine panel widths to odd integer widths on a
common fine lattice. It integrates each distinct unit-cell offset once per
frequency and assembles coarse panel integrals with signed-offset prefix sums.
Precision checks include prefix roundoff; entries that fail the estimate are
summed directly without relaxing the requested tolerance. Conventional odd-width
hierarchies remain available for comparison.

Both models work with the existing full `FrequencyResponseSolver`. Their
structural force weights are the exact panel areas,
`Q = Δx Δy`, giving `G = E.T @ diag(Q)`. The mobility matrix already contains
source integration, so its columns do not receive a second area factor.
The implementation remains in `research`; no weighted screen formulation was
added by this experiment.

## Full spectral comparison

The common test is a 500×50×5 µm isotropic cantilever in water
(`rho=997 kg/m³`, `mu=0.000890 Pa s`, `E=169 GPa`, `rho_s=2330 kg/m³`, `nu=0.3`),
under a uniform 1 Pa harmonic load. All methods solve the same full Kirchhoff
FE system on a 48×6 structural mesh with 2400 free DOFs, using identical fluid
Schur algebra. No modal truncation is used. The saved displacement comparisons
use 165 common physical probes; the solve itself includes every free FE DOF.

The sweep spans 1–400 kHz with 98 samples, including 65 samples in a dense
7.36–20.03 kHz window around the first dominant wet resonance. Peak and
half-power linewidth metrics apply to that first peak. The more sparsely sampled
higher resonances need separate windows before making comparable Q claims.

| Method and grid | Sweep time | Peak RSS | Complex tip difference vs 40×60 | First-peak Q difference |
|---|---:|---:|---:|---:|
| Quadpy, shared cosine 6×12 | 41.3 s | 230 MiB | 13.124% | 12.815% |
| Quadpy, native Chebyshev 12×24 | 95.0 s | 238 MiB | 2.709% | 0.483% |
| Radial, uniform x / cosine y 12×24 | 10.1 s | 198 MiB | 13.074% | 13.444% |
| Direct hybrid, cosine 24×36 | 209.0 s | 300 MiB | 1.765% | 0.572% |
| Hybrid reuse, hierarchy 29×49 | 55.0 s | 386 MiB | 1.562% | 1.248% |
| Hybrid reuse, lattice cosine 24×36 | 37.0 s | 326 MiB | 1.732% | 0.633% |
| Hybrid reuse, lattice cosine 32×48 | 105.4 s | 430 MiB | 0.598% | 0.061% |

“Shared” uses the same panels, nodes and exact area weights as the direct hybrid
case. “Native” retains the original Chebyshev nodes and quadrature weights;
it changes the discretization as well as the integration method. Its agreement
must therefore be assessed by grid refinement, rather than as an integration-only
comparison.

Timing entries are medians of three complete fresh-process sweeps; memory is
measured in a separate fresh process without retaining response arrays. Backend
warmup, process startup and data export are excluded from timing. One BLAS thread
is used. Absolute RSS includes Python, Quadpy and the FE runtime. The direct
32×48 reference was a single 540 s sweep with about 430 MiB peak RSS measured in
its timing worker, rather than the separate memory worker used for candidates.
These measurements are specific to the local WSL environment.

The 32×48 reuse setting spends about 60 s in mobility assembly, versus 495 s for
the direct reference. Structural solves and the dense fluid Schur solve account
for most of the remainder. Reuse accelerates assembly but does not compress the
dense matrices or reduce the asymptotic LU cost. Its fine offset and prefix
tables also consume memory; smaller pressure grids are the practical memory
control. The lattice-size cap rejects oversized grids before allocation.

![Complex displacement differences](../results/hybrid/displacement_error.png)

## Separating integration from discretization

On identical 7×9 hierarchical panels for the slender geometry, the five-frequency
prescribed-motion test at tolerance `1e-5` gives:

| Integration method | Assembly and pressure time | Maximum generalized resistance difference |
|---|---:|---:|
| Quadpy | 1.332 s | 3.94e-8 |
| Radial | 0.330 s | 6.00e-9 |
| Direct hybrid | 0.366 s | 2.35e-8 |
| Original Quadpy multigrid | 0.053 s | 4.57e-9 |
| Hybrid multigrid | 0.018 s | 4.65e-9 |

The reference is tight radial integration checked against independent
tensor-Gauss cubature. This demonstrates integration agreement and reuse speed;
the coarse 7×9 pressure discretization is not a continuum-accurate grid.
Pure radial integration can beat direct hybrid on slender panels, so the
hybrid rule alone is not a universal speed improvement.

The broader grid study covers widths 50 and 250 µm, rigid/bending/torsional
prescribed fields and frequencies through 400 kHz. For the slender geometry,
24×36 direct cosine and lattice-cosine grids differ from the 48×72 pressure
reference by about 0.72% and 0.70% in generalized complex resistance. The lattice
version takes about 0.96 s instead of 8.84 s for the four-frequency pressure run.
Both-edge clustering performs better than the tested uniform-x or tip-only
grids. The native Chebyshev implementation remains useful for independent
discretization checks, but it is not the measured best accuracy/time compromise.

Dissipation converges more slowly than the complex resistance. For the wide
geometry, even 32×48 cosine grids differ from 48×72 by about 36% in the worst
dissipated-power comparison, despite about 1.3% complex resistance differences.
The recommended slender-cantilever settings therefore do not certify damping
at all frequencies or for the wide geometry. Smaller edge panels or a different
pressure representation may be needed there; this experiment does not establish
which next method will resolve that limitation.

## Validation and reproducibility

All 96 repository unit tests passed after implementation. The eight new tests
include independent tensor-Gauss panel checks, force weights and virtual work,
symmetry, direct-versus-lattice matrix comparisons, prefix precision fallback,
pressure solves, invalid controls and a full coupled solve checked against an
independent joint block system. An exact SHO response checks peak and linewidth
measurement. Ruff formatting and lint checks pass for the new Python files.

Frequency half-sampling changes the first-peak Q by about 0.029%; changing the
FE mesh from 48×6 to 64×8 changes scaled complex tip displacement by 0.163%.
Changing the initial direct 32×48 fluid reference to the tighter lattice 40×60
calculation changes it by 0.602%. These checks are retained rather than treating
a small quadrature tolerance as proof of spatial convergence.

Run the scientific environment with one MPI rank:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python research/stokeslet_hybrid.py \
  --stage all --output results/hybrid --repeats 3 --resume
```

The [implementation guide](stokeslet_hybrid.md) describes each stage.
Raw [integration](../results/hybrid/integration.json),
[grid](../results/hybrid/grid.json) and
[spectrum](../results/hybrid/spectrum.json) summaries, configurations, NPZ
responses and per-worker source hashes are retained in `results/hybrid`.
The first integration/grid runs have a saved `source_snapshot`; subsequent
stages use automatic hash-named source snapshots. Per-worker configurations
identify the actual numerical code used. Final API-report/cache validation and
plotting changes do not change the benchmarked numerical algorithm. `--resume`
checks the entire worker configuration and source hash; running with changed
source correctly recomputes affected jobs.
