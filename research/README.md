# Research workspace

Weighted pressure basis/mobility and singular quadrature have been promoted to
`src/mufsi/hydrodynamics/weighted_pressure.py` and the canonical `Stokes3D`.
The weighted research import modules now delegate to this library implementation;
existing projection, modal spectrum and convergence drivers remain available.
The hybrid solver classes moved to `src/mufsi/hydrodynamics/legacy/stokeslet_hybrid.py`;
`research/stokeslet_hybrid.py` remains a compatibility/benchmark entry point.
See [the weighted 3D guide](../docs/f3d_spectrum.md) and
`examples/formulation_comparison.py` for the active
full-FE workflow. Historical statements below describe the original experiments.


The [weighted-pressure Kirchhoff-Love plate spectrum](weighted_plate_spectrum.md)
extends the continuous fluid mobility to even and odd transverse polynomials.
Run `weighted_plate_spectrum.py` for cantilever plate displacement and pressure,
or an antisymmetric force pair. It reuses dry KL FEM modes read-only and keeps
the new coupling code, checks and generated results in this research folder.

Use this folder for experimental quadratures, alternative panel algorithms,
operator/compression studies, and profiling scripts. It is not installed with
`mufsi`, and production modules must not import from it.

When promoting an experiment into the package, document its formulation and
scope, add independently meaningful validation, and move the reusable code to
the appropriate library module. Keep measured runtime/memory studies under
`benchmarks/` and reproducible user workflows under `examples/`.

The [hybrid Stokeslet experiment](stokeslet_hybrid.md) implements direct radial /
Quadpy integration on cosine-clustered panels and hybrid integration with
hierarchical lattice reuse. Run `benchmarks/legacy/benchmark_stokeslet_hybrid.py` in the scientific
environment to execute its staged accuracy, grid, spectrum, time and memory
benchmark. `research/stokeslet_hybrid.py` remains a compatible entry point. The implementation remains isolated from production imports.
The [benchmark results](stokeslet_hybrid_results.md) compare the measured
accuracy, runtime and memory and give two recommended grid settings.

The [first-mode pressure polynomial analysis](pressure_polynomial.md) uses the
existing analytic unsteady Stokeslet solver with prescribed Euler–Bernoulli
motion. It compares ordinary and square-root-weighted polynomial pressure
representations, grid refinement, fluid-force errors and corner diagnostics.
Run `research/pressure_polynomial.py`; its reports, figures and numerical data
stay under `research/results/pressure_polynomial/` by default.

The [weighted Stokeslet integration notebook](weighted_stokeslet_analytic.ipynb)
extends the earlier SymPy radial derivation to the weighted Chebyshev pressure
basis. It contains bounded symbolic attempts, a steady elliptic-integral check,
independent numerical quadratures and an optional continuous-density check of
the existing pressure fit. Run locally and save the notebook with its outputs.

The [continuous weighted-pressure mobility experiment](weighted_pressure_mobility.md)
assembles the Stokeslet against a weighted Chebyshev pressure basis using numerical
cosine/Duffy quadrature. Run `weighted_pressure_projection.py` for prescribed
motion, or `weighted_beam_spectrum.py` for a truncated modal Euler-Bernoulli
displacement spectrum. Both remain isolated from production source changes.

The [matched beam spectrum comparison](weighted_beam_comparison.md) compares
the continuous weighted 3D method with Sader and the existing 2D Tuck force
model under identical uniform loading. Run `weighted_beam_comparison.py`; it
includes a pressure-order study, numerical checks and a comparison figure.
