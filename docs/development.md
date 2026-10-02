# Development

## Current state

The DOLFINx Euler–Bernoulli beam and Kirchhoff–Love plate implement structural
assembly, constraints, point/distributed loads, and dry SLEPc eigenproblems.
Weighted polynomial 3D Stokes loading and 2D loading support both structures;
Sader section loading supports EB. Full-DOF coupled response, SHO/energy Q,
and 2D field recovery are implemented. See [the architecture](architecture.md),
[the weighted 3D guide](f3d_spectrum.md), [the 2D guide](f2d_spectrum.md), and
[the beam guide](beam_cantilever.md).

`examples/formulation_comparison.py` provides a compact comparison of all five
active combinations. The [formulation study](formulation_study.md) covers finer
spectra, frequency/Q comparisons, antisymmetric KL motion, discretization
checks, independent timing samples, and peak process memory. Saved arrays and
metadata are written by the workflows directly.

Published constant-panel algorithms remain under `hydrodynamics/legacy/` and
`solvers/legacy/`, with matching example, benchmark, documentation, and test
folders. Weighted research modules delegate to promoted library code while
retaining the original experimental drivers and reports.

Python 3.11 is the minimum and `3.0.0.dev0` is a development version. NumPy and
SciPy are runtime dependencies. Install matched DOLFINx/PETSc/SLEPc separately
as described in [the plate guide](plate_eigenproblem.md). Matplotlib is optional
through the `plot` extra; `quadpy` provides the legacy cubature backend. The
active weighted quadrature defaults to NumPy Gauss rules.

## Verification

Install development tools in a suitable Python environment:

```console
python -m pip install -e ".[dev]"
```

From the repository root, run:

```console
PYTHONPATH=src python -m unittest discover -s tests -t . -v
python -m ruff check src examples benchmarks tests research
```

The tests use `unittest` and can also be collected with `python -m pytest`.
Fluid, coupling algebra, and many postprocessing checks run without FEM.
Scientific tests skip if the matched FEM environment is unavailable; skipped
tests do not establish numerical validation. See [the test guide](../tests/README.md).

Tests should check numerical behavior against independent formulations,
analytical results, or physical identities. Integration tests must state their
environment requirements. Stored regression data need traceable provenance,
units, tolerances, and generation instructions.

## Examples, benchmarks, and research

Examples should demonstrate reproducible workflows and state required
dependencies, observables, and units. Benchmarks should record problem sizes,
solver options, environment, accuracy, runtime, and memory with a clear
measurement method. Keep published-method comparisons in `legacy` subfolders.

Use `research/` for exploratory quadratures, compression, profiling, and
derivations. Promote validated reusable algorithms into the library after
their interface and independent checks are clear. Preserve research evidence
and historical numerical results; production modules must not import research.

## Deferred roadmap

- Generic configuration/result I/O: design validation and a versioned schema
  for complex arrays, units, and metadata before adding a public API.
- Particle loading/detection: add a defined particle model, loading or mass
  perturbation, and a runnable sensing example.
- PETSc coupled-solver backend: implement and validate it when parallel or
  larger coupled problems require it. Existing FEM assembly/eigenproblems
  already use PETSc; this concerns the reusable response backend.
- Extended scaling studies: add systematic size/thread/MPI scaling if needed
  beyond the implemented formulation timing, refinement, and memory study.
- Full 3D field recovery, fluid-loaded eigenproblems, physical-container input
  validation, and confirmed citation/release metadata remain future work.

These are planned capabilities, without placeholder modules or empty scripts
in the active implementation.
