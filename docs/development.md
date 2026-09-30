# Development

## Current state

This is an importable package skeleton. Physical input/result containers and
class signatures exist. Numerical methods, grid factories, postprocessing, and
I/O explicitly raise `NotImplementedError`. `Fluid.kinematic_viscosity` is a
simple derived data property. Physical input validation is not implemented yet.

Python 3.11 is the initial minimum and `3.0.0.dev0` is a development version.
The skeleton has no third-party runtime dependencies. Add numerical packages
and supported version ranges when their first implementation is introduced.
DOLFINx/PETSc/SLEPc environment setup will be documented and verified separately.

## Suggested implementation sequence

1. Establish input validation, units, harmonic convention, and reference data
   from the old implementation. Record reference environment and tolerances.
2. Implement the plate mesh, stiffness, mass, and boundary conditions in DOLFINx.
   Check structural eigenfrequencies and convergence against independent cases.
3. Implement fluid grids and sparse basis evaluation. Verify interpolation,
   force projection, point loads, cell-edge behavior, and constrained DOFs.
4. Implement and validate the Stokeslet, panel integration, and the 2D/3D
   hydrodynamic pressure action. Check regular and singular panels separately.
5. Implement reusable linear solvers and coupled frequency response. Measure
   residuals, agreement with reference results, runtime, and peak memory.
6. Add postprocessing, persistent results, runnable examples, and benchmark
   studies for the SoftwareX manuscript. Finalize citation and release metadata.

## Verification

Install development tools in a suitable Python environment:

```console
python -m pip install -e ".[dev]"
```

Once meaningful tests are added:

```console
python -m pytest
python -m ruff check src examples benchmarks
```

Currently `tests/` contains only directories and planning notes. Running pytest
will report that no tests were collected; that is not numerical validation.
Until implementations exist, checking syntax and importing every package module
is sufficient to verify the scaffold itself.

Add unit tests with their implementation. Integration tests should state their
scientific-environment requirements. Regression fixtures must have traceable
provenance; do not populate expected results by running the new implementation
and comparing it against itself.

## Examples, benchmarks, and experiments

Example scripts should demonstrate small, reproducible user workflows and state
the required environment, expected observables, and relevant units.
Benchmark scripts should record problem sizes, solver options, environment,
runtime, accuracy, and memory with a clear measurement method.

Use `research/` for exploratory algorithms and profiling experiments. Promote
validated algorithms into the library after their interface and tests are clear.
Keep the original code as a reference outside this installed package.
