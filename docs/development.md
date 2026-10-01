# Development

## Current state

The isotropic DOLFINx plate implements mesh/space construction, bending/mass
assembly, support constraints, and distributed pressure loads. The SLEPc
eigen solver and `examples/plate_eigenvalue_problem.py` implement the first
complete structural workflow. F2D, adaptive F3D, the Sader reference, sparse
basis transfer, SciPy LU, and coupled frequency response are also implemented; see
[the F2D guide](f2d_spectrum.md) and [the F3D guide](f3d_spectrum.md).
The Euler-Bernoulli beam, beam eigen example, and local Sader/Tuck loading
are implemented; see [the beam guide](beam_cantilever.md). Postprocessing
and generic I/O remain placeholders.
Plate/fluid/solver controls are validated; physical
containers still only store their data.

Python 3.11 is the initial minimum and `3.0.0.dev0` is a development version.
NumPy and SciPy are runtime dependencies. Install DOLFINx/PETSc/SLEPc in the
scientific environment, as described in [the plate guide](plate_eigenproblem.md).
Matplotlib is optional through the `plot` extra. The optional `quadpy` extra
provides the legacy cubature backend; the NumPy Gauss backend needs no extra.
See the F3D guide for runtime and dependency details.

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

Run the tests:

```console
python -m pytest
python -m ruff check src examples benchmarks
```

The tests use `unittest` and can also be collected by
pytest. Fluid/Sader and input tests run without FEM libraries; FEM tests skip if
the scientific environment is unavailable. See the plate guide for test cases
and run instructions. Skipped tests do not count as numerical validation.

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
