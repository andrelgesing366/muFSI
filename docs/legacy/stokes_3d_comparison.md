> Legacy reference: these panel implementations are now under
> `mufsi.hydrodynamics.legacy`. The active `mufsi.Stokes3D` is documented in
> [the weighted-pressure guide](../f3d_spectrum.md).

# Analytic F3D versus Quadpy F3D

The preserved example `examples/legacy/compare_stokes_3d.py` compares `Stokes3DAnalytic`
with `Stokes3D(quadrature_backend="quadpy")`. The benchmark entry point
`benchmarks/legacy/compare_stokes_3d.py` runs the same workflow. Neither existing model
nor the other examples are modified.

The completed quick runs and model-selection recommendation are recorded in
[the benchmark results](stokes_3d_benchmark_results.md).

Run from the repository in the existing scientific Python environment:

```console
PYTHONPATH=src .venv/bin/python examples/legacy/compare_stokes_3d.py --quick --plot
```

The quick run uses 6x12 and 12x24 fluid grids, frequencies 1, 10, and 100 kHz,
and three independent timing trials per model and geometry. It considers both
the 500x50 micrometre slender plate and the 500x250 micrometre wide plate in water
(`rho=997 kg/m^3`, `mu=890e-6 Pa*s`). Both models share the exact same uniform-x,
Chebyshev-y grid, pressure/velocity ordering, force weights, symmetry settings,
batch size, and `2e-3` relative integration tolerance. They both reuse x
translation and y reflection, so this isolates the change in integration method.

An individual configuration or a tighter accuracy check can be selected:

```console
PYTHONPATH=src .venv/bin/python benchmarks/legacy/compare_stokes_3d.py --case slender --grids 12x24 --frequencies 1000 10000 100000 --repeats 5 --tolerance 1e-4 --agreement 1e-3 --output results/stokes_3d_comparison_tight
```

## Numerical comparisons

For each frequency the example compares the complex mobility matrix, pressures
for rigid/bending/torsional velocity fields, integrated rigid force, and
generalized resistance `V.conj().T @ Q @ P`. The torsional field is antisymmetric
in y; its net force is approximately zero, so generalized resistance is a better
comparison than a relative error in its net force. The pressure solves are also
checked for no-slip residuals below `1e-10`.

`comparison.csv` contains the relative Frobenius mobility difference, maximum
relative significant-entry difference, each pressure field's relative L2
difference, relative generalized resistance difference, and rigid-force
difference. The program exits nonzero if any exceeds `--agreement`, if either
integrator fails, or if source hashes change while the benchmark is running.
The default agreement target is 1%. Quadpy is the comparison baseline, rather
than an exact solution; agreement between methods does not establish fluid-grid
convergence.

The small automated cross-method test uses a tighter `2e-5` integration target
on 3x5 grids for both geometries, at 1 and 100 kHz:

```console
PYTHONPATH=src .venv/bin/python -m unittest tests.hydrodynamics.legacy.test_stokes_3d_comparison -v
```

## Time and memory measurements

Every timing trial uses a fresh child process. A small 2x3 problem warms imports,
quadrature rules and BLAS outside the measurement. Assembly is then measured
with an empty model cache at each frequency. LU factorization, the first pressure
solve, and the mean of 100 cached pressure solves are recorded separately. Each
pressure solve has three RHSs. Model order alternates between repetitions.
Workers use one BLAS/OpenMP thread; startup, file serialization, plotting, and
warmup are excluded from numerical timings.

`frequency_timings.csv` gives medians across trials at each frequency.
`benchmark.csv` summarizes the median across trials of each trial's mean time
per frequency, with assembly and assembly-plus-first-pressure speedups. Both
models still use the same dense LU method; an integration speedup need not
translate into the same speedup for a large LU-dominated problem.

Memory uses additional fresh workers. One runs without tracing and reports OS
process high-water RSS, including the interpreter and native libraries, and
the increase above the post-warmup high-water mark. Another uses `tracemalloc`
to measure peak Python and registered NumPy allocations after warmup. Traced
peaks omit some native allocations; an RSS growth of zero means the computation
did not exceed the earlier high-water mark, rather than zero memory use. Keeping
the RSS and tracing workers separate avoids counting tracing overhead as the
model's native memory requirement. Matrix and cached-block byte counts are also
recorded. Dense matrix storage is identical for the same grid; the analytic
model additionally retains its smaller x-separation blocks.

`report.json` preserves individual trials, integration reports, versions,
thread settings, parameters, and SHA-256 hashes of the implementations.
`raw/` contains per-model numerical arrays and raw timing/memory measurements.
`comparison.png` is created with `--plot`.

This quick benchmark covers fluid assembly and pressure solves. It does not run
a coupled FEM spectrum, establish performance at large grid sizes, or compare
the nonuniform-x mode that only the Quadpy implementation currently supports.
