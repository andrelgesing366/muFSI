# Planned tests

The folders mirror the library boundaries. No tests or expected numerical
results have been added yet.

- `models/`: input validation, units, and derived physical properties.
- `structure/`: assembly, boundary conditions, point/distributed loads, and
  independent in-vacuo beam/plate benchmarks.
- `hydrodynamics/`: grid ordering/weights, limiting kernels, singular panels,
  and 2D/3D pressure solves.
- `coupling/`: exact interpolation of representable fields, projection/virtual
  work consistency, cell-edge points, and old/new transfer comparisons.
- `solvers/`: residuals and known coupled responses, plus backend agreement.
- `postprocessing/`: resonance/Q recovery from independently generated curves.
- `io/`: complex-data round trips, metadata, and schema compatibility.
- `data/regression/`: selected reference outputs with environment, generation
  instructions, provenance, units, and tolerances.

Add tests as numerical components are implemented. Use the `integration` marker
for tests needing DOLFINx and the `regression` marker for reference comparisons.
