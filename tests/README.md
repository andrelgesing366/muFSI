# Tests and planned coverage

The folders mirror the library boundaries. `structure/test_kirchhoff.py` and
`solvers/test_eigen.py` cover input validation, analytical plate/beam frequencies,
convergence, physical scaling, matrix symmetry, pressure assembly, constraints,
and modal mass orthogonality. FEM tests require DOLFINx/PETSc/SLEPc; they skip
when those libraries are unavailable. No old-code regression data is stored yet.

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

Add tests as numerical components are implemented. The current tests use
`unittest`, so they run without pytest and are also collected by it. A skipped
scientific test is not a passing numerical check.
