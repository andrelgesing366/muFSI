# Tests and planned coverage

The folders mirror the library boundaries. `structure/test_kirchhoff.py` and
`solvers/test_eigen.py` cover input validation, analytical plate/beam frequencies,
convergence, physical scaling, matrix symmetry, pressure assembly, constraints,
and modal mass orthogonality. FEM tests require DOLFINx/PETSc/SLEPc; they skip
when those libraries are unavailable. The hydrodynamics tests cover the old
Kelvin formula, batch pressure solves, Sader rigid-section impedance, uniform
beam compliance, and its exact static limit. Coupling tests check basis values
against DOLFINx, virtual work, and block/dense eliminated response agreement.
F3D tests also check the legacy Stokeslet, steady/unsteady singular integrals,
adaptive refinement and failure limits, Quadpy/Gauss agreement, symmetry and
odd grid counts, positive rigid dissipation, and the slender F2D limit. Its
coupled Schur solve is checked against a separate joint dense solve, including
a singular in-vacuo stiffness fallback. Coupled F2D/F3D implementations are
currently serial; those FEM tests skip under MPI.
No old-code regression data is stored yet.

Beam tests check exact cantilever/bridge/simply-supported spectra, mesh
convergence, physical scaling, consistent modal mass, line-load units, static
point/distributed compliance and point-load virtual work. Local-force tests
compare the Tuck force with the legacy cosine-panel formula and Sader loading;
beam response is checked against analytical Sader fields. Beam structural
checks support MPI; driven beam response and arbitrary point loads are serial.

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
