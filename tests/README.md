# Tests

The folders follow the implemented library boundaries. Run the full suite from
the repository root inside the matched scientific environment:

```console
PYTHONPATH=src python -m unittest discover -s tests -t . -v
```

The tests use `unittest` and are also collected by pytest. FEM tests require
DOLFINx/PETSc/SLEPc and skip when unavailable. Coupled fluid response and
arbitrary point loading are serial; beam structural checks also support MPI.
A skipped scientific test is not a passing numerical check.

| Folder | Implemented checks |
| --- | --- |
| `structure/` | Beam/plate spectra against analytical limits, mesh convergence, physical scaling, matrix symmetry, loads, constraints, modal mass and orthogonality |
| `hydrodynamics/` | F2D Kelvin kernel and pressure solves, Sader/section-force compliance and static limits, weighted EB/KL coefficient recovery and degree selection |
| `coupling/` | Basis evaluation, representable-field interpolation, virtual work, boundary/cell-edge behavior, and coupled block response |
| `solvers/` | Eigen residuals, beam response against analytical Sader fields, frequency-response validation and backend behavior |
| `postprocessing/` | Independently generated SHO resonances, energy/work balance, FEM Q, and 2D velocity/strain/dissipation recovery |
| `hydrodynamics/legacy/` | Constant-panel kernels, singular/regular integration, Quadpy/Gauss and analytic agreement, refinement failure limits, multigrid/hybrid assembly, symmetry, pressure batches and dissipation |
| `solvers/legacy/` | Analytic/multigrid panel response against separate joint systems, including dry-pole fallback |

`hydrodynamics/test_stokes_3d.py` also checks weighted force projection against
independent integration, the coefficient Schur solution against a joint solve,
full-FE virtual work and fluid-work consistency, EB/KL responses, and 2D field
recovery from weighted 3D displacement. Existing weighted research checks
remain under `research/` with their experiment drivers.

There are no stored old-code regression fixtures yet. Add those only with
documented provenance, generation environment, units, and tolerances. Generic
I/O round-trip tests and particle-detection checks belong with their future
implementations; see [the development roadmap](../docs/development.md).
