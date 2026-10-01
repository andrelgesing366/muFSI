# Architecture

The installed package lives under `src/mufsi/`. The repository name is `muFSI`;
the Python import name is `mufsi`. The isotropic plate and in-vacuo eigen solver
are implemented, together with F2D, adaptive F3D, Sader, and serial coupled
response.
Euler-Bernoulli beam assembly and local Sader/Tuck line-force response are also
implemented. Other skeleton components reserve interfaces for the rewrite.

## Responsibilities and interfaces

| Layer | Responsibility | Main entry points |
| --- | --- | --- |
| `models` | Physical data in SI units | `PlateGeometry`, `BeamGeometry`, `Material`, `Fluid` |
| `structure` | FEM mesh, stiffness, mass, and loading | `StructuralModel`, `KirchhoffPlate`, `EulerBernoulliBeam` |
| `hydrodynamics` | Fluid discretization and pressure solution | `FluidGrid`, `HydrodynamicModel`, `Stokes2D`, `Stokes3D` |
| `coupling` | Structural basis evaluation and force projection | `build_evaluation_matrix`, `CouplingOperator` |
| `solvers` | Numerical backends and problem orchestration | `LinearSolver`, `CoupledProblem`, `FrequencyResponseSolver`, `EigenSolver` |
| `postprocessing` | Observables and field recovery | `q_factor`, `resonance_frequency`, `evaluate_mode`, `reconstruct_flow` |
| `io` | Validated configuration and result storage | `read_config`, `save_results`, `load_results` |

The models contain data and elementary derived properties. The structure layer
handles DOLFINx assembly. The coupling layer uses DOLFINx/Basix only to
build the basis-evaluation operator. Once built, the coupling operator and the
fluid models will work with arrays and sparse matrices. Optional numerical
FEM libraries are loaded on use; NumPy and SciPy are runtime dependencies.

`SectionForce2D` is the local beam variant of the fluid layer. It returns
resisting force per unit length using rigid transverse motion, with either
Sader or numerical Tuck loading. `BeamFrequencyResponseSolver` projects that
force with the consistent beam line-integration matrix. It has a line-force
contract rather than the plate pressure-grid contract below; see
[the beam guide](beam_cantilever.md).

## Coupling contract

Let `u` contain structural DOFs, and let the rows of sparse `E` correspond to
fluid collocation points:

```text
E[i, j] = phi_j(x_i)
fluid motion = E @ u
resisting force = E.T @ (weights * pressure)
```

The basis-evaluation routine locates cells, maps physical points to reference
coordinates, and evaluates scalar basis functions. It builds columns in serial
structural DOF order, without a second FEM mesh.

Point ordering, constraints, boundary-point behavior, and pressure signs are
specified in [the F2D guide](f2d_spectrum.md). Coupling currently requires one
MPI rank. The implemented coupling describes transverse plate motion; beam/fluid
mapping details will be defined before adding a coupled beam example.

## Fluid contract

The primary operation is:

```python
pressure = hydrodynamics.pressure_from_velocity(omega, velocity)
```

`omega` is angular frequency in rad/s, velocity is in m/s, and pressure is in Pa.
Velocity, pressure, and weights share x-major grid ordering. Positive pressure
is resisting traction with exp(+i omega t). Stokes2D is validated against the
old Kelvin formula and Sader rigid-section impedance.

`assemble_matrix(omega)` is an optional inspection method. The coupled problem
can use the fluid action without requiring a dense hydrodynamic matrix.
F2D additionally provides a sparse mobility for a displacement/pressure block
solve. F3D assembles dense mobility with adaptive regular/singular panel
integration and uses an exact fluid Schur solve, processing structural RHSs
in bounded batches. See [the F3D guide](f3d_spectrum.md) for its tolerance,
grid, memory, and Quadpy/Gauss backends. Backends may reuse factorizations and
process multiple right-hand sides in blocks, or support iterative/operator
approaches. Dense assembly can still be a
useful reference for small problems. Explicit matrix inversion is unnecessary.

The `LinearSolver` abstraction lives in `solvers/linear.py` and can be shared by
fluid and structural implementations. Physics modules do not depend on the
coupled problem or the frequency-response workflow.

## Coupled and eigen solvers

`CoupledProblem` stores the structure, hydrodynamics, and an optional coupling
operator. Its `frequency_response(frequencies, load)` method accepts Hz
and an explicit driving load. The lower-level `FrequencyResponseSolver` handles
conversion to angular frequency and orchestration at each frequency.

`EigenSolver` solves the in-vacuo structural generalized eigenproblem using
SLEPc, eliminating supported displacement DOFs from both matrices and returning
mass-normalized DOLFINx functions and residual errors. See
[the plate guide](plate_eigenproblem.md). A fluid-loaded eigenproblem with
frequency-dependent hydrodynamics requires a separate formulation.

## Public and internal code

The provisional convenience API is listed in `mufsi.__all__`. Specialist
interfaces remain accessible through their modules. Internal panel-integral
functions in `hydrodynamics/stokeslet.py` and `panel_quadrature.py` use leading
underscores so the numerical implementation can evolve without making them
part of the convenience API.
`kernels.py` reserves future general analytical panel reductions.

Only the library package is installed. Tests, examples, benchmarks, and research
scripts remain outside it. Production code must never import from `research/`.
