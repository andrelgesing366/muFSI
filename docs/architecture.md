# Architecture

The installed package lives under `src/mufsi/`. The repository name is `muFSI`;
the Python import name is `mufsi`. The isotropic plate and in-vacuo eigen solver
are implemented; the remaining skeleton reserves interfaces for the rewrite.

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
will handle DOLFINx assembly. The coupling layer will use DOLFINx/Basix only to
build the basis-evaluation operator. Once built, the coupling operator and the
fluid models will work with arrays and sparse matrices. Optional numerical
libraries must be loaded at the point of use rather than during package import.

## Coupling contract

Let `u` contain structural DOFs, and let the rows of sparse `E` correspond to
fluid collocation points:

```text
E[i, j] = phi_j(x_i)
fluid motion = E @ u
structural force = E.T @ (weights * pressure)
```

The basis-evaluation routine will locate structural cells, map physical points
to reference coordinates, evaluate basis functions, and assign the correct
global columns. It will not construct a second FEM mesh for the fluid grid.

Point ordering, constrained DOFs, points on cell boundaries, pressure signs,
and MPI ownership must be documented and validated with this implementation.
The current skeleton describes the scalar transverse plate problem; beam/fluid
mapping details will be defined before adding a coupled beam example.

## Fluid contract

The primary operation is:

```python
pressure = hydrodynamics.pressure_from_velocity(omega, velocity)
```

`omega` is angular frequency in rad/s, velocity is in m/s, and pressure is in Pa.
Velocity, pressure, and weights share the fluid grid ordering. Kernel signs and
normalization will be checked against the reference implementation before this
contract becomes stable.

`assemble_matrix(omega)` is an optional inspection method. The coupled problem
must work through the fluid action rather than require a dense hydrodynamic
matrix. Backends may reuse factorizations, process multiple right-hand sides in
blocks, or support iterative/operator approaches. Dense assembly can still be a
useful reference for small problems. Explicit matrix inversion is unnecessary.

The `LinearSolver` abstraction lives in `solvers/linear.py` and can be shared by
fluid and structural implementations. Physics modules do not depend on the
coupled problem or the frequency-response workflow.

## Coupled and eigen solvers

`CoupledProblem` stores the structure, hydrodynamics, and an optional coupling
operator. Its future `frequency_response(frequencies, load)` method accepts Hz
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
functions in `hydrodynamics/kernels.py` use leading underscores so the numerical
implementation can evolve without making them part of the convenience API.

Only the library package is installed. Tests, examples, benchmarks, and research
scripts remain outside it. Production code must never import from `research/`.
