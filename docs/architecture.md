# Architecture

The installed package lives under `src/mufsi/`; the Python import name is
`mufsi`. The active formulations are Euler–Bernoulli (EB) and Kirchhoff–Love
(KL) structures with weighted polynomial 3D Stokes loading or 2D section
loading, plus EB with Sader loading. Each supports displacement spectra and
SHO/energy Q. Fluid field recovery currently uses the 2D approximation.

## Responsibilities and interfaces

| Layer | Responsibility | Main entry points |
| --- | --- | --- |
| `models` | Physical data in SI units | `PlateGeometry`, `BeamGeometry`, `Material`, `Fluid` |
| `structure` | FEM mesh, stiffness, mass, and loading | `StructuralModel`, `KirchhoffPlate`, `EulerBernoulliBeam` |
| `hydrodynamics` | Fluid discretization and resisting traction | `FluidGrid`, `HydrodynamicModel`, `Stokes2D`, `Stokes3D`, `SectionForce2D` |
| `coupling` | Structural evaluation and integrated force projection | `build_evaluation_matrix`, `CouplingOperator`, `WeightedCouplingOperator` |
| `solvers` | Linear algebra, coupled response, and dry eigenproblems | `LinearSolver`, `SciPyLUSolver`, `CoupledProblem`, `FrequencyResponseSolver`, `BeamFrequencyResponseSolver`, `EigenSolver` |
| `postprocessing` | Resonance/Q and 2D field recovery | `fit_sho`, `analyze_q_factor`, `energy_from_response`, `reconstruct_flow_from_response` |

The structure layer owns DOLFINx assembly. Coupling uses DOLFINx/Basix to
evaluate structural basis functions and integrate pressure forces; its
assembled operators act on arrays and sparse matrices. Fluid integration is
independent of DOLFINx. FEM libraries are loaded on use; NumPy and SciPy are
runtime dependencies. Coupled fluid response currently requires one MPI rank.

## Coupling contracts

For sampled 2D pressure, `CouplingOperator` uses structural evaluation E and
fluid area weights Q:

```text
E[i,j] = phi_j(fluid_point_i)
fluid motion = E @ u
resisting structural force = E.T @ (Q * pressure)
```

`SectionForce2D` provides an EB line-force contract with rigid transverse
motion. It computes either Sader or numerical Tuck loading.
`BeamFrequencyResponseSolver` projects force per unit length using consistent
beam integration. See [the beam guide](beam_cantilever.md) and
[the 2D plate guide](f2d_spectrum.md) for ordering, units, and boundary points.

Weighted 3D pressure coefficients require a separate force integral:

```text
p(x,y) = sum a[k] * psi_k(x,y)
H a ~= i*omega*E*u
C[j,k] = integral phi_j(x,y)*psi_k(x,y) dA
(K-omega^2*M) u + C a = F
```

`Stokes3D` builds the continuous Chebyshev pressure basis with square-root edge
weights. Cosine coordinates remove those weights; Duffy quadrature treats the
coincident Stokeslet singularity. `weighted_pressure.py` contains the reusable
basis and mobility integration. EB uses x-only structural evaluation and even
transverse pressure degrees; KL uses both coordinates and even/odd degrees.
`WeightedCouplingOperator` integrates C independently of collocation E.
Multiplying sampled singular pressure by collocation areas does not replace
this integral. See [the weighted 3D guide](f3d_spectrum.md).

## Fluid and solver contracts

The common fluid action is
`pressure_from_velocity(omega, velocity)`, with angular frequency in rad/s,
velocity in m/s, and pressure in Pa. The convention is `exp(+i omega t)`;
positive pressure is resisting traction. `assemble_matrix` is optional and
its normalization is model-specific: Stokes2D returns a sampled-pressure
mobility, while Stokes3D returns a rectangular coefficient mobility.

`CoupledProblem.frequency_response(frequencies, load)` accepts frequencies in
Hz and an explicit drive. `FrequencyResponseSolver` retains full structural
DOFs. It uses a sparse block solve for 2D loading and a pressure-coefficient
Schur solve for weighted 3D loading, with a scaled joint fallback at dry poles.
`BeamFrequencyResponseSolver` handles local EB section forces. Linear algebra
interfaces live in `solvers/linear.py`; the available backend is SciPy LU.

`EigenSolver` solves the dry generalized eigenproblem with SLEPc, eliminating
supported displacement DOFs and returning mass-normalized modes and physical
residuals. See [the plate guide](plate_eigenproblem.md). Frequency-dependent
fluid-loaded eigenproblems remain a separate future formulation.

## Q and field recovery

Response objects retain full displacement and model-specific fluid data.
Weighted 3D responses include coefficients, sampled pressure, and integrated
resisting FE forces. SHO Q fits an isolated, resolved displacement resonance;
energy Q uses structural bending-plus-kinetic energy and the model's work per
cycle. Weighted pressure work uses C*a. No stored fluid energy is added.

`reconstruct_flow_from_response` evaluates structural velocity on a separate
2D grid and solves Stokes2D to recover streamfunction, velocity, strain, and
dissipation. This remains a 2D approximation when displacement comes from 3D
or Sader loading. Its dissipation does not replace model-specific work for Q.

## Public code, references, and workflows

The provisional convenience API is listed in `mufsi.__all__`. Specialist
interfaces remain available through their modules. Constant-panel Quadpy,
analytic, multigrid, and hybrid methods and their grid/integration helpers are
preserved under `hydrodynamics/legacy/`; their analytic response adapter lives
under `solvers/legacy/`. Matching examples, benchmarks, documentation, and
tests have `legacy` subfolders. Shared active numerical helpers remain outside
those folders. See [the legacy panel guide](legacy/f3d_spectrum.md).

Only the library package is installed. Examples demonstrate workflows;
`benchmarks/formulation_study.py` performs numerical refinement and measured
comparisons. Examples currently save arrays and metadata directly. Research
contains derivations, experiments, and compatibility imports into promoted
library code; production code must never import from `research/`.
