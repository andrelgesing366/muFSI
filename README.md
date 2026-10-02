# µFSI

Viscous fluid-structure interaction for micro- and nanomechanical resonators.
This repository contains the µFSI v3 rewrite and its planned SoftwareX examples.
The package implements DOLFINx Euler-Bernoulli beams and Kirchhoff-Love plates,
dry eigenproblems, weighted polynomial 3D Stokes loading for both structures,
2D section loading, and Sader loading for beams. SHO and energy Q postprocessing
work with each active combination. Field recovery currently uses the 2D
approximation, including for displacement obtained with 3D/Sader loading.
Examples save their numerical arrays and metadata directly; a generic I/O API
is deferred.

Run `examples/formulation_comparison.py` in the matched scientific environment
for compact displacement spectra and Q-versus-frequency plots covering all five
active combinations. See [the 3D/Q guide](docs/f3d_spectrum.md). Previous published
panel methods remain available in `mufsi.hydrodynamics.legacy`.

## Repository layout

```text
muFSI/
├── pyproject.toml
├── LICENSE
├── README.md
├── src/
│   └── mufsi/
│       ├── __init__.py
│       ├── models/
│       │   ├── geometry.py
│       │   ├── material.py
│       │   └── fluid.py
│       ├── structure/
│       │   ├── base.py
│       │   ├── kirchhoff.py
│       │   ├── euler_bernoulli.py
│       │   └── loads.py
│       ├── hydrodynamics/
│       │   ├── base.py
│       │   ├── grid.py
│       │   ├── quadrature.py
│       │   ├── stokeslet.py
│       │   ├── stokes_2d.py
│       │   ├── sader.py
│       │   ├── section_force.py
│       │   ├── stokes_3d.py
│       │   ├── weighted_pressure.py
│       │   └── legacy/          # Published constant-panel models and helpers
│       ├── coupling/
│       │   ├── basis_evaluation.py
│       │   ├── operator.py
│       │   └── weighted.py
│       ├── solvers/
│       │   ├── problem.py
│       │   ├── linear.py
│       │   ├── frequency_response.py
│       │   ├── beam_frequency_response.py
│       │   ├── eigen.py
│       │   └── legacy/          # Analytic panel response adapter
│       └── postprocessing/
│           ├── qfactor.py
│           └── flow.py
├── tests/                 # Unit, FEM, and analytical checks; legacy subfolders
├── examples/              # EB/KL eigen, spectra, Q and flow; legacy subfolder
├── benchmarks/            # Formulation study and preserved legacy benchmarks
├── docs/                  # Current guides and legacy reference guides
└── research/              # Experiments outside the installed package
```

Each package folder contains an `__init__.py`.

## Architecture

- `models`: geometry, material, and fluid data in SI units.
- `structure`: Kirchhoff plates and Euler–Bernoulli beams using FEniCSx/DOLFINx.
- `hydrodynamics`: fluid grids, quadrature, Stokeslets, and 2D/3D fluid models.
- `coupling`: structural basis evaluation at fluid points and force projection.
- `solvers`: linear algebra backends, coupled problems, frequency response,
  and structural eigenproblems.
- `postprocessing`: resonance/Q extraction and 2D flow recovery.

The structural and coupling implementations own the DOLFINx interaction.
The fluid layer operates on numerical arrays independently of DOLFINx.
The primary fluid interface is `pressure_from_velocity(omega, velocity)`;
dense matrix assembly is optional. Linear algebra backends will remain separate
from the physics. CPU execution is the initial target.

See [the architecture notes](docs/architecture.md) for the interfaces and
[the development notes](docs/development.md) for the implementation sequence.
The preserved [legacy padded multigrid F3D method](docs/legacy/stokeslet_multigrid.md) provides
edge-refined pressure panels, exact cell-area coupling weights, and runnable
pressure/spectrum comparisons against analytic and Quadpy integration.

## Running the library

With Python 3.11 or newer, install the package from this repository:

```console
python -m pip install -e ".[dev]"
```

FEM backend imports are deferred until they are used. NumPy and SciPy are package
dependencies; the eigen workflow needs a matched DOLFINx/PETSc/SLEPc
environment. Installing µFSI alone does not install that scientific runtime.

The geometry and physical parameter containers can already be constructed:

```python
from mufsi import Fluid, Material, PlateGeometry

geometry = PlateGeometry(length=800e-6, width=100e-6, thickness=5e-6)
material = Material(young_modulus=169e9, density=2330.0, poisson_ratio=0.064)
fluid = Fluid(density=997.0, dynamic_viscosity=890e-6)
```

These containers currently store data; input validation is a later step.
The plate eigen workflow is described in [the plate guide](docs/plate_eigenproblem.md).
Run `examples/plate_eigenvalue_problem.py` inside the scientific environment to
print eigenfrequencies and export mode shapes; `--plot` saves a mode-shape figure.
It uses the same silicon plate geometry and material as the old Example_1 notebook.
The F2D/Sader workflow is described in [the fluid guide](docs/f2d_spectrum.md).
Run `examples/cantilever_2d.py` in the scientific environment to compare the
full 200-frequency spectrum and export complex fields, CSV data, and a plot.
`--quick` uses a smaller grid and mesh. Coupled fluid solves currently use one
MPI rank. Install the `plot` extra for plotting.

The weighted-pressure F3D workflow is described in [the 3D fluid guide](docs/f3d_spectrum.md).
Run `examples/formulation_comparison.py` for a compact EB/KL comparison,
including displacement spectra, Q factors, and a 2D field reconstruction.
`examples/plate_3d.py` selects the two KL formulations. Published panel methods
and their original broad spectrum example are retained under `legacy`.

The [extensive study guide](docs/formulation_study.md) describes the finer
frequency comparisons for 800×50, 500×50, and 500×250 µm cantilevers.
`benchmarks/formulation_study.py` runs spectra, numerical refinement,
antisymmetric plate loading, and independent runtime benchmarks.
`examples/formulation_spectra.py` regenerates the PNG/PDF figures and report
from saved results without rerunning the expensive solves.

The [beam guide](docs/beam_cantilever.md) describes the port from the 1D
cantilever folders. Run `examples/beam_eigenvalue_problem.py --plot` for six
vacuum modes with analytical comparison, and `examples/beam_2d.py` for FEM
with local Sader/Tuck forces. The latter uses 64 transverse panels by default;
`--quick` uses 16. These beam examples do not require Quadpy.

The [Q-factor guide](docs/qfactor_2d.md) describes symmetric/antisymmetric
corner excitation, SHO fits, energy Q and the Sader reference. Run
`examples/qfactor_2d.py` for the first three flexural resonances of a slender
silicon cantilever in water, or use `--quick` for one resonance. It saves
comparison data, complex responses and a figure.

The [2D flow guide](docs/flow_2d.md) describes field recovery and phase plots.
Run `examples/flow_visualization.py` for a prescribed roof-tile displacement,
or pass `--response results/cantilever_2d/response.npz --frequency 100000`
to postprocess a saved F2D pressure solution without rerunning FEM.

Tests cover structural analytical results, the F2D kernel, Sader compliance,
weighted coefficient mobility and force projection, basis transfer, virtual
work, Q and field recovery, and coupled block/Schur solves. Legacy subfolders
retain the constant-panel integration and response checks. See
[the test guide](tests/README.md) and [the development roadmap](docs/development.md).

Only `src/mufsi/` is included in the installed package. The public API is exposed
through `mufsi.__init__` and remains provisional during the rewrite.

## License and citation

The repository uses the existing [Apache License 2.0](LICENSE). Publication
metadata and a `CITATION.cff` will be added when the authors, release information,
and paper details are confirmed.
