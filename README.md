# µFSI

Viscous fluid-structure interaction for micro- and nanomechanical resonators.
This repository contains the µFSI v3 rewrite and its planned SoftwareX examples.
The isotropic DOLFINx Kirchhoff–Love plate and in-vacuo SLEPc eigen solver are
implemented, together with F2D and adaptive F3D fluid formulations, sparse
plate/fluid coupling, driven response, and Sader reference. The Euler-Bernoulli
beam, in-vacuo beam eigen workflow and local Sader/Tuck fluid-force models are
also implemented. Postprocessing and generic I/O remain placeholders.

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
│       │   ├── panel_quadrature.py
│       │   ├── stokeslet.py
│       │   ├── stokes_2d.py
│       │   ├── sader.py
│       │   ├── section_force.py
│       │   ├── stokes_3d.py
│       │   └── kernels.py
│       ├── coupling/
│       │   ├── basis_evaluation.py
│       │   └── operator.py
│       ├── solvers/
│       │   ├── problem.py
│       │   ├── linear.py
│       │   ├── frequency_response.py
│       │   ├── beam_frequency_response.py
│       │   └── eigen.py
│       ├── postprocessing/
│       │   ├── qfactor.py
│       │   ├── modes.py
│       │   └── flow.py
│       └── io/
│           ├── config.py
│           └── results.py
├── tests/                 # Unit, integration, and analytical regression tests
├── examples/              # Plate eigen and F2D/F3D/Sader workflows
├── benchmarks/            # Future accuracy, runtime, and memory studies
├── docs/                  # Architecture and development notes
└── research/              # Experiments outside the installed package
```

Each package folder contains an `__init__.py`. The existing root-level
`Test_gpt.py` and `test.py` remain as blank permission-check files.

## Architecture

- `models`: geometry, material, and fluid data in SI units.
- `structure`: Kirchhoff plates and Euler–Bernoulli beams using FEniCSx/DOLFINx.
- `hydrodynamics`: fluid grids, quadrature, Stokeslets, and 2D/3D fluid models.
- `coupling`: structural basis evaluation at fluid points and force projection.
- `solvers`: linear algebra backends, coupled problems, frequency response,
  and structural eigenproblems.
- `postprocessing`: resonance/Q extraction, mode evaluation, and flow recovery.
- `io`: configuration and result persistence.

The structural and coupling implementations own the DOLFINx interaction.
The fluid layer operates on numerical arrays independently of DOLFINx.
The primary fluid interface is `pressure_from_velocity(omega, velocity)`;
dense matrix assembly is optional. Linear algebra backends will remain separate
from the physics. CPU execution is the initial target.

See [the architecture notes](docs/architecture.md) for the interfaces and
[the development notes](docs/development.md) for the implementation sequence.

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

The adaptive F3D workflow is described in [the 3D fluid guide](docs/f3d_spectrum.md).
Run `examples/plate_3d.py --quick` for a 6x12 fluid grid, or use the default
12x24 grid to compare slender and wide plates. The script caps fluid counts at
32x64 and saves both spectra, complex fields and quadrature reports. Install
the optional `quadpy` extra for the legacy cubature rules, or select
`--quadrature gauss` for NumPy quadrature.

The [beam guide](docs/beam_cantilever.md) describes the port from the 1D
cantilever folders. Run `examples/beam_eigenvalue_problem.py --plot` for six
vacuum modes with analytical comparison, and `examples/beam_2d.py` for FEM
with local Sader/Tuck forces. The latter uses 64 transverse panels by default;
`--quick` uses 16. These beam examples do not require Quadpy.

Other examples and benchmarks remain placeholders. Tests cover structural
analytical results, the F2D kernel, Sader compliance, basis transfer, virtual
work, adaptive regular/singular F3D panels, and coupled block/Schur solves.

Only `src/mufsi/` is included in the installed package. The public API is exposed
through `mufsi.__init__` and remains provisional during the rewrite.

## License and citation

The repository uses the existing [Apache License 2.0](LICENSE). Publication
metadata and a `CITATION.cff` will be added when the authors, release information,
and paper details are confirmed.
