# µFSI

Viscous fluid-structure interaction for micro- and nanomechanical resonators.
This repository contains the µFSI v3 rewrite and its planned SoftwareX examples.
The isotropic DOLFINx Kirchhoff–Love plate and in-vacuo SLEPc eigen solver are
implemented. Fluid models, coupled response, and other planned numerical
components remain explicit `NotImplementedError` placeholders.

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
│       │   ├── stokes_3d.py
│       │   └── kernels.py
│       ├── coupling/
│       │   ├── basis_evaluation.py
│       │   └── operator.py
│       ├── solvers/
│       │   ├── problem.py
│       │   ├── linear.py
│       │   ├── frequency_response.py
│       │   └── eigen.py
│       ├── postprocessing/
│       │   ├── qfactor.py
│       │   ├── modes.py
│       │   └── flow.py
│       └── io/
│           ├── config.py
│           └── results.py
├── tests/                 # Future unit, integration, and regression tests
├── examples/              # Future user workflows and SoftwareX examples
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

The structural and coupling implementations will own the DOLFINx interaction.
The fluid layer will operate on numerical arrays independently of DOLFINx.
The primary fluid interface is `pressure_from_velocity(omega, velocity)`;
dense matrix assembly is optional. Linear algebra backends will remain separate
from the physics. CPU execution is the initial target.

See [the architecture notes](docs/architecture.md) for the interfaces and
[the development notes](docs/development.md) for the implementation sequence.

## Working with the skeleton

With Python 3.11 or newer, install the package from this repository:

```console
python -m pip install -e ".[dev]"
```

Numerical backend imports are deferred until they are used. NumPy is a package
dependency; the plate workflow additionally needs a matched DOLFINx/PETSc/SLEPc
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
The other example and benchmark files remain placeholders. Tests now cover
plate input validation and analytical structural/eigenproblem checks.

Only `src/mufsi/` is included in the installed package. The public API is exposed
through `mufsi.__init__` and remains provisional during the rewrite.

## License and citation

The repository uses the existing [Apache License 2.0](LICENSE). Publication
metadata and a `CITATION.cff` will be added when the authors, release information,
and paper details are confirmed.
