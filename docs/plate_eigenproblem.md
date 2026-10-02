# Isotropic plate eigenproblem

This guide covers the in-vacuo homogeneous isotropic Kirchhoff–Love plate.
For the implemented coupled frequency response, see the
[2D loading guide](f2d_spectrum.md) and
[weighted 3D loading guide](f3d_spectrum.md).

## Formulation

The physical domain is `[0, L] x [-W/2, W/2]`, matching the original code.
For Hessian `H(w)`, the bending moment tensor and surface density are

```text
D = E t^3 / (12 (1 - nu^2))
moment(w) = D ((1 - nu) H(w) + nu tr(H(w)) I)
surface_density = rho t
```

The stiffness contains the element bending energy, symmetric interior-facet
consistency terms, and a penalty on jumps of normal slope. Clamped boundary
facets have analogous consistency and penalty terms for zero normal slope.
The penalty coefficient is `penalty * D / h`, with the mean cell diameter on
interior facets. The default dimensionless penalty is 16, matching the legacy
P2 implementation. For higher polynomial degrees, verify stability and
convergence and tune the penalty; 16 is not a universal stability guarantee.

Continuous Lagrange elements of degree at least two are used on triangular
meshes. The default crossed diagonal matches the original meshing choice.
Displacement boundary conditions are enforced by elimination during solution.

| Boundary condition | Zero displacement | Weak zero slope | Remaining edges |
| --- | --- | --- | --- |
| `cantilever` | x=0 | x=0 | Free |
| `bridge` | x=0 and x=L | x=0 and x=L | Free |
| `clamped` | All four edges | All four edges | None |
| `simply_supported` | All four edges | None | Natural zero normal moment |

The isotropic moment tensor is retained in the facet terms, which is important
for correct Poisson coupling and free-edge behavior. No anisotropic material
tensor is implemented.

## Eigen solver and result

`EigenSolver` restricts both the unmodified stiffness and mass matrices to
unconstrained displacement DOFs, then solves

```text
K_free u = lambda M_free u
omega = sqrt(lambda)
frequency_Hz = omega / (2 pi)
```

The SLEPc problem is generalized Hermitian. Krylov–Schur with a zero-shift
shift-invert transformation targets the lowest eigenvalues. The reduced
stiffness is factored by PETSc LU. `factor_solver_type` can select an available
parallel backend such as MUMPS. SLEPc/PETSc options use the `mufsi_eigen_` prefix.

Both reduced matrices are scaled by the same mass-based factor before solving.
This preserves the spectrum and avoids convergence estimates dominated by tiny
mass entries at micrometre scales. Modal mass normalization uses the original
physical mass. The reported residual is the dimensionless force-balance error
`||K u - lambda M u|| / (||K u|| + |lambda| ||M u||)` on the free DOFs.

Insufficient convergence or invalid eigenvalues cause explicit errors.
The result contains ascending frequencies, mass-normalized modes, DOLFINx mode
functions, relative eigenpair residuals, and iteration count. Constrained
entries are restored as zero, and ghost values are synchronized. In serial the
mode array is complete; under MPI it contains only the local owned/ghost DOFs.
It is not an automatically gathered global array.

Plate configuration is immutable and FEM setup is lazy. New matrix calls return
caller-owned PETSc matrices; the eigen solver destroys its temporary matrices,
index sets, vectors, and SLEPc solver. Returned DOLFINx Functions remain usable.

Distributed transverse pressure loading is implemented by interpolating
the load callable into the plate space. Serial `PointLoad` and `PointLoads`
use the shared arbitrary-point basis evaluator and preserve virtual work.
The [Q-factor guide](qfactor_2d.md) shows corner excitation and postprocessing.

## Environment and example

The API targets DOLFINx 0.10.x. Use a matched installation of DOLFINx, UFL,
Basix, FFCx, MPI, PETSc/petsc4py, and SLEPc/slepc4py. NumPy is a package dependency.
FEniCSx is not supplied by installing the legacy `fenics` pip package.
See the [official installation guidance](https://docs.fenicsproject.org/dolfinx/v0.10.0/python/installation.html).

From the repository root inside that environment:

```console
python3 -m pip install -e .
python3 examples/plate_eigenvalue_problem.py
python3 examples/plate_eigenvalue_problem.py --nx 32 --ny 32 --modes 6 --plot
python3 examples/plate_eigenvalue_problem.py --boundary-condition bridge
```

Plotting additionally requires Matplotlib (available via the `plot` extra).
Alternatively, use `PYTHONPATH=src python3 examples/plate_eigenvalue_problem.py`
from the repository root without an editable installation.
The example reuses the original notebook's 500 x 500 x 5 micrometre plate,
E=169 GPa, rho=2330 kg/m^3, nu=0.3, 64 x 64 crossed mesh, and ten modes.
It prints frequencies/residuals and writes:

- `frequencies.csv`: mode number, Hz, rad/s, and relative residual.
- `mode_shapes.xdmf` plus HDF5 data: peak-normalized P1 interpolation for
  visualization; the frequency in Hz is used as the visualization time value.
- `eigenmodes.npz` in serial: original mass-normalized modes and FEM DOF
  coordinates, together with the frequencies and residuals.
- `mode_shapes.png` with `--plot`: a serial, noninteractive mode-shape figure.

The P1 visualization does not preserve all higher-order FEM information; use
the original mode arrays/functions for numerical work.

## Verification

```console
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The tests include an exact simply supported rectangular plate spectrum with
mesh refinement, cantilever/bridge beam-limit frequencies at nu=0, thickness
and density scaling, matrix symmetry, distributed pressure assembly, supported
DOF elimination, modal mass normalization, and orthogonality. FEM tests skip
when the scientific dependencies are missing; skipped tests are not numerical
validation. Independent comparison with the old FEniCS output remains useful.

### Verified run

All 11 tests passed in Ubuntu WSL with DOLFINx 0.10, both in serial and on two
MPI processes. The original-size 64 x 64 crossed P2 example ran with 33,025
DOFs and ten modes, including CSV, NPZ, XDMF/HDF5, and PNG export. Its first
three frequencies were 28.477510, 69.796807, and 174.666192 kHz.

The largest directly measured force-balance residual in that run was about
2.6e-6. The requested SLEPc tolerance controls its iterative convergence;
the separately reported physical residual also reflects matrix conditioning
and floating-point assembly/factorization error. This is not a comparison
against stored output from the old FEniCS notebook.

References:
[DOLFINx interior-penalty demo](https://docs.fenicsproject.org/dolfinx/v0.10.0/python/demos/demo_biharmonic.html),
[DOLFINx PETSc assembly](https://docs.fenicsproject.org/dolfinx/v0.10.0/python/generated/dolfinx.fem.petsc.html),
[SLEPc eigenpair API](https://slepc.upv.es/release/slepc4py/reference/slepc4py.SLEPc.EPS.html).
