# Euler-Bernoulli cantilever and local 2D fluid force

`EulerBernoulliBeam` implements the beam formulation from
`1D Cantilever in Vacuum` in FEniCSx/DOLFINx. It uses the existing
`EigenSolver` for in-vacuo modes. `SectionForce2D` and
`BeamFrequencyResponseSolver` implement the local fluid loading in
`1D Cantilever in Fluid`.

## Review of the reference folders

The vacuum folder contains static, eigenvalue and frequency-domain scripts,
plus their v2 variants. They use continuous Lagrange displacement fields,
interior derivative-jump terms and weak clamped-slope conditions. The
eigen scripts compare with roots of cos(beta)cosh(beta)+1=0. Their mode
selection based on the eigenvector mean and absolute eigenvalues is replaced
by elimination of constrained displacement DOFs and the existing positive,
mass-normalized SLEPc eigen solve.

The fluid folder's `Cantilever_in_Fluid.py` applies the Sader hydrodynamic
function as an added complex line-mass term. `main.py` compares an analytical
beam, FEM with Sader force, and FEM with numerical Tuck force. Its
`My_Package` imports are absent from that folder. The corresponding helpers
are present in the older archive at:

```text
2025 - Arthur/2025-03/stokeslet_integration-main/src/stokeslet_integration/
  fsi_cpu/older_versions/FSI_v1/PlAFeM/
    beam/beam_fluid_coupling.py
    fluid/fluid_dynamics.py
```

The Tuck helper solves a transverse Kelvin-panel problem for a rigid section
and integrates its pressure to a line force. The port reuses the stable F2D
kernel and LU solver. The current Sader implementation retains the corrected
Eq. (21b) coefficient described in [the F2D guide](f2d_spectrum.md).
The legacy thermal normalization and GIF routines are outside this driven
response implementation.

## Beam formulation

The physical interval is [0,L]. For out-of-plane bending of a rectangular
beam,

```text
A = W*t                    [m^2]
I = W*t^3/12               [m^4]
EI = E*I                   [N m^2]
line density = rho*A       [kg/m]
EI*u'''' + rho*A*u_tt = q   q in N/m
```

There is no plate factor 1/(1-nu^2) in EI. Continuous Lagrange degree p>=2 is
used, with symmetric C0 interior penalty:

```text
a(u,v) = integral EI*u''*conj(v'') dx
         - sum EI*avg(u'')*conj(jump(v'*n))
         - sum EI*jump(u'*n)*conj(avg(v''))
         + sum alpha/avg(h)*jump(u'*n)*conj(jump(v'*n))
alpha = penalty*(p/2)^2*EI
```

Clamped endpoints add the two symmetric consistency terms and a
2*alpha/h slope penalty. Displacement is constrained strongly; slope is
enforced weakly. This follows the original beam construction and the
[DOLFINx C0 interior-penalty weak form](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/demos/demo_biharmonic.html),
with beam rigidity and endpoint terms.

Supported conditions:

| Name | Displacement | Slope |
| --- | --- | --- |
| cantilever | zero at x=0 | zero at x=0 |
| bridge / clamped | zero at both ends | zero at both ends |
| simply_supported | zero at both ends | free |

Free slopes imply the natural zero-moment condition. A free cantilever tip
also has zero shear. The current API implements homogeneous supports.

The default beam uses 64 quadratic elements and penalty=16. The eigen
example uses 40 cubic elements, giving accurate six-mode comparisons at a
small matrix size. Increasing degree or mesh size can worsen floating-point
conditioning; residuals are reported separately from analytical frequency
errors.

K and M are returned in the full DOF space as caller-owned PETSc matrices.
The mass is consistent, `M_ij = integral rho*A*phi_i*phi_j dx`. Construction
is lazy and configuration immutable. Structural assembly/eigen solves
support MPI.

`DistributedLoad` receives coordinates of shape (gdim,npoints) and returns
line-load values in N/m. A scalar constant is also accepted.
`PointLoad((x,), amplitude)` uses newtons and evaluates the basis at the
actual point; arbitrary point loading currently requires one MPI rank.
Complex loading requires a complex PETSc build. No additional width factor
is applied to a beam line load.

## Local fluid force

`SectionForce2D(geometry, fluid, method="sader" or "tuck")` assumes each
section moves uniformly across its width, with no longitudinal fluid
interactions. It is a reduction of transverse 2D flow to one line impedance,
suitable for an Euler-Bernoulli beam. The existing F2D plate model permits
motion varying across y. Both models assume an unbounded Newtonian,
incompressible fluid and negligible resonator thickness for the flow.

All response calculations use exp(+i omega t). The local operator returns
positive **resisting** line force; actual fluid force on the structure is
its negative:

```text
v = i*omega*u
resisting line force = Z(omega)*v              [N/m]
Z = line force / velocity                     [N s/m^2]
dynamic_stiffness = i*omega*Z                  [N/m^2]
```

For Sader, with the library-convention Gamma conjugated from the paper,

```text
Re = omega*W^2/(4*nu)
m_fluid = pi*rho_f*W^2/4*conj(Gamma_paper(Re))
Z = i*omega*m_fluid
dynamic_stiffness = -omega^2*m_fluid
```

This is the hydrodynamic loading used in `Cantilever_in_Fluid.py`.
It includes frequency-dependent added mass and viscous damping.

For Tuck, the transverse edges follow the original cosine rule,
`edge_j = W/2*cos(theta_j)`, theta spanning [-pi,0]. Collocation nodes
lie at physical panel midpoints. The integrated Kelvin kernel provides
`B p = 1` for unit section velocity. The default impedance is
`Z = sum(diff(edges)*p)`, integrating each piecewise-constant pressure
over its whole panel.

`integration="legacy_trapezoid"` reproduces the original helper's
centre-to-centre trapezoid integration. That integral omits the edge
half-panels; full panel integration is the default. Tests compare the legacy
setting with the original dimensionless formula, including its force sign.

`section_pressure(omega)` returns read-only unit-velocity pressure
in Pa/(m/s) for Tuck. `line_impedance`, `dynamic_stiffness`, and
`resisting_force_from_velocity` use angular frequency in rad/s.
Only the most recent Tuck pressure and section LU factors are cached.
The Sader force provides an integrated result without a pressure field.

## Full beam response

The local force is projected with the consistent FEM line-integration
matrix, `C = M/(rho*A)`:

```text
[K - omega^2*M + i*omega*Z(omega)*C] u = F
```

`BeamFrequencyResponseSolver(beam, force_model).solve(frequencies, load)`
accepts Hz and returns `BeamFrequencyResponseResult` with frequency-first
displacement in metres, resisting line-force values in N/m, and relative
force-balance residuals. Fixed DOFs are zero. `line_force` contains values
of the distributed force at structural DOFs, not an assembled force vector.

The solver uses complex sparse SciPy LU after assembling real or complex
FEniCSx operators, with two residual corrections. No mixed real/imaginary FEM
space, explicit inverse, separate longitudinal fluid grid, or modal
truncation is needed. Driven response currently requires one MPI rank.
Omitting the force model gives the vacuum response. Frequency zero gives
static response with zero fluid force for stationary displacement.

```python
import numpy as np
from mufsi import (
    BeamGeometry, Material, Fluid, EulerBernoulliBeam, EigenSolver,
    SectionForce2D, BeamFrequencyResponseSolver, DistributedLoad,
)

geometry = BeamGeometry(800e-6, 50e-6, 10e-6)
beam = EulerBernoulliBeam(geometry, Material(169e9, 2330, .3))
modes = EigenSolver(beam).solve(6)
force = SectionForce2D(geometry, Fluid(997, 890e-6), method="tuck", ny=32)
load = DistributedLoad(lambda x: np.full(x.shape[1], 1e-3))  # N/m
response = BeamFrequencyResponseSolver(beam, force).solve([1e3, 1e4], load)
```

## Run the examples

Use the matched FEniCSx/PETSc/SLEPc environment described in
[the plate setup guide](plate_eigenproblem.md). Install the `plot` extra
for figures. These examples need no Quadpy dependency.

```console
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 examples/beam_eigenvalue_problem.py --plot
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 examples/beam_2d.py
PYTHONPATH=src python3 examples/beam_2d.py --quick
PYTHONPATH=src python3 examples/beam_2d.py --integration legacy_trapezoid
```

The eigen example retains the old vacuum geometry and material:
L=2 mm, W=100 um, t=10 um, E=200 GPa, rho=2650 kg/m^3.
It writes frequency/error CSV, mass-normalized modes and sampled
analytical/FEM fields in NPZ, P1 mode-shape visualization in XDMF/HDF5,
parameters in JSON, and an optional mode-comparison PNG.
CLI controls include `--nx`, `--degree`, `--modes`, `--penalty`,
`--factor-solver-type` and `--output`. Plotting/NPZ are serial;
XDMF and eigenvalue CSV work with MPI.

The fluid example follows `main.py`: L=800 um, W=50 um, t=10 um,
silicon E=169 GPa, rho=2330 kg/m^3, nu=0.3, and water rho=997 kg/m^3,
mu=890e-6 Pa s. The drive is 1e-3 N/m, giving total force 8e-7 N.
It compares analytical Sader compliance, FEM with Sader force and FEM with
numerical Tuck force over 200 logarithmic frequencies from 1 to 500 kHz.
Defaults use 40 cubic beam elements and 64 transverse panels; quick mode
uses 24 elements, 16 panels and 48 frequencies. `--ny` is capped at 64.
Only one fluid section is solved per frequency; x variation is handled by
the local FEM force term.

Outputs under `results/beam_2d/` include amplitude/phase PNG, complex tip
response and residual CSV, full sampled/FEM fields and line forces in NPZ,
and physical parameters, conventions, discretization, versions and runtimes
in JSON. CLI `--line-load` is in N/m.

## Verification

Eleven new tests cover independent cantilever, clamped/clamped and simply
supported frequencies, quadratic mesh convergence, width/thickness/density
scaling, modal mass orthogonality, symmetric matrices, line-load units,
exact static point/distributed compliance, point-force virtual work,
Tuck normalization against the legacy helper, positive dissipation,
the Sader limit, and full FEM fields against analytical Sader compliance.

The complete current suite passed 62 serial tests in the scientific
environment. Three structural beam checks also passed on two MPI ranks.
The eigen example's maximum relative frequency error for six modes was
2.1e-5 (0.0021%), with maximum eigen residual 2.7e-7:

| Mode | FEM [kHz] | Analytical [kHz] |
| --- | ---: | ---: |
| 1 | 3.508426 | 3.508426 |
| 2 | 21.986933 | 21.986931 |
| 3 | 61.564098 | 61.564043 |
| 4 | 120.641366 | 120.640954 |
| 5 | 199.430000 | 199.428141 |
| 6 | 297.917301 | 297.911110 |

The fluid example completed both 200-frequency FEM sweeps in approximately
0.6 s and 0.8 s after setup, with maximum equilibrium residuals 1.7e-6.
The curves closely overlap for this slender geometry. These examples
demonstrate the implementation; publication results still require
independent discretization and frequency-resolution convergence studies.

```console
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v
PYTHONPATH=src mpiexec -n 2 python3 -B -m unittest tests.structure.test_euler_bernoulli.BeamFEMTests.test_cantilever_spectrum_constraints_and_modal_mass
```
