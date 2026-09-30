# F2D and Sader frequency response

`Stokes2D` ports the transverse panel formulation in the old `Fluid/F2D.py`.
`examples/cantilever_2d.py` reproduces the deterministic uniform-pressure
spectrum in `Example_2_F2D_spectrum.ipynb`, using the isotropic DOLFINx plate
and a Sader Euler–Bernoulli beam reference.

## Grid, assumptions, and units

The plate occupies `[0,L] x [-W/2,W/2]`. The fluid is incompressible,
Newtonian, unbounded, and linearized about rest. Each x section is an infinitely
long, zero-thickness strip. Sections do not interact in x. The model includes
transverse variations in motion/pressure, but no fluid end effects, walls, or
finite-thickness flow.

All quantities use SI units. Public frequencies are Hz; kernel frequencies
are angular frequencies in rad/s. The harmonic convention is
`exp(+i omega t)`. Positive pressure is **resisting traction**: the actual
fluid force on the structure is its negative.

`FluidGrid.chebyshev_gauss(geometry, nx=32, ny=128)` reproduces the old rule:

- Composite Simpson quadrature in x, including endpoints. An even nx is
  increased by one, so this request produces 33 sections.
- Ascending Chebyshev–Gauss nodes in y, clustered at the edges.
- Transverse panel boundaries halfway between nodes, with outer edges at ±W/2.
- Ordinary integration weights
  `wy = pi/ny * sqrt((W/2)^2 - y^2)`; these converge to W and are not
  renormalized, preserving the original quadrature.
- x-major ordering: index `ix*ny + iy`; weights `wx[ix]*wy[iy]`.

The midpoint factory is also implemented. Grid coordinates, positive area
weights, and panel boundaries are validated and stored as read-only copies.

## F2D kernel

Let nu = mu/rho and s = sqrt(omega/nu). Define the odd primitive

```text
f(z) = sign(z) [1/abs(z) + ker'(abs(z)) + i kei'(abs(z))]
A[i,j] = [f(s*(edge[j+1]-y[i])) - f(s*(edge[j]-y[i]))] / (2*pi*i*s)
B = A/mu
v = B @ p
```

`section_mobility(omega)` returns one transverse B in m/(Pa s).
`pressure_from_velocity` solves B p = v using one LU factorization for
all x sections and multiple RHS columns. A small-argument series removes
cancellation between 1/z and the Kelvin derivative. Calculations use
complex128; no explicit inverse is formed.

The optional `assemble_matrix` returns the sparse block diagonal mobility,
one identical transverse block per x section. It maps pressure to velocity,
not structural displacement to force. Frequencies must be strictly positive.

## Coupling and solution

The coupling layer locates cells and tabulates scalar Lagrange basis functions
directly with DOLFINx/Basix, without an auxiliary fluid FEM mesh:

```text
E[i,j] = phi_j(fluid_point[i])
fluid velocity = i*omega*E @ u
resisting structural force = E.T @ Q @ p
```

Q contains area weights. The transpose projection preserves discrete virtual
work. Cell-edge and plate-boundary points are supported; outside points are
rejected. Initially, coupling requires one MPI rank and continuous scalar
elements with identity DOF transformations (including P2). The structural
eigen solver retains MPI support.

After eliminating supported displacement DOFs, the F2D solver solves

```text
[ K - omega^2 M     E.T Q ] [u] = [F]
[ -i omega E          B  ] [p]   [0]
```

This equals the old
`(K - omega^2 M + i omega E.T Q B^-1 E) u = F`.
A sparse block solve avoids a dense structural hydrodynamic matrix. Rows and
pressure variables are scaled; two refinement solves reuse LU factors.
A generic matrix-free path accepts hydrodynamic models that implement only
`pressure_from_velocity`.

`FrequencyResponseResult` holds frequency-first complex displacement/pressure
arrays in the full structural/grid order, with fixed displacement entries zero.
`relative_errors` measures force balance divided by the applied force norm;
`fluid_errors` measures no-slip divided by the velocity norm for F2D, and is
NaN for action-only models.

## Sader reference

`SaderMethod` implements the rectangular hydrodynamic function in Sader,
*Journal of Applied Physics* **84**, 64–76 (1998),
[doi:10.1063/1.368002](https://doi.org/10.1063/1.368002), Eqs. (18), (20)–(22).
A scaled Bessel ratio avoids underflow. The paper's exp(-i omega t) Gamma is
conjugated for the library response convention. `gamma_function(Re)` retains
the paper convention for direct reference use.

The imaginary correction's fifth-order coefficient is **-0.000044510**,
as printed in Eq. (21b); the old `Fluid/Sader.py` used -0.000045510.
The fit was validated in the paper over 1e-6 <= Re <= 1e4; values outside that
range are extrapolations.

Uniform-load compliance solves the clamped/free beam equation in Appendix B,
Eq. (B4). A static series handles small frequencies; bounded exponential basis
functions replace large hyperbolic terms. `displacement_per_line_force`
returns m/(N/m); `displacement_per_pressure` multiplies by W and returns m/Pa.
Both return shape `(nfrequencies,npositions)`, with positions in metres and
the tip as the default.

The reference assumes a slender cantilever with W much greater than t. Sader
uses an Euler–Bernoulli beam; F2D is coupled to a Kirchhoff–Love plate. Exact
agreement is not expected, particularly for higher modes. The example is a
driven uniform-pressure response, not a thermal noise PSD.

## Run the example

Install NumPy/SciPy and a matched DOLFINx/PETSc environment; use the `plot`
extra for the example. SLEPc is required by the separate eigen workflow.
See [the plate setup guide](plate_eigenproblem.md) for the WSL environment.

From the repository in Linux/WSL:

```console
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 examples/cantilever_2d.py
PYTHONPATH=src python3 examples/cantilever_2d.py --quick
```

Defaults match the old notebook: L=500 um, W=50 um, t=5 um; E=169 GPa,
rho_s=2330 kg/m^3, nu_s=0.3; water rho=997 kg/m^3, mu=890e-6 Pa s;
64x6 crossed P2 mesh; 33x128 fluid points; 200 logarithmic frequencies from
1 to 400 kHz; uniform pressure 1 Pa; measurement at (L,W/2).

Outputs in `results/cantilever_2d/`:

- `spectrum.png`: amplitude comparison in nm/Pa.
- `spectrum.csv`: complex tip responses, magnitudes and residuals.
- `response.npz`: full complex displacement/pressure, coordinates and weights.
- `parameters.json`: parameters, discretization, convention and provenance.

## Verification

```console
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

Tests compare the Kelvin matrix with the legacy formula, batch/vector pressure
solves with the fluid equation, rigid-section impedance with Sader at three
frequencies, Sader compliance with Eq. (B4) and its exact static limit, basis
transfer with quadratic fields and DOLFINx evaluation, discrete virtual work,
and the block response with an independently eliminated dense system. Existing
plate/eigen tests remain in the suite.

On DOLFINx 0.10.0.post5, PETSc 3.24.4, NumPy 2.3.5 and SciPy 1.16.3, the
full example (3213 structural DOFs) yielded sampled amplitude peaks:

| Mode | F2D [kHz] | Sader [kHz] |
| --- | ---: | ---: |
| 1 | 11.119 | 11.119 |
| 2 | 76.366 | 76.366 |
| 3 | 225.747 | 219.052 |

These are maxima on the specified 200-point grid, not fitted resonance
frequencies. Maximum force-balance residual was 6.7e-6 and no-slip residual
1.7e-15. This verifies a working comparison, not a completed mesh/grid
convergence study.
