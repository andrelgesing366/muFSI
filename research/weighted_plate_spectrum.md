# Kirchhoff-Love plate spectrum with weighted polynomial pressure

This experiment couples the existing dry isotropic KL finite-element modes to
the continuous Stokeslet mobility. New code and outputs live in `research/`;
production modules are imported read-only. No copy of the production FEM
implementation is needed. Coupling currently runs on one MPI rank.

## Pressure basis and integration

The sheet occupies x in [0,L] and y in [-W/2,W/2]. With xi=2x/L-1 and eta=2y/W,

```text
p(x,y) = sum_(m=0..M,n=0..K) a[m,n] T_m(xi) T_n(eta)
         / sqrt((1-xi^2)(1-eta^2))
H[i,(m,n)] = integral Gzz(target_i-source,omega) psi_(m,n)(source) dA
```

Here **K is the maximum transverse degree, with ALL degrees 0..K**.
The default M=4,K=6 has 35 coefficients: 20 even and 15 odd in y. This is
different from the beam's even-only `T_(2*k)` convention. The old beam convention
and its current defaults are preserved. Coefficients flatten m-major, n-minor.
The experimental shared quadrature now reads the basis's transverse degrees.

`weighted_plate_mobility.py` supplies the full basis and reflection-aware
assembly. It uses the same numerical cosine/Duffy transformation, stable
Stokeslet kernel and graded Gauss-Legendre quadrature as the beam experiment.
Quadpy generates the default line rules; `--backend gauss` selects NumPy rules.
There is no analytic primitive for the general weighted unsteady integral.

Fluid collocation covers both signs of y (default 10 x 14, H=140 x 35).
Least squares uses equal row weights on this angularly clustered grid, not
an area-weighted residual norm. Check-point diagnostics use a different grid.
Only distinct positive-y targets are integrated; reflected rows are obtained
by multiplying each column by (-1)^n. This follows from the rectangular domain
and isotropic unbounded fluid kernel and is valid for mixed-mode excitation.
It does not constrain pressure or velocity to be even. Centreline odd columns
vanish by reflection. Independent tests compare this reuse to direct integration
of negative-y targets. These symmetry shortcuts would need revision for walls,
asymmetric geometry, or an asymmetric fluid environment.

Pressure is the signed normal traction applied to the ideal fluid sheet in Pa.
The force exerted on the plate has the opposite sign. This is not a calculation
of separate absolute fluid pressures on finite-thickness top and bottom faces.
The product endpoint weight remains a candidate, with no claim about the exact
rectangular corner exponent.

## Plate and coupled response

Default plate: 500 x 500 x 5 micrometres, E=169 GPa, rho_s=2330 kg/m^3,
Poisson ratio 0.3. It is clamped at x=0 and free on the other three edges.
Fluid: rho=997 kg/m^3, mu=890 micro-Pa s, unbounded 3D linear Stokes flow.
`--boundary-condition` also exposes bridge, clamped and simply supported plates.
The default crossed 24 x 24 P2 mesh uses the production KL interior-penalty
formulation, with its default penalty of 16.

Three dry FEM eigenmodes are retained by default, including the first
antisymmetric mode. Eigenvectors are normalized by their largest DOF value;
the corresponding modal masses and stiffnesses are rescaled consistently.
Arbitrary-point evaluation uses the original FEM basis, not interpolation of
plotted mode samples. Modal reflection errors are measured, not imposed.
Degenerate eigenmodes can mix parities; the full pressure basis accommodates
this without assuming each eigenvector has a definite parity.

The full two-dimensional force projection is

```text
Phi[i,j] = phi_j(collocation_i)
C[j,(m,n)] = integral phi_j(x,y) psi_(m,n)(x,y) dA
H A = Phi                         (column-scaled least squares)
Z = C A
(K_struct - omega^2 Mass_struct + i omega Z) d = F
pressure_coefficients = i omega A d
w(x,y,omega) = sum_j phi_j(x,y) d_j
```

The time convention is exp(+i omega t). H maps Pa to m/s, and Z maps unit modal
velocity to generalized resisting force in N s/m. All retained modes couple
through Z; it is neither diagonalized nor forcibly symmetrized. Unlike the beam
shortcut, C includes both dimensions and all transverse degrees. Cosine-space
Gauss quadrature computes C; successive order checks compare every mode row.
FEM mode shapes are piecewise polynomials, so this projection needs its own
convergence check even when the pressure-weight Jacobian cancels exactly.

The default real point force is 1 nN at x/L=1,y/W=0.3, and displacement is
monitored at that same point. This off-centre load excites both parities.
`--load antisymmetric` applies +F there and -F at the mirrored point, so each
force has the requested magnitude. This yields odd displacement and traction
to the FEM/coupling accuracy. A point on a supported edge or a centreline odd
force pair can have zero modal forcing and is rejected.

## Run and outputs

In the repository's matched DOLFINx/PETSc/SLEPc scientific environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/weighted_plate_spectrum.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/test_weighted_plate.py -v
```

The default sweep uses 25 geometric frequencies, from 0.01 times the first dry
frequency to 1.1 times the second. Its sample maximum is not a fitted resonance.
Explicit `--frequencies` may include zero. At zero frequency, pressure and
velocity vanish; the reported response check instead uses displacement-shaped
unit velocities to keep the fluid approximation diagnostic meaningful.
No material damping is imposed. A frequency exactly at a dry eigenfrequency
can make the dry comparison singular.

Results go to `research/results/weighted_plate_spectrum/`:

- `spectrum.npz`: complex spatial displacement and pressure at every frequency,
  modal coefficients, impedance matrices, force projection, collocation and
  check points, modal mass/stiffness, dry mode DOFs and shapes, selected H matrices.
- `monitor_spectrum.csv`: frequency, complex monitor displacement, amplitude,
  phase. Displacement grid is shaped (frequency,x,y); pressure uses a separate
  strictly interior grid because of its assumed endpoint singularities.
- `report.json`, `report.md`: parameters, eigen residuals, mode parity,
  force-projection convergence, independent-point velocity errors, rank,
  conditioning, quadrature, reciprocity and minimum resistance eigenvalue.
- `monitor_spectrum.png`, `displacement.png`, `pressure.png`, `dry_modes.png`.

H is a small dense coefficient mobility, not hierarchical matrix compression.
Positive eigenvalues of (Z+Z.conj().T)/2 indicate dissipative retained-mode
loading. Oversampled collocation does not guarantee this or exact reciprocity.
Integration convergence is a successive-rule estimate, not a rigorous bound.
Every completed run checks that production source hashes remain unchanged.

## Executed checks and first results

The default run gave dry frequencies 28.490, 69.877, and 175.010 kHz, with measured
parities even, odd, even. Its largest sampled off-centre displacement was
0.1995 nm at 5.909 kHz. Maximum independent-point response error over the sweep
was 11.46%; maximum reciprocity defect 1.40%. All sampled resistance eigenvalues
were positive. This is an exploratory low-order spectrum.

At four selected frequencies (1.000, 5.909, 18.966, 48.211 kHz), increasing
to M=8,K=8 reduced the response approximation errors to 0.98%, 1.93%, 3.57%,
and 2.83%, respectively. Maximum reciprocity defect was 0.028%. The displacement
at 5.909 kHz changed from 0.1995 to 0.19845 nm (about 0.5%). These points do not
establish convergence over the whole frequency sweep.

A 32 x 32 structural mesh at 5.909 kHz, retaining the default pressure order,
gave 0.19985 nm. Its first three dry frequencies were 28.484, 69.837, and
174.838 kHz. Structural mesh refinement and pressure refinement are independent.

At M=8,K=8, increasing the retained mode count from three to six changed monitor
amplitudes by 0.44% at 5.909 kHz and 1.01% at 18.966 kHz. This is a two-frequency
truncation check, not proof of convergence at higher frequencies. The
antisymmetric force-pair run had even-pressure coefficient norm fractions below
3e-8; the pressure plots show opposite signs at corresponding side positions.

Reproduce the selected checks:

```sh
.venv/bin/python research/weighted_plate_spectrum.py --M 8 --K 8 --frequencies 1000 5908.9 18965.7 48210.7 --output research/results/weighted_plate_refined
.venv/bin/python research/weighted_plate_spectrum.py --load antisymmetric --frequencies 1000 10000 20000 --output research/results/weighted_plate_antisymmetric
.venv/bin/python research/weighted_plate_spectrum.py --mesh-x 32 --mesh-y 32 --frequencies 5908.9 --output research/results/weighted_plate_mesh32
.venv/bin/python research/weighted_plate_spectrum.py --M 8 --K 8 --modes 6 --frequencies 5908.9 18965.7 --output research/results/weighted_plate_six_modes
```

Seven new independent tests passed: odd-pressure mobility against analytical
x integration with an elliptic function and adaptive y integration; reflected
rows against direct negative-y quadrature; a known nonzero odd-mode force
moment; odd pressure from antisymmetric velocity; simply supported plate
frequencies against the exact analytical spectrum; FEM modal mass and dry
response; and supported-edge displacement/point-force virtual work. Existing
beam pressure tests also passed after the degree-property extension.
