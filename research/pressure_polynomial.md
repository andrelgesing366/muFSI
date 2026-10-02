# First-mode pressure and polynomial analysis

`pressure_polynomial.py` is an isolated postprocessing experiment. It imports
the existing `Stokes3DAnalytic` solver and changes no production modules. It
prescribes the exact first clamped/free Euler–Bernoulli mode, normalized to
real unit tip velocity, and solves the unsteady 3D thin-sheet Stokeslet problem.
Pressure means resisting normal traction, divided by tip velocity (Pa s/m).
The actual force on the beam has the opposite sign. There is no coupled
structural solve and no requirement that the frequency equal a resonance.

## Run

From the repository in its scientific Python environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/pressure_polynomial.py --quick --output research/results/pressure_polynomial_quick
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/pressure_polynomial.py
```

Defaults: L=500 µm, W=25 µm (aspect ratio 20), rho=997 kg/m³,
mu=890 µPa s. The frequency satisfies omega*(W/2)^2/nu=1 (about 909 Hz).
The flow has zero thickness; the geometry container's thickness is unused.
Grids are 48×24, 96×24, 48×48, 96×48. These separate longitudinal and
transverse refinement. The quick run uses 24×12, 48×12, 24×24, 48×24 and
is only a pipeline/initial-shape diagnostic. Dense complex LU still limits
memory and runtime; the final default grid needs roughly 650 MiB for matrix
and LU alone. Older matrices are released before each new grid solve.

Choose frequencies, grid sizes and polynomial orders explicitly:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/pressure_polynomial.py --frequencies 1000 10000 --grids 32x24 64x24 32x48 64x48 --orders 2x2 4x4 8x6 12x8 --output research/results/pressure_polynomial_frequencies
```

M×N denotes the maximum longitudinal and transverse degree. N must be even;
only even transverse terms are present by reflection symmetry. The final grid
must dominate every earlier grid in both panel counts. All configuration,
package versions, production source hashes and the experiment source hash
are saved in `manifest.json`. Each frequency has its own report and figures.

## Representations and fitting

Let xi=2x/L−1 and eta=2y/W. Compare the tensor Chebyshev polynomial P(xi,eta),
P/sqrt(1−eta²), and P/sqrt((1−xi²)(1−eta²)). The last representation is a
candidate, not an established rectangular-corner law. Coefficients are complex;
the spatial basis is real. Real and imaginary plots use the velocity phase.

The solver returns constant rectangular-panel coefficients, not exact point
pressure samples. Therefore each candidate basis is integrated over the same
panels analytically before fitting. Unweighted Chebyshev moments use polynomial
antiderivatives; weighted moments use theta=acos(t). Continuous reconstructed
curves and fitted panel averages are both shown in transverse plots.

Every family minimizes the same objective:

```text
sum_j area_j * (1-xi_j^2) * (1-eta_j^2) * abs(p_fit_j-p_reference_j)^2.
```

This is a finite-grid norm of a commonly weighted pressure remainder, not an
unweighted continuum L2 norm of singular pressure. 20% of panels are held out
using a fixed spatial pattern. Two fitting domains are compared: all corners
included, and four corner squares of side 0.25W excluded. The latter's held-out
error covers only its retained domain, while velocity and force errors always
cover the whole sheet. Condition numbers and numerical ranks are recorded.

Metrics include held-out/interior/corner weighted pressure errors, physical
area L1 pressure error, no-slip error of the fitted panel field, integrated
line-load error, net force and first-mode generalized resisting force. Physical
panel areas are used throughout; the analytic grid's legacy Chebyshev force
weights differ from these areas and are intentionally not used here.

Grid comparisons restrict the finest piecewise-constant field to each coarse
grid by exact area overlaps. This preserves net force even when grids are not
nested. Modal force comparisons use each grid's prescribed mode values, so
they include the corresponding projection discretization. Tightening the
base grid's integration tolerance by ten independently checks quadrature
sensitivity. A small solve residual alone does not establish spatial accuracy.

## Outputs and limits

Each frequency directory contains `report.md`, `report.json`, `fit_metrics.csv`,
`grid_metrics.csv`, `corner_probes.csv`, reference pressure NPZ files, complex
fit coefficient/panel NPZ files and eight PNG figures. The plot set covers:

- Real/imaginary/magnitude pressure over the whole sheet.
- Transverse and longitudinal polynomial fits.
- Real/imaginary fits at different polynomial orders near the tip.
- Weighted panel remainder proxies.
- Fit error versus coefficient count, with and without corner fitting.
- Independent x/y grid refinements.
- Corner approach rays at both root and tip, with three physical dy/dx ratios.

Corner probes interpolate raw complex panel coefficients bilinearly inside
the collocation-node hull. They never extrapolate to an edge. The plotted
logarithmic slopes are finite-resolution diagnostics, not certified corner
exponents. Uniform longitudinal spacing can leave too little resolved radial
range to establish an asymptotic law. The last-section refinement plot changes
its x location with nx and is labelled accordingly. The clamped support is not
represented as a separate hydrodynamic surface.

The best fit listed in a generated report is ranked by full-sheet velocity
error on that frequency/grid, not declared a universal choice. This experiment
does not assemble Stokeslet integrals against the continuous singular basis:
its mobility checks use projected constant-panel reconstructions. A future
weighted spectral solver would need its own singular integration and validation.

Independent experiment checks:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s research -p test_pressure_polynomial.py -v
```

These check basis integrals against Gaussian quadrature, manufactured complex
density recovery on held-out panels, area-preserving restriction, a passive
and symmetric existing fluid solve, and the distinction between nodal values
and panel means. Production analytic-panel checks can be run separately.

The completed default run is linked from
[`results/pressure_polynomial/README.md`](results/pressure_polynomial/README.md).
Generated numerical data and plots are excluded from Git by this research
folder's local ignore file; they remain available on disk and are reproducible.
