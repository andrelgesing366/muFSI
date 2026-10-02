# Weighted 3D / Sader / 2D beam spectrum comparison

Run this standalone example from `muFSI/` in the existing scientific Python
environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/weighted_beam_comparison.py
```

On Windows, run the same command in the project's WSL environment. This creates
`research/results/weighted_beam_comparison/` with the PNG/PDF figure, complex
displacement fields, complex pressure coefficients, modal fluid impedances,
CSV spectra, JSON diagnostics, an order study and a Markdown report. Production
source is imported read-only; the scripts and generated files stay in research.

## Identical beam and forcing

The defaults match `examples/beam_2d.py`: 800 x 50 x 10 um silicon cantilever,
E=169 GPa, solid density=2330 kg/m3, water density=997 kg/m3, viscosity=890 uPa s,
and uniform line forcing 1e-3 N/m. All three plotted models use the same three
exact dry Euler-Bernoulli modes, modal load vector and exp(+i omega t)
convention. This isolates hydrodynamic differences from structural differences.

The frequency interval is 1--30 kHz with 41 samples, focusing on the first bending
peak. This is a focused comparison, rather than the full 1--500 kHz interval of
the original FEM example. Extending to higher bending modes requires checking
the structural mode count, polynomial pressure orders and sampling again.

The curves are:

- **3D weighted Stokeslet:** finite rectangular thin sheet, continuous weighted
  Chebyshev traction, numerical cosine/Duffy integration and nonlocal fluid
  coupling along the beam.
- **Sader:** the existing rectangular hydrodynamic function, projected onto the
  same beam modes with the consistent line-force integral.
- **2D Tuck / Kelvin:** the existing `SectionForce2D(method="tuck")` model,
  identical to the fluid model in the beam example; 64 transverse panels,
  full-panel force integration and independent sections along the beam.

Sader's analytical full-beam displacement is also evaluated as an independent
structural reference. It is saved in the NPZ output and compared numerically
with the modal Sader response. A fourth structural mode and 128 Tuck panels
provide separate truncation/refinement checks. None of the three spectra is
rescaled, fitted or adjusted to match another curve.

## Selected pressure orders

The new research defaults are **M=16, K=4**, giving **85 complex coefficients**
and maximum transverse degree 8. The default 3D collocation matrix is 340 x 85.
Both orders remain configurable. These defaults were selected for this frequency
range and slender-beam example, rather than asserted to work for every geometry.

The script's order study evaluates candidates (8,4), (12,4), (16,4), (20,4),
(16,6) and (20,6) on the same 588 interior collocation rows and separate check
points. Shared quadrature/targets prevent changes in collocation density from
being mistaken for polynomial convergence. At 1, 12 and 30 kHz:

| M, K | Largest held-out error among three unit beam-mode velocities |
|---|---:|
| 8, 4 | 6.20% |
| 12, 4 | 1.75% |
| **16, 4** | **0.77%** |
| 20, 6 | 0.34% |

At M=16, K=4 the complex tip response differs from the M=20, K=6 result by
less than 0.003% at these sampled study frequencies. K=6 gives little additional
response accuracy at M=16, so the numerical budget is better spent on the
longitudinal degree than on more transverse terms. The largest tested basis is
still a finite approximation; the study does not establish exact corner behaviour.

## Options

```sh
# Only the pressure-order study:
.venv/bin/python research/weighted_beam_comparison.py --study-only

# Spectrum only, without repeating the study:
.venv/bin/python research/weighted_beam_comparison.py --skip-study

# More pressure terms and denser frequency sampling:
.venv/bin/python research/weighted_beam_comparison.py --M 20 --K 6 --samples 81 --output research/results/weighted_beam_comparison_fine

# Use the earlier 500 x 25 x 5 um beam with the same uniform forcing:
.venv/bin/python research/weighted_beam_comparison.py --length 500e-6 --width 25e-6 --thickness 5e-6 --output research/results/weighted_beam_comparison_slender
```

Parameters for solid/fluid properties, geometry, force, frequencies, Tuck panel
count and quadrature are also available through `--help`. Changing any physical
parameters applies to all three models. With `--skip-study`, previously saved
study files are not used to select or certify orders for a new configuration.

At every frequency the script reports independent-point 3D no-slip error,
individual mode errors, numerical rank, conditioning, integration-order estimates,
modal equilibrium, impedance reciprocity and resistance eigenvalues. The output
report also records the analytical Sader agreement and Tuck/mode refinement.
Small algebraic residuals alone are not a spatial-convergence claim. Peak values
in the report are the largest sampled amplitudes, not fitted resonance parameters.

Run the independent checks with:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/test_weighted_pressure.py -v
```

See the [generated report](results/weighted_beam_comparison/report.md) and
[comparison figure](results/weighted_beam_comparison/spectrum_comparison.png).

## Completed default comparison

The 41-frequency spectrum was run with the selected M=16, K=4 orders. The
three curves have similar shapes; the finite 3D fluid model moves the largest
sampled response to a slightly higher frequency than the local models.

| Model | Largest sampled response frequency | Tip amplitude |
|---|---:|---:|
| 3D weighted Stokeslet | 11.875 kHz | 362.56 nm |
| Sader | 11.150 kHz | 358.53 nm |
| 2D Tuck / Kelvin | 11.150 kHz | 358.47 nm |

The frequency spacing is 725 Hz; these sampled values should not be read as
precise resonance frequencies. No curve was adjusted to match another model.

The maximum 3D independent-point no-slip error is 0.830%. Modal Sader versus
analytical full-beam tip response differs by at most 0.0755%. Three to four
structural modes changes the local-model tip response by at most 0.0956%.
Refining Tuck from 64 to 128 panels changes its complex tip response by at most
0.104%. The 3D impedance reciprocity defect is below 0.00061%, and all tested
resistance eigenvalues are positive. Production source hashes stayed unchanged.

The spectrum and its independent check-point integrations took about 9.3 minutes
on the existing scientific environment; the optional order study adds its own
assembly cost. `--skip-study` avoids repeating that study. All nine independent
research checks and the code-quality checks passed.
