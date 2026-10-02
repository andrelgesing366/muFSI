# Extensive formulation study

`benchmarks/formulation_study.py` compares EB + weighted 3D, KL + weighted
3D, EB + 2D, KL + 2D, and EB + Sader with equal total corner forcing.
The response uses the full structural finite-element system. Geometry cases
are 800 × 50, 500 × 50, and 500 × 250 µm, all 5 µm thick.

Run with the matched DOLFINx/PETSc environment, in serial. Set
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` for reproducible timings.
From the repository root:

```bash
export PYTHONPATH=src:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for case in beam slender wide; do
  python benchmarks/formulation_study.py --stage sweep --case "$case" --resume
  python benchmarks/formulation_study.py --stage validation --case "$case"
  python benchmarks/formulation_study.py --stage adaptive --case "$case"
done
python benchmarks/formulation_study.py --stage odd --case wide
python benchmarks/formulation_study.py --stage section-validation --case wide
python benchmarks/formulation_study.py --stage refined-sections --case wide
python benchmarks/formulation_study.py --stage refined-odd --case wide
python examples/formulation_flow.py
# Run these sequentially when other computations have finished:
for case in beam slender; do
  python benchmarks/formulation_study.py --stage timings --case "$case"
done
python benchmarks/formulation_study.py --stage timings --case wide --section-nx 129 --section-ny 256
python examples/formulation_spectra.py
```

Each initial spectrum has 404 distinct frequency samples: 161 logarithmic
broadband points and 81 linear samples around each of three initial
resonance windows. The windows are seeded by Sader estimates. `adaptive`
locates actual response peaks and adds targeted samples where necessary;
for the wide plate it also searches beyond the old 150 kHz limit so finite
length shifts do not hide a higher resonance. All figures join actual
computed samples. The plot example can regenerate figures without DOLFINx.

Baseline pressure degrees are 16 and 8, with 24 × 6 independent positive-y
EB collocation points and 24 × 12 KL points. EB contains only even transverse
pressure degrees; KL contains both parities. Integration uses graded Duffy
quadrature with relative tolerance 5×10⁻⁶. The spectrum reuses identical
kernel integrals between matching EB/KL points and basis columns, checked
against an independently assembled EB matrix. Structural responses remain
independent. This sharing is disabled in timing benchmarks.

`validation` varies one parameter at a time at the three Sader wet-frequency
estimates: pressure degrees (20, 12), collocation (32 × 10 independent rows),
kernel integration tolerance (10⁻⁶), structural meshes (twice each mesh
dimension), and section quadrature (128 transverse and 129 longitudinal
points). It reports complex displacement and energy-Q changes, plus
no-slip errors at 35 surface points excluded from collocation. These checks
are response probes, not complete refined-spectrum convergence or rigorous
error bounds. Higher KL modes can remain sensitive to the structural mesh.

The wide plate also gets a separate 2D grid check: 128 versus 256 transverse
points at 129 fixed longitudinal points. The highest KL-mode energy Q still
changes by about 3.6%, compared with about 19% in the original coarse-grid
refinement at the third wet-frequency estimate. `refined-sections` therefore generates the primary wide-plate
2D curves at 256 transverse points, and `refined-odd` uses the same grid for
the opposite-corner-load example. These results are stored separately, so
the initial 64-point spectra and their diagnostics remain available.

The Q plot distinguishes fitted SHO Q from energy/work Q. SHO fits use
displacement magnitudes, a coherent background, and an isolated frequency
window. The fitted undamped frequency differs from the amplitude maximum.
Energy Q uses maximum structural bending plus kinetic energy and the
formulation's own dissipated fluid work, without a stored-fluid-energy term.
A low fit residual does not make these two definitions identical. The wide
plate's separate opposite-corner-load example compares the first
antisymmetric KL response, absent from the EB model.

EB and Sader results at length/width = 2 are extrapolations for comparison;
the plate models permit transverse deformation. Reduced models agreeing
with one another is not validation for a wide plate.

`timings` launches a fresh process for each model and performs nine solves
(three repetitions of three resonance estimates) after warm-up. Setup and
force projection are recorded separately. Total peak process RSS includes
the scientific runtime and both common study meshes, so it is not a measure
of isolated solver storage. Run timing workers sequentially without other
active sweeps.

Output is in `results/formulation_study/`. Each case retains complex
displacement NPZ data, Q CSV/JSON, integration and work-balance diagnostics,
source SHA256, configuration, refinement probes, and timing samples.
`figures/` contains PNG and vector PDF plots plus `report.md` and a combined
Q table. The initial sweeps checkpoint every ten frequencies; `--resume`
checks source, parameters, and the frequency grid before continuing.

The existing `examples/formulation_comparison.py` remains the compact
example, including a 2D fluid-field reconstruction from the coupled
displacement. The extensive spectra compare model-specific hydrodynamic
forces; no 2D field dissipation is substituted for 3D Q.
`examples/formulation_flow.py` adds in-phase and quadrature field snapshots
for the wide plate's first antisymmetric weighted-3D response. Complex field
data and PNG/PDF plots are saved under `results/formulation_study/flow/`.
