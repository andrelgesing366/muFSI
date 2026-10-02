# Two-dimensional fluid flow postprocessing

`mufsi.postprocessing.flow` recovers the complex streamfunction, velocity,
symmetric strain-rate tensor, and viscous dissipation from one frequency's
`Stokes2D` pressure solution. Recovery needs only NumPy and SciPy. Plotting
also needs Matplotlib (`pip install -e ".[plot]"`). No FEM import or 3D
mobility evaluation is needed.

## From a coupled solution

```python
import numpy as np
from mufsi import reconstruct_section, plot_flow

# response = FrequencyResponseSolver(problem).solve(frequencies, load)
# Use the same Stokes2D model/grid as the pressure solve.
i = 0
omega = 2 * np.pi * response.frequencies[i]
width = problem.structure.geometry.width
field = reconstruct_section(
    omega,
    np.linspace(-0.75 * width, 0.75 * width, 241),
    np.linspace(-0.15 * width, 0.15 * width, 120),
    response.pressure[i],
    problem.hydrodynamics,
    section_index=-1,
)
fig, axes = plot_flow(field, phase=0)
fig.savefig("flow.png", dpi=180)
```

`section_index` selects the longitudinal fluid section (default: last).
The original pressure vector is x-major with shape `(nx*ny,)`; `(nx,ny)`
and a single `(ny,)` section are also accepted. Complex phases are preserved.
To start from sampled plate displacement, obtain pressure with
`hydro.pressure_from_velocity(omega, 1j * omega * displacement)`.
To start from FE DOFs, first evaluate displacement on the fluid grid using
the problem's coupling evaluation matrix.

`reconstruct_flow` also accepts arbitrary points with shape `(...,2)` for
`(y,z)`, or `(...,3)` for physical `(x,y,z)`. The latter linearly interpolates
full-grid pressure between x sections, with no extrapolation. This is still
independent 2D section flow: it adds no axial velocity or longitudinal
interaction. Strain is the y/z section tensor, excluding axial derivatives.

## Signs, kernels, and dissipation

The harmonic convention is `exp(+i omega t)`. Pressure is resisting traction,
as in the current library: the fluid force on the plate is `-p`. The kernel is

```text
alpha² = i omega / nu
R = sqrt((y - y_source)² + z²)
F(R) = -(log(R) + K0(alpha R)) / (2 pi alpha²)
psi_x = -sum_j p_j [F(y-b_j,z) - F(y-a_j,z)] / mu
u_y = d(psi_x)/dz
u_z = -d(psi_x)/dy
```

Each pressure panel `[a_j,b_j]` is constant, so its derivative integral is
evaluated exactly at the panel endpoints. This matches the pressure
discretization of `Stokes2D`. The minus sign in the streamfunction endpoint
sum corrects the opposite sign used in `Water_1_3_r8-16.ipynb`; reconstructed
surface velocity satisfies `u_z = B p` with the existing mobility matrix.
The fundamental-solution constant is removed as a streamfunction gauge;
it cancels in all panel differences. A small-argument series avoids Bessel
and logarithm cancellation at small distances or frequencies.

Analytic first and second derivatives of the endpoint kernel give velocity
and strain; no finite-difference grid derivative is needed. In particular,

```text
eps_yy = psi_yz; eps_zz = -psi_yz
eps_yz = eps_zy = (psi_zz - psi_yy)/2
mean_dissipation = mu * sum_ij |eps_ij|²          [W/m³]
energy_dissipation = (2 pi/omega) * mean_dissipation [J/m³ per cycle]
```

The factor `1/2` in shear strain corrects the legacy notebook expression.
Both off-diagonal components enter the tensor contraction. The time average
also includes the `1/2` from real, peak-amplitude harmonic fields. These
expressions follow the symmetric strain tensor and full-cycle losses in
Gesing et al., J. Appl. Phys. 131, 134502 (2022), Eqs. (6), (9)-(13), and
Appendix A ([DOI](https://doi.org/10.1063/5.0085514)).

Density integrated over a y/z section is loss per unit length. Total loss
needs integration along x, and Q additionally needs the chosen stored-energy
definition. A finite plotted window does not contain all fluid dissipation.

Points exactly at pressure-panel endpoints on `z=0` have singular derivatives.
They are marked with NaNs and `field.singular_points`, or cause a `ValueError`
with `singular="raise"`. Other surface points retain the no-slip velocity.
Use an even number of z samples for plots to avoid the singular plane.
The plate is infinitesimally thin for this calculation, consistent with
`Stokes2D`; no finite-thickness surfaces or walls are introduced.

## Plots and runnable example

```console
PYTHONPATH=src python examples/flow_visualization.py
PYTHONPATH=src python examples/flow_visualization.py --mode rigid
PYTHONPATH=src python examples/flow_visualization.py --response results/cantilever_2d/response.npz --frequency 100000 --output results/flow_2d_saved
```

The default demo prescribes a 1 nm, three-node-line roof-tile displacement
at 300 kHz on an 800 by 400 um plate. It is a prescribed pattern, not a newly
solved structural eigenmode. The saved-response option uses the pressure and
grid from `cantilever_2d.py` and its adjacent `parameters.json`; frequency
selection chooses the closest saved sample. Other exporters have different
schemas and should be passed to the library API directly.

The example exports `flow.npz`, `parameters.json`, `flow_in_phase.png`, and
`flow_quadrature.png`. Each plot has streamfunction contours with streamlines,
velocity arrows with instantaneous speed, and full-cycle dissipation.
The phase reference is the phase of the largest plate surface velocity;
quadrature plots show the imaginary part in that reference frame. All spatial
phase differences remain intact. Streamlines are instantaneous phase snapshots,
not oscillatory particle trajectories. The two dissipation maps are identical
because full-cycle loss is independent of the selected phase.

Tests check no slip including the low-frequency series, direct numerical
integration of the boundary kernel, curl and incompressibility, strain-rate
derivatives, time-averaged dissipation, phase/amplitude scaling, reflections,
x interpolation, singularities, plotting, and volume/surface work balance.

```console
PYTHONPATH=src python -m unittest discover -s tests/postprocessing -p test_flow.py -v
```


## Fields from 3D or Sader displacement

Use `reconstruct_flow_from_response(structure, response, fluid, grid, points,
frequency_index=0, section_index=...)` with a separate `FluidGrid` for 2D flow.
It evaluates EB/KL structural velocity, solves a fresh Stokes2D pressure field
and calls the existing field recovery. This is a 2D approximation; it does not
interpret 3D polynomial coefficients or Sader line forces as panel pressure.
Its dissipation is separate from the model-specific work used for energy Q.
