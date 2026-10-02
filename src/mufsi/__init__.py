"""Public entry points; the matched DOLFINx backend is loaded on demand."""

from mufsi.coupling.operator import CouplingOperator
from mufsi.coupling.weighted import WeightedCouplingOperator
from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.hydrodynamics.sader import SaderMethod, gamma_function
from mufsi.hydrodynamics.section_force import SectionForce2D
from mufsi.hydrodynamics.stokes_2d import Stokes2D
from mufsi.hydrodynamics.stokes_3d import Stokes3D
from mufsi.models.fluid import Fluid
from mufsi.models.geometry import BeamGeometry, PlateGeometry
from mufsi.models.material import Material
from mufsi.postprocessing.flow import (
    FlowField2D,
    plot_flow,
    reconstruct_flow,
    reconstruct_flow_from_response,
    reconstruct_section,
)
from mufsi.postprocessing.qfactor import (
    EnergyQResult,
    QFactorResult,
    SHOFitResult,
    analyze_q_factor,
    corner_displacement,
    corner_loads,
    energy_from_response,
    energy_q_factor,
    fit_sho,
    q_factor,
    resonance_frequency,
)
from mufsi.solvers.beam_frequency_response import (
    BeamFrequencyResponseResult,
    BeamFrequencyResponseSolver,
)
from mufsi.solvers.eigen import EigenResult, EigenSolver
from mufsi.solvers.frequency_response import (
    FrequencyResponseResult,
    FrequencyResponseSolver,
)
from mufsi.solvers.problem import CoupledProblem
from mufsi.structure.euler_bernoulli import EulerBernoulliBeam
from mufsi.structure.kirchhoff import KirchhoffPlate
from mufsi.structure.loads import DistributedLoad, PointLoad, PointLoads

__all__ = [
    "BeamFrequencyResponseResult",
    "BeamFrequencyResponseSolver",
    "BeamGeometry",
    "CoupledProblem",
    "CouplingOperator",
    "DistributedLoad",
    "EigenResult",
    "EigenSolver",
    "EnergyQResult",
    "EulerBernoulliBeam",
    "FlowField2D",
    "Fluid",
    "FluidGrid",
    "FrequencyResponseResult",
    "FrequencyResponseSolver",
    "KirchhoffPlate",
    "Material",
    "PlateGeometry",
    "PointLoad",
    "PointLoads",
    "QFactorResult",
    "SHOFitResult",
    "SaderMethod",
    "SectionForce2D",
    "Stokes2D",
    "Stokes3D",
    "WeightedCouplingOperator",
    "analyze_q_factor",
    "corner_displacement",
    "corner_loads",
    "energy_from_response",
    "energy_q_factor",
    "fit_sho",
    "gamma_function",
    "plot_flow",
    "q_factor",
    "reconstruct_flow",
    "reconstruct_flow_from_response",
    "reconstruct_section",
    "resonance_frequency",
]
