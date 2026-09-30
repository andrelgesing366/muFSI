"""Public entry points; the matched DOLFINx backend is loaded on demand."""

from mufsi.coupling.operator import CouplingOperator
from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.hydrodynamics.stokes_2d import Stokes2D
from mufsi.hydrodynamics.stokes_3d import Stokes3D
from mufsi.hydrodynamics.sader import SaderMethod, gamma_function
from mufsi.models.fluid import Fluid
from mufsi.models.geometry import BeamGeometry, PlateGeometry
from mufsi.models.material import Material
from mufsi.solvers.eigen import EigenResult, EigenSolver
from mufsi.solvers.frequency_response import (
    FrequencyResponseResult, FrequencyResponseSolver,
)
from mufsi.solvers.problem import CoupledProblem
from mufsi.structure.euler_bernoulli import EulerBernoulliBeam
from mufsi.structure.kirchhoff import KirchhoffPlate
from mufsi.structure.loads import DistributedLoad, PointLoad

__all__ = [
    "BeamGeometry",
    "CoupledProblem",
    "CouplingOperator",
    "DistributedLoad",
    "EulerBernoulliBeam",
    "EigenResult",
    "EigenSolver",
    "Fluid",
    "FluidGrid",
    "FrequencyResponseResult",
    "FrequencyResponseSolver",
    "KirchhoffPlate",
    "Material",
    "PlateGeometry",
    "PointLoad",
    "SaderMethod",
    "gamma_function",
    "Stokes2D",
    "Stokes3D",
]
