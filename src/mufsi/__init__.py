"""Public entry points for the µFSI v3 skeleton.

Imports require only the standard library. Numerical implementations are pending.
"""

from mufsi.coupling.operator import CouplingOperator
from mufsi.hydrodynamics.grid import FluidGrid
from mufsi.hydrodynamics.stokes_2d import Stokes2D
from mufsi.hydrodynamics.stokes_3d import Stokes3D
from mufsi.models.fluid import Fluid
from mufsi.models.geometry import BeamGeometry, PlateGeometry
from mufsi.models.material import Material
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
    "Fluid",
    "FluidGrid",
    "KirchhoffPlate",
    "Material",
    "PlateGeometry",
    "PointLoad",
    "Stokes2D",
    "Stokes3D",
]
