"""Load data, independent of FEM assembly."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PointLoad:
    """Transverse point force: position in metres, complex amplitude in newtons."""

    position: tuple[float, ...]
    amplitude: complex


@dataclass(frozen=True)
class PointLoads:
    """Simultaneous point forces, each with its own position and phase."""

    loads: tuple[PointLoad, ...]

    def __post_init__(self):
        loads = tuple(self.loads)
        if not loads or not all(isinstance(load, PointLoad) for load in loads):
            raise ValueError("PointLoads requires a nonempty sequence of PointLoad.")
        object.__setattr__(self, "loads", loads)


@dataclass(frozen=True)
class DistributedLoad:
    """Transverse load callable evaluated at physical coordinates.

    Values use N/m^2 for plates and N/m for beams. The callable's vectorized
    array convention will be specified with the assembly implementation.
    """

    values: Callable[[Any], Any]


Load = PointLoad | PointLoads | DistributedLoad
