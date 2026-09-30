"""User-facing definition of a coupled fluid-structure problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mufsi.coupling.operator import CouplingOperator
from mufsi.hydrodynamics.base import HydrodynamicModel
from mufsi.structure.base import StructuralModel
from mufsi.structure.loads import Load

if TYPE_CHECKING:
    from mufsi.solvers.frequency_response import FrequencyResponseResult


@dataclass
class CoupledProblem:
    """Combine structural, hydrodynamic, and coupling components.

    An omitted coupling operator is constructed lazily from the structure and
    fluid grid when solving. No assembly occurs in the constructor.
    """

    structure: StructuralModel
    hydrodynamics: HydrodynamicModel
    coupling: CouplingOperator | None = None

    def frequency_response(
        self, frequencies: Any, load: Load
    ) -> FrequencyResponseResult:
        """Solve a driven response at frequencies in Hz using an explicit load.

        Delegates assembly and solution to FrequencyResponseSolver.
        """
        from mufsi.solvers.frequency_response import FrequencyResponseSolver

        return FrequencyResponseSolver(self).solve(frequencies, load)
