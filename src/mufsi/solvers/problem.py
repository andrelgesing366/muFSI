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

    An omitted coupling operator will eventually be constructed from the
    structure and fluid grid. No assembly occurs in this skeleton constructor.
    """

    structure: StructuralModel
    hydrodynamics: HydrodynamicModel
    coupling: CouplingOperator | None = None

    def frequency_response(
        self, frequencies: Any, load: Load
    ) -> FrequencyResponseResult:
        """Solve a driven response at frequencies in Hz using an explicit load.

        TODO: construct coupling if needed and delegate to FrequencyResponseSolver.
        """
        raise NotImplementedError("The coupled frequency-response workflow is pending.")
