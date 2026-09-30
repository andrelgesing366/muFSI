"""Frequency-domain FSI orchestration and result data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mufsi.structure.loads import Load

if TYPE_CHECKING:
    from mufsi.solvers.problem import CoupledProblem


@dataclass(frozen=True)
class FrequencyResponseResult:
    """Proposed response storage; numerical array shapes are pending.

    frequencies uses Hz; displacement uses metres in structural DOF order;
    optional pressure uses Pa in fluid-grid order. Frequency is the first axis.
    """

    frequencies: Any
    displacement: Any
    pressure: Any | None = None


@dataclass
class FrequencyResponseSolver:
    """Coordinate structure, fluid response, and coupling at each frequency.

    TODO: define the harmonic/sign convention and constrained DOF handling,
    then assemble or apply the coupled operator with reusable linear solves.
    """

    problem: CoupledProblem

    def solve(self, frequencies: Any, load: Load) -> FrequencyResponseResult:
        """Convert Hz to angular frequency and solve the driven FSI systems."""
        raise NotImplementedError("Frequency-response solution is pending.")
