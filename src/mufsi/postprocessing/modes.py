"""Evaluate computed structural modes for visualization or comparison."""

from typing import Any

from mufsi.structure.base import StructuralModel


def evaluate_mode(structure: StructuralModel, mode: Any, points: Any) -> Any:
    """Evaluate a structural mode at physical points via basis evaluation."""
    raise NotImplementedError("Structural mode evaluation is pending.")
