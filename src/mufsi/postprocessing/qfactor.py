"""Resonance and quality-factor extraction from frequency-response data."""

from typing import Any


def resonance_frequency(frequencies: Any, amplitudes: Any) -> float:
    """Estimate a selected resonance frequency in Hz.

    TODO: define peak selection and interpolation for the chosen observable.
    """
    raise NotImplementedError("Resonance-frequency extraction is pending.")


def q_factor(frequencies: Any, amplitudes: Any) -> float:
    """Estimate Q for an isolated resonance using a documented method.

    TODO: define the amplitude convention and check frequency resolution and
    applicability before implementing half-power bandwidth or a fitted model.
    """
    raise NotImplementedError("Quality-factor extraction is pending.")
