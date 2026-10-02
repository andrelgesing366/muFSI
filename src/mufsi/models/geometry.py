"""Geometry containers. All lengths are expressed in metres."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlateGeometry:
    """Dimensions of a rectangular plate; validation is pending."""

    length: float
    width: float
    thickness: float


@dataclass(frozen=True)
class BeamGeometry:
    """Dimensions of a beam with a rectangular cross-section."""

    length: float
    width: float
    thickness: float
