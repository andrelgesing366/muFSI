"""Continuous polynomial traction with inverse-square-root rectangle edge weights."""

from __future__ import annotations

import numpy as np

from mufsi.hydrodynamics.base import HydrodynamicModel
from mufsi.hydrodynamics.weighted_pressure import (
    PlateMobility,
    PlatePressureBasis,
    Quadrature,
    WeightedBasis,
    _integer,
    solve_coefficients,
)


class Stokes3D(HydrodynamicModel):
    """Unbounded unsteady Stokes mobility for EB or KL motion, exp(+i omega t).

    p = sum a_mn T_m(2x/L-1) T_n(2y/W) / sqrt((1-xi²)(1-eta²)).
    x_degree/y_degree are maximum polynomial degrees for BOTH formulations.
    EB uses even y degrees <= y_degree; KL includes every y degree. This
    product weight is a numerical representation, not an exact corner exponent.

    H maps pressure coefficients [Pa] to collocation velocities [m/s].
    Oversampled, column-scaled least squares enforces no-slip approximately.
    Quadrature removes edge weights by cosine coordinates and treats the
    coincident Stokeslet by Duffy triangles. No kernel regularization is used.
    Only the most recent frequency's matrix/factorization is cached.
    """

    def __init__(
        self,
        fluid,
        geometry,
        *,
        formulation="KL",
        x_degree=16,
        y_degree=8,
        nx=None,
        ny=None,
        quadrature_backend="gauss",
        tolerance=2e-5,
        absolute_tolerance=1e-10,
        orders=(12, 20, 32, 48, 72, 104),
    ):
        formulation = formulation.upper()
        if formulation not in {"EB", "KL"}:
            raise ValueError("formulation must be 'EB' or 'KL'.")
        _integer("x_degree", x_degree)
        _integer("y_degree", y_degree)
        self.fluid, self.geometry, self.formulation = fluid, geometry, formulation
        self.x_degree, self.y_degree = x_degree, y_degree
        self.basis = (
            WeightedBasis(geometry, x_degree, y_degree // 2)
            if formulation == "EB"
            else PlatePressureBasis(geometry, x_degree, y_degree)
        )
        # EB retains only independent positive-y equations. KL uses full width.
        self.collocation_points = self.basis.collocation(nx, ny)
        self.collocation_points.setflags(write=False)
        self.quadrature = Quadrature(
            quadrature_backend, tuple(orders), tolerance, absolute_tolerance
        )
        self.mobility = PlateMobility(self.basis, fluid, self.quadrature)
        self.clear_cache()

    @property
    def coefficient_count(self):
        return self.basis.count

    def clear_cache(self):
        self._omega = None
        self._matrix = self._inverse_action = None
        self.integration_report = self.solve_report = None

    def assemble_matrix(self, omega):
        """Return rectangular coefficient mobility; not a panel-pressure matrix."""
        if self._omega != omega:
            matrix, report = self.mobility.assemble(omega, self.collocation_points)
            action, solve_report = solve_coefficients(matrix, np.eye(len(matrix)))
            matrix.setflags(write=False)
            action.setflags(write=False)
            self._omega, self._matrix, self._inverse_action = omega, matrix, action
            self.integration_report, self.solve_report = report, solve_report
        return self._matrix

    def coefficient_action(self, omega):
        """Cached least-squares map from collocation velocity to coefficients."""
        self.assemble_matrix(omega)
        return self._inverse_action

    def coefficients_from_velocity(self, omega, velocity):
        velocity = np.asarray(velocity, complex)
        if (
            velocity.ndim not in (1, 2)
            or velocity.shape[0] != len(self.collocation_points)
            or not np.isfinite(velocity).all()
        ):
            raise ValueError(
                "velocity must be finite with one row per collocation point."
            )
        return self.coefficient_action(omega) @ velocity

    def pressure_from_coefficients(self, coefficients, points=None):
        """Evaluate continuous pressure at strictly interior points, in Pa."""
        coefficients = np.asarray(coefficients, complex)
        if (
            coefficients.ndim not in (1, 2)
            or coefficients.shape[0] != self.coefficient_count
            or not np.isfinite(coefficients).all()
        ):
            raise ValueError("coefficients must be finite with one row per basis term.")
        points = self.collocation_points if points is None else points
        return self.basis.values(points) @ coefficients

    def pressure_from_velocity(self, omega, velocity):
        return self.pressure_from_coefficients(
            self.coefficients_from_velocity(omega, velocity)
        )

    def velocity_from_coefficients(self, omega, coefficients, points=None):
        """Evaluate velocity at independent interior points for no-slip checks."""
        matrix = (
            self.assemble_matrix(omega)
            if points is None
            else self.mobility.assemble(omega, points)[0]
        )
        return matrix @ np.asarray(coefficients, complex)
