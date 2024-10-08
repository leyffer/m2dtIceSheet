from __future__ import annotations
from functools import cached_property
import sys

sys.path.insert(0, "../../../source/")

import numpy as np
import scipy.sparse as sparse
from typing import Dict, Any, Optional, Literal
from Navigation import Navigation
from Flight import Flight


class NavigationDAE(Navigation):
    lower = None
    upper = None

    def __init__(self, grid_t: np.ndarray, *args, **kwargs):
        """Initialization of the Navigation class

        When writing the child class specific for the application, remember to
        call super().__init__

        The time discretization grid_t will be used for all flights. We are
        assuming uniform time stepping. This might change in the future.

        @param grid_t : the time discretization for the drone, np.ndarray with
        len(grid_t.shape)=1

        Options to be passed in **kwargs:

        bool_allow_multiple_attachments:
            Whenever a drone equips a detector, it will tell the detector that
            it was just equipped through a call to detector.attach_to_drone. By
            default, we allow a single detector to be attached to multiple
            drones. However, if the user expects detector parameters to change,
            they might want to set
        bool_allow_multiple_attachments = False
            to ensure that any instance of Detector can only be equipped by a
            single drone. This avoids copy-issues and saves time on debugging.
            It will, however, make code testing harder within notebooks when
            running blocks out of order, which is why, per default, we are
            enabling multiple attachments.
        """
        super().__init__(grid_t, *args, **kwargs)

        self.n_controls = 2 * self.n_timesteps + 3
        self.n_constraints = 3 * self.n_timesteps

    def split_controls(self, alpha):
        # todo: this split assumes we are in 2D. Generalize.
        n_timesteps = self.n_timesteps

        initial_position = alpha[:3]
        velocity = alpha[3:3 + n_timesteps]
        angular_velocity = alpha[3 + n_timesteps:3 + 2 * n_timesteps]

        return initial_position, velocity, angular_velocity

    def solve_DAE(self, alpha):
        initial_position, velocity, angular_velocity = self.split_controls(alpha)
        sol = np.zeros((self.n_timesteps, 3))
        dt = self.grid_t[1] - self.grid_t[0]

        # initial position
        sol[0, :] = initial_position

        for i in range(1, self.n_timesteps):
            sol[i, 0] = sol[i - 1, 0] + dt * velocity[i - 1] * np.cos(sol[i - 1, 2])
            sol[i, 1] = sol[i - 1, 1] + dt * velocity[i - 1] * np.sin(sol[i - 1, 2])
            sol[i, 2] = sol[i - 1, 2] + dt * angular_velocity[i - 1]

        return sol

    def get_trajectory(
            self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """! Get the trajectory of the drone given the flight parameters alpha

        @param alpha The specified flight parameters
        @param grid_t the time grid on which the drone position shall be computed
        @return  Tuple of (position over flight path, corresponding time for each position), grid_t
        """
        x_y_theta = self.solve_DAE(alpha)
        positions = x_y_theta[:, :2]
        valid_positions = self.drone.fom.identify_valid_positions(positions)
        return x_y_theta, self.grid_t, valid_positions

    def d_position_d_position_and_control(self, flight: Flight) -> np.ndarray:
        """
        computes the derivative of the flightpath with respect to the its positions
        and the control parameters in alpha. If all positions are computed independently
        from each other, this is just going to be the zero matrix stacked next to
        the d_position_d_control output. However, in some cases the position at time $t_k$
        may be computed as an adjustment to the position at time $t_{k-1}$ (for example),
        in which case the derivative of position w.r.t. position is not the identity. These
        special cases need to be implemented in the subclass.

        @param flight: Flight object
        @return: gradient vector
        """
        data = self.derivative_sparsity_data(flight)
        deriv = sparse.coo_matrix((data, self.derivative_sparsity_pattern),
                                  shape=(3 * self.n_timesteps, 5 * self.n_timesteps + 3))

        return deriv

    def derivative_sparsity_data(self, flight: Flight):
        dt = self.grid_t[1] - self.grid_t[0]
        x_y_theta = flight.flightpath
        theta = x_y_theta[:-1, 2]
        initial_position, velocity, angular_velocity = self.split_controls(flight.alpha)

        data = np.ones((3 + 3 * (self.n_timesteps - 1),))
        yolo = dt * np.ones((self.n_timesteps - 1))
        data = np.hstack([data, yolo])
        data = np.hstack([data, dt * np.cos(theta)])
        data = np.hstack([data, dt * np.sin(theta)])
        data = np.hstack([data, -dt * velocity[:-1] * np.sin(theta)])
        data = np.hstack([data, dt * velocity[:-1] * np.cos(theta)])

        return data

    @cached_property
    def derivative_sparsity_pattern(self):
        splits_row = [0, self.n_timesteps, 2 * self.n_timesteps, 3 * self.n_timesteps]
        splits_col = [0, self.n_timesteps, 2 * self.n_timesteps, 3 * self.n_timesteps,
                      3 + 3 * self.n_timesteps, 3 + 4 * self.n_timesteps, 3 + 5 * self.n_timesteps]

        # initial positions
        rows = [splits_row[0], splits_row[1], splits_row[2]]
        cols = [splits_col[3], splits_col[3] + 1, splits_col[3] + 2]

        # last x-direction
        rows += [*range(splits_row[0] + 1, splits_row[1])]
        cols += [*range(splits_row[0], splits_row[1] - 1)]

        # last y-direction
        rows += [*range(splits_row[1] + 1, splits_row[2])]
        cols += [*range(splits_row[1], splits_row[2] - 1)]

        # last theta-direction
        rows += [*range(splits_row[2] + 1, splits_row[3])]
        cols += [*range(splits_row[2], splits_row[3] - 1)]

        # forward Euler for theta (angular velocity)
        rows += [*range(splits_row[2] + 1, splits_row[3])]
        cols += [*range(splits_col[5], splits_col[6] - 1)]

        # forward Euler for x-direction (velocity)
        rows += [*range(splits_row[0] + 1, splits_row[1])]
        cols += [*range(splits_col[4], splits_col[5] - 1)]

        # forward Euler for y-direction (velocity)
        rows += [*range(splits_row[1] + 1, splits_row[2])]
        cols += [*range(splits_col[4], splits_col[5] - 1)]

        # forward Euler for x-direction (theta)
        rows += [*range(splits_row[0] + 1, splits_row[1])]
        cols += [*range(splits_col[2], splits_col[3] - 1)]

        # forward Euler for y-direction (theta)
        rows += [*range(splits_row[1] + 1, splits_row[2])]
        cols += [*range(splits_col[2], splits_col[3] - 1)]

        return (np.array(rows), np.array(cols))

    @cached_property
    def positional_constraint_sparsity_pattern(self):
        rows, cols = self.derivative_sparsity_pattern
        extension = np.arange(3 * self.n_timesteps)
        rows_new = np.hstack([extension, rows])
        cols_new = np.hstack([extension, cols])
        return (rows_new, cols_new)

    def d_positional_constraint(self, flight: Flight):
        data = self.derivative_sparsity_data(flight)
        data = np.hstack([np.ones(3 * self.n_timesteps), -data])

        jacobian = sparse.coo_matrix((data, self.positional_constraint_sparsity_pattern),
                                     shape=(3 * self.n_timesteps, 5 * self.n_timesteps + 3))

        return jacobian

    def evaluate_positional_constraints(self, flightpath_1d, alpha):
        pos_x = flightpath_1d[:self.n_timesteps]
        pos_y = flightpath_1d[self.n_timesteps:2 * self.n_timesteps]
        theta = flightpath_1d[2 * self.n_timesteps:3 * self.n_timesteps]
        initial_position, velocity, angular_velocity = self.split_controls(alpha)
        dt = self.grid_t[1] - self.grid_t[0]

        diff = np.zeros((self.n_timesteps, 3))
        diff[0, 0] = pos_x[0] - initial_position[0]
        diff[0, 1] = pos_y[0] - initial_position[1]
        diff[0, 2] = theta[0] - initial_position[2]

        diff[1:, 0] = pos_x[1:] - pos_x[:-1] - dt * velocity[:-1] * np.cos(theta[:-1])
        diff[1:, 1] = pos_y[1:] - pos_y[:-1] - dt * velocity[:-1] * np.sin(theta[:-1])
        diff[1:, 2] = theta[1:] - theta[:-1] - dt * angular_velocity[:-1]

        return np.hstack([diff[:, 0], diff[:, 1], diff[:, 2]])

    def set_control_bounds(self, control_min, control_max):
        self.lower = np.hstack([control_min[:3],
                                np.array([control_min[3]] * self.n_timesteps),
                                np.array([control_min[4]] * self.n_timesteps)])

        self.upper = np.hstack([control_max[:3],
                                np.array([control_max[3]] * self.n_timesteps),
                                np.array([control_max[4]] * self.n_timesteps)])

    def d_position_d_control(self, flight: Flight) -> np.ndarray:
        """
        computes the derivative of the flightpath with respect to the control
        parameters in alpha. This class is problem specific and needs to be
        written by the user.

        @param flight: Flight object
        @return: gradient vector
        """
        raise RuntimeError(
            "NavigationDAE.d_position_d_control is not defined for DAE system, due to the dependence on the positions "
            "in each step. Use NavigationDAE.d_position_d_position_and_constrol instead."
        )
