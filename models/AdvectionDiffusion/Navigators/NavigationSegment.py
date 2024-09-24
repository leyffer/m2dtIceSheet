from __future__ import annotations
import sys

sys.path.insert(0, "../../../source/")

import numpy as np
from typing import Dict, Any, Optional, Literal
from Navigation import Navigation
from Flight import Flight


class NavigationSegment(Navigation):
    """
    0: starting position x value
    1: starting position y value
    2: starting heading
    3: velocity
    4: angular velocity
    """

    def __init__(self, grid_t: np.ndarray, *args, **kwargs):

        super().__init__(grid_t, *args, **kwargs)
        self.initial_time = self.grid_t[0]

    def get_trajectory(
            self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None, bool_ignore_validity_check=False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """! Get the trajectory of the drone given the flight parameters alpha

        @param alpha The specified flight parameters
        @param grid_t the time grid on which the drone position shall be computed
        @return  Tuple of (position over flight path, corresponding time for each position), grid_t
        """
        # use default time grid if none is provided
        if grid_t is None:
            grid_t = self.grid_t

        relative_position = self.relative_position(alpha=alpha, grid_t=grid_t)
        position = relative_position + np.array([alpha[0], alpha[1]])
        if bool_ignore_validity_check:
            return position, grid_t

        valid_positions = self.drone.fom.identify_valid_positions(position)
        return position, grid_t, valid_positions

    def d_position_d_initial_x(self, alpha, grid_t=None):

        # use default time grid if none is provided
        if grid_t is None:
            grid_t = self.grid_t

        deriv = np.ones((grid_t.shape[0], 2))
        deriv[:, 1] = 0
        return deriv

    def d_position_d_initial_y(self, alpha, grid_t=None):

        # use default time grid if none is provided
        if grid_t is None:
            grid_t = self.grid_t

        deriv = np.ones((grid_t.shape[0], 2))
        deriv[:, 0] = 0
        return deriv

    def relative_position(self, alpha, grid_t=None):

        # use default time grid if none is provided
        if grid_t is None:
            grid_t = self.grid_t

        # extract meaning of control parameters
        heading = alpha[2]
        velocity = alpha[3]
        angular_velocity = alpha[4]

        # initializations
        t = grid_t - self.initial_time
        positions = np.nan * np.ones((t.shape[0], 2))

        # avoid division by zero issues
        if np.abs(angular_velocity) < 1e-14:
            positions[:, 0] = (t * velocity * np.cos(heading))
            positions[:, 1] = (t * velocity * np.sin(heading))
            # todo: these don't match the derivatives
            return positions

        # rescaling
        radial_progression = t * angular_velocity + heading
        velocity /= angular_velocity

        # compute positions
        positions[:, 0] = velocity * (np.sin(radial_progression) - np.sin(heading))
        positions[:, 1] = velocity * (-np.cos(radial_progression) + np.cos(heading))

        return positions

    def final_heading(self, alpha):

        heading = alpha[2]
        angular_velocity = alpha[4]

        if np.abs(angular_velocity) < 1e-14:
            return heading
        return (self.grid_t[-1] - self.initial_time) * angular_velocity + heading

    def d_position_d_heading(self, alpha, grid_t=None):

        # use default time grid if none is provided
        if grid_t is None:
            grid_t = self.grid_t

        # Thomas found a clever way to extract the derivatives for the heading from the relative position
        # by exploiting that the heading is only used within sines and cosines, and applies the switch
        rel_pos = self.relative_position(alpha, grid_t)
        deriv = np.empty(rel_pos.shape)
        deriv[:, 0] = -rel_pos[:, 1]
        deriv[:, 1] = rel_pos[:, 0]

        return deriv

    def d_position_d_velocity(self, alpha, grid_t=None) -> np.ndarray:
        """
        Derivative of position with respect to velocity at time(s) t
        """
        # use default time grid if none is provided
        if grid_t is None:
            grid_t = self.grid_t

        # velocity is a simple multiplicant in the position
        deriv = self.relative_position(alpha, grid_t) / alpha[3]
        # todo: it might be more stable to compute the derivative using the formulas instead of this division
        return deriv

    def d_position_d_angular_velocity(self, alpha, grid_t=None) -> np.ndarray:
        """
        Derivative of position with respect to angular velocity at time(s) t
        """

        # use default time grid if none is provided
        if grid_t is None:
            grid_t = self.grid_t

        # extract meaning of control parameters
        heading = alpha[2]
        velocity = alpha[3]
        angular_velocity = alpha[4]

        # initializations
        t = grid_t - self.initial_time
        positions = np.nan * np.ones((t.shape[0], 2))
        radial_progression = t * angular_velocity + heading

        # avoid division by zero issues
        if np.abs(angular_velocity) < 1e-14:
            positions[:, 0] = -velocity * np.sin(radial_progression) * (t ** 2) / 2
            positions[:, 1] = +velocity * np.cos(radial_progression) * (t ** 2) / 2
            # todo: these don't match initial function
            return positions

        # weird rescaling
        velocity = velocity / angular_velocity

        # compute derivatives (first part, derivative from rescaling)
        positions[:, 0] = -velocity * (np.sin(radial_progression) - np.sin(heading)) / angular_velocity
        positions[:, 1] = -velocity * (-np.cos(radial_progression) + np.cos(heading)) / angular_velocity

        # compute derivatives (second part, derivative from radial progression)
        positions[:, 0] += velocity * np.cos(radial_progression) * t
        positions[:, 1] += velocity * np.sin(radial_progression) * t

        return positions

    def d_position_d_control(self, flight: Flight) -> np.ndarray:
        """
        computes the derivative of the flightpath with respect to the control
        parameters in alpha. This class is problem specific and needs to be
        written by the user.

        0: starting position x value
        1: starting position y value
        2: starting heading
        3: velocity
        4: angular velocity

        @param flight: Flight object
        @return: gradient matrix,  Shape $<n_spatial * n_timesteps> \times <n_controls>$
        """

        # initialization
        deriv = np.zeros(5, dtype=object)

        # compute derivatives with respect to the five controls
        deriv[0] = self.d_position_d_initial_x(alpha=flight.alpha, grid_t=flight.grid_t).T.flatten()
        deriv[1] = self.d_position_d_initial_y(alpha=flight.alpha, grid_t=flight.grid_t).T.flatten()
        deriv[2] = self.d_position_d_heading(alpha=flight.alpha, grid_t=flight.grid_t).T.flatten()
        deriv[3] = self.d_position_d_velocity(alpha=flight.alpha, grid_t=flight.grid_t).T.flatten()
        deriv[4] = self.d_position_d_angular_velocity(alpha=flight.alpha, grid_t=flight.grid_t).T.flatten()
        # note: the returns of d_position_d_<control-parameter> has shape (<number of time steps>x<spatial dim>)
        #  we flatten the returns such that they get stacked up, one position at a time

        # stack up the individual control derivatives to get matrix of shape
        # $<n_spatial * n_timesteps> \times <n_controls>$
        deriv = np.vstack(deriv).T

        return deriv

    def d_position_d_subcontrol(self, alpha, grid_t, subcontrols) -> np.ndarray:
        """
        computes the derivative of the flightpath with respect to the control
        parameters in alpha. This class is problem specific and needs to be
        written by the user.

        0: starting position x value
        1: starting position y value
        2: starting heading
        3: velocity
        4: angular velocity

        @param flight: Flight object
        @return: gradient matrix,  Shape $<n_spatial * n_timesteps> \times <n_controls>$
        """

        # initialization
        deriv = np.zeros(len(subcontrols), dtype=object)

        # compute derivatives with respect to the five controls
        counter = 0
        if 0 in subcontrols:
            deriv[counter] = self.d_position_d_initial_x(alpha=alpha, grid_t=grid_t).T
            counter += 1

        if 1 in subcontrols:
            deriv[counter] = self.d_position_d_initial_y(alpha=alpha, grid_t=grid_t).T
            counter += 1

        if 2 in subcontrols:
            deriv[counter] = self.d_position_d_heading(alpha=alpha, grid_t=grid_t).T
            counter += 1

        if 3 in subcontrols:
            deriv[counter] = self.d_position_d_velocity(alpha=alpha, grid_t=grid_t).T
            counter += 1

        if 4 in subcontrols:
            deriv[counter] = self.d_position_d_angular_velocity(alpha=alpha, grid_t=grid_t).T
            counter += 1
        # note: the returns of d_position_d_<control-parameter> has shape (<number of time steps>x<spatial dim>)
        #  we flatten the returns such that they get stacked up, one position at a time

        # stack up the individual control derivatives to get matrix of shape
        # $<n_spatial * n_timesteps> \times <n_controls>$
        deriv = np.vstack(deriv).T

        return deriv
