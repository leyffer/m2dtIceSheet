import sys

sys.path.insert(0, "../../../source/")

import numpy as np
from typing import Dict, Any, Optional

from Navigation import Navigation
from CirclePath import CirclePath


class NavigationCircle(Navigation):

    def __init__(
        self,
        center: np.ndarray = None,
        grid_t: np.ndarray = None,
    ):
        """!Initialize the circle navigator"""

        # Default values for circle center and time grid
        if center is None:
            center = np.array([0.75 / 2, 0.55 / 2])
        if grid_t is None:
            grid_t = np.arange(0, 4 + 1e-2, 1e-2)

        # individual settings for this child of Navigation
        self.center = center  # the center around which the drone is flying
        self.n_spatial = 2  # we are in 2D space
        self.n_controls = 2  # radius and speed

        # call parent class to initialize the rest
        super().__init__(grid_t=grid_t)

    def get_trajectory(
        self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """! Get the trajectory of the drone given the flight parameters alpha
        @param alpha The specified flight parameters
        @param grid_t the time grid on which the drone position shall be computed
        @return  Tuple of (position over flight path, corresponding time for each position)
        """
        if grid_t is None:
            grid_t = self.grid_t
        circle = CirclePath(alpha=alpha, center=self.center, grid_t=grid_t)

        return circle.position(grid_t), grid_t

    def d_position_d_control(self, flight: "Flight"):
        """
        Computes the derivative of the flightpath with respect to the control parameters in alpha.
        This class is problem specific and needs to be written by the user.

        Since the position is determined by <spatial dimension>*<number of time steps> parameters, and there are
        <number of controls> control parameters, the return of this function has to have shape

        $$ <spatial dimension>*<number of time steps> \times <number of controls> $$

        @param flight: flight containing controls `alpha`, time grid `grid_t`, and positions `flightpath`

        @return: derivative of positions `flightpath` with respect to the flight controls alpha
        """
        grid_t = flight.grid_t
        circle = CirclePath(alpha=flight.alpha, center=self.center, grid_t=grid_t)

        # get the derivatives for each control parameter
        d_speed = circle.d_position_d_velocity(grid_t).T
        d_radius = circle.d_position_d_radius(grid_t).T

        # reshape such that derivatives in x direction come first, then derivatives in y direction
        d_speed = np.reshape(d_speed, (self.n_timesteps * self.n_spatial,))
        d_radius = np.reshape(d_radius, (self.n_timesteps * self.n_spatial,))

        # stack everything next to each other
        # return np.vstack([d_speed, d_radius]).T

        return np.vstack([d_radius, d_speed]).T
