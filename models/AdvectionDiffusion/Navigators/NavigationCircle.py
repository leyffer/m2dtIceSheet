import sys

sys.path.insert(0, "../../../source/")

import numpy as np
from typing import Dict, Any, Optional, Literal

from Navigation import Navigation
from CirclePath import CirclePath

class NavigationCircle(Navigation):
    
    def __init__(self, 
        center: np.ndarray = np.array([0.75 / 2, 0.55 / 2]),
        grid_t: np.ndarray = np.arange(0, 4 + 1e-2, 1e-2),):
        
        self.center = center
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

    def d_position_d_control(self, flight):
        """
        computes the derivative of the flightpath with respect to the control parameters in alpha.
        This class is problem specific and needs to be written by the user.

        @param alpha:
        @param flightpath:
        @param grid_t:
        @return:
        """
        grid_t = flight.grid_t
        circle = CirclePath(alpha=flight.alpha, center=self.center, grid_t=grid_t)
        
        d_speed = circle.d_position_d_velocity(grid_t).T
        d_radius = circle.d_position_d_radius(grid_t).T

        return np.array([d_radius, d_speed])