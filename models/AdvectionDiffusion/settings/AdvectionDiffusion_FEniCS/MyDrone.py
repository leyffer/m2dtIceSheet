import sys

sys.path.insert(0, "../../../../source/")
sys.path.insert(0, "../Navigation/")

from typing import Optional, Tuple

import numpy as np
from Drone import Drone
from myState import State
from CirclePath import CirclePath

FlightPath = np.dtype([("position", "<f8", 2), ("time", "<f8")])


class MyDrone(Drone):
    """the special thing about this class is that it is specific to circle path
    but uses an interpretation of alpha where alpha contains speed and radius
    instead of speed and angular velocity"""
    center = np.array([0.75 / 2, 0.55 / 2])
    
    def __init__(
        self, 
        navigation: "Navigation", 
        detector: "Detector",
        grid_t=None,
        **kwargs,
    ):
        """! Initializer for the drone class
        @param fom  Full-order-model (FOM) object. The drone takes
        measurements from this
        @param eval_mode  Evaluation mode of the drone's measurements:

            - `"gaussian, truncated"`: The drone takes a measurement that is dispersed
              over a 2D truncated gaussian
            - `"gaussian"`: The drone takes a measurement that is dispersed
              over a 2D gaussian
            - `"uniform"`: The drone takes a measurement that is dispersed
              uniformly over a circle
            - `"point-eval"`: The drone takes a measurement at its exact
              location

        @param grid_t the time grid the drone should fly in
        @param **kwargs  Keyword arguments including `sigma_gaussian`
        (Gaussian radius) and `radius_uniform`
        """
        super().__init__(navigation=navigation, detector=detector)

        #self.grid_t = grid_t if grid_t is not None else np.arange(0, 4 + 1e-2, 1e-2)

        self.path_class = CirclePath

        # TODO: get parameterization for other eval modes, in particular give them a common name, not individual ones:
        # self.sigma_gaussian = kwargs.get("sigma_gaussian", 0.1)
        # self.radius_uniform = kwargs.get("radius_uniform", 0.1)

    def path(self, alpha: np.ndarray) -> CirclePath:
        """Instantiate the path class with alpha"""
        return self.path_class(alpha=alpha, center=self.center)

    # def get_trajectory(
    #     self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """! Get the trajectory of the drone given the flight parameters alpha
    #     @param alpha  The specified flight parameters
    #     @param grid_t  the time grid on which the drone position shall be computed
    #     @return  Position over flight path
    #     """
    #     # default time grid if None is provided
    #     if grid_t is None:
    #         grid_t = self.grid_t

    #     return self.path(alpha).position(grid_t), grid_t

    # def d_position_d_control(
    #     self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None
    # ):
    #     """
    #     Computes the derivative of the flightpath with respect to the control parameters in alpha.
    #     This class is problem specific and needs to be written by the user.

    #     @param alpha:
    #     @param grid_t:
    #     @return:
    #     """
    #     # for the Drone class
    #     if grid_t is None:
    #         grid_t = self.grid_t

    #     d_speed = self.path(alpha).d_position_d_velocity(grid_t).T
    #     d_radius = self.path(alpha).d_position_d_radius(grid_t).T

    #     return np.array([d_radius, d_speed])


    def measure_pointwise(
        self, flightpath: np.ndarray, grid_t: np.ndarray, state: State
    ) -> np.ndarray:
        """! Get measurements along the flight path at the drone location

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        raise RuntimeError("measure_pointwise is depricated since it is no longer consistent with naming convention")

    
