import sys

sys.path.insert(0, "../../../../source/")
sys.path.insert(0, "../Navigation/")

from typing import Optional, Tuple

import numpy as np
from Drone import Drone
from myState import State
from CirclePath import CirclePath

FlightPath = np.dtype([("position", "<f8", 2), ("time", "<f8")])


# todo: It seems to me like this class is completely redundant at this point. The only reason I see for why it's there
#  is the path function, which, if necessary for the optimization, should be part of the parent class Drone.

class MyDrone(Drone):
    """the special thing about this class is that it is specific to circle path
    but uses an interpretation of alpha where alpha contains speed and radius
    instead of speed and angular velocity"""
    center = np.array([0.75 / 2, 0.55 / 2])

    # todo: does this center ever get used?
    
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
        self.path_class = CirclePath
        # todo: path characterizations should be interchangeable, they should not be hard-coded in like this
        # todo: does this path_class variable have any information that is not already in the navigator?

    def path(self, alpha: np.ndarray) -> CirclePath:
        """Instantiate the path class with alpha"""
        # todo: if this is a necessary function for the optimization, is there a reason why this is not in the parent
        #  class?

        return self.path_class(alpha=alpha, center=self.center)

    def measure_pointwise(
        self, flightpath: np.ndarray, grid_t: np.ndarray, state: State
    ) -> np.ndarray:
        """! Get measurements along the flight path at the drone location

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        # TODO: remove this function and make sure it doesn't get called upon anywhere else
        raise RuntimeError("measure_pointwise is depricated since it is no longer consistent with naming convention")

    
