import numpy as np
from typing import Optional
from Flight import Flight
#TODO

class Navigation:
    """
    
    """

    drone = None

    def __init__(self, grid_t):
        
        self.grid_t=grid_t
        # TODO: write initializer
        return

    def attach_to_drone(self, drone):
        if self.drone is not None:
            pass
            # raise RuntimeWarning("Navigation system was attached to a new drone. Was this intentional? If attaching "
            #                      "the navigation system to several drones, make sure they'll not accidentally change "
            #                      "each other (make an appropriate copy or make sure they have no changable "
            #                      "parameters)")
        self.drone = drone

    def get_trajectory(
        self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """! Get the trajectory of the drone given the flight parameters alpha
        @param alpha The specified flight parameters
        @param grid_t the time grid on which the drone position shall be computed
        @return  Tuple of (position over flight path, corresponding time for each position), grid_t
        """
        raise NotImplementedError(
            "Drone.get_trajectory: Needs to be implemented in subclass"
        )

    def d_position_d_control(self, flight : Flight):
        """
        computes the derivative of the flightpath with respect to the control parameters in alpha.
        This class is problem specific and needs to be written by the user.

        @param alpha:
        @param flightpath:
        @param grid_t:
        @return:
        """
        raise NotImplementedError(
            "Drone.d_position_d_control: Needs to be implemented in subclass"
        )
        
    def plan_flight(self, alpha) -> Flight:
        flight = Flight(alpha=alpha, navigation=self)
        return flight