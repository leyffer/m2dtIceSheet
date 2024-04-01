import numpy as np


class Detector:
    """
    Takes measurements of a state at locations
    """
    drone = None

    def __init__(self, *args, **kwargs):
        return

    def attach_to_drone(self, drone):
        if self.drone is not None:
            pass
            # raise RuntimeWarning("Navigation system was attached to a new drone. Was this intentional? If attaching "
            #                      "the navigation system to several drones, make sure they'll not accidentally change "
            #                      "each other (make an appropriate copy or make sure they have no changable "
            #                      "parameters)")
        self.drone = drone

    def measure(
        self, flight : "Flight", state: "State"
    ) -> np.ndarray:
        """! Method to take a measurement

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        raise NotImplementedError("Drone.measure: Needs to be implemented in subclass")

    def d_measurement_d_control(self, flight, state):
        """
        derivative of the measurement function for a given flightpath in control direction alpha

        @param alpha:
        @param flightpath:
        @param grid_t:
        @param state:
        @return: np.ndarray of shape (grid_t.shape[0], self.n_parameters)
        """
        # TODO: Take out this function
        raise NotImplementedError("Deprecated")

    def d_measurement_d_position(self, flight, state):
        """
        derivative of the measurement function for a given flightpath in control direction alpha

        @param alpha:
        @param flightpath:
        @param grid_t:
        @param state:
        @return: np.ndarray of shape (grid_t.shape[0], self.n_parameters)
        """
        # TODO: I think we can generalize the code from myDrone in models/AdvectionDiffusion_FEniCS such that the user
        #  doesn't need to write this function themselves.
        raise NotImplementedError(
            "Drone.d_measurement_d_control: Needs to be implemented in subclass"
        )
