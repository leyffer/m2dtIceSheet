import numpy as np
from typing import Optional
from State import State


class Drone:
    """! This is a general parent class for the drones.

    For any particular model the user should create a subclass and specify the
    functions below.
    """

    def __init__(self, navigation: "Navigation", detector: "Detector"):
        """! Initialization for the drone class
        In this call we specify the main setup parameters that the drone class
        has to have. The user needs to specify their own __init__ call, and call
        super().__init__ for the setup here.

        @param fom  Full-order-model (FOM) object. The drone needs to measure
        states computed by the FOM
        """
        # TODO: are there any other setup parameters?
        self.navigation = navigation
        self.detector = detector
        self.grid_t = navigation.grid_t

    # TODO: specify and describe other functions the general drone class has to have

    def get_position(self, t: float | np.ndarray, flight : "Flight"):
        """! Get the position of the drone given the time and flying parameters

        @param t  The time to evaluate the position of the drone
        @param alpha  The parameters of the flight path
        @return  spatial position of the drone
        """
        return flight.get_position(t)

    def get_trajectory(
        self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """! Get the trajectory of the drone given the flight parameters alpha
        @param alpha The specified flight parameters
        @param grid_t the time grid on which the drone position shall be computed
        @return  Tuple of (position over flight path, corresponding time for each position)
        """
        return self.navigation.get_trajectory(alpha, grid_t)

    def measure(
        self, flight : "Flight", state: State
    ) -> np.ndarray:
        """! Method to take a measurement

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        return self.detector.measure(flight, state)

    def d_position_d_control(self, flight : "Flight"):
        """
        computes the derivative of the flightpath with respect to the control parameters in alpha.
        This class is problem specific and needs to be written by the user.

        @param alpha:
        @param flightpath:
        @param grid_t:
        @return:
        """
        return flight.d_position_d_control()

    def d_measurement_d_control(self, flight : "Flight", state):
        """
        derivative of the measurement function for a given flightpath in control direction alpha

        @param alpha:
        @param flightpath:
        @param grid_t:
        @param state:
        @return: np.ndarray of shape (grid_t.shape[0], self.n_parameters)
        """
        return self.detector.d_measurement_d_control(flight=flight, state=state, navigation=self.navigation)
    
    def d_measurement_d_position(self, flight : "Flight", state):
        """
        """
        return self.detector.d_measurement_d_position(flight, state)
