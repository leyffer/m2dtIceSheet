import numpy as np

class Drone():
    """!
    This is a general parent class for the drones.
    For any particular model the user should create a subclass and specify the functions below.
    """

    def __init__(self, fom):
        """! Initialization for the drone class
        In this call we specify the main setup parameters that the drone class has to have.
        The user needs to specify their own __init__ call, and call super().__init__ for the setup here.

        @param fom  Full-order-model (FOM) object. The drone needs to measure states computed by the FOM
        """
        # TODO: are there any other setup parameters?
        self.fom = fom

    # TODO: specify and describe other functions the general drone class has to have

    def get_position(self, t: float, alpha: np.ndarray):
        """! Get the position of the drone given the time and flying parameters

        @param t  The time to evaluate the position of the drone
        @param alpha  The parameters of the flight path
        @return  spatial position of the drone
        """
        pos, __ = self.get_trajectory(alpha=alpha, grid_t=t * np.ones((1,)))
        return pos[0, :]

    def get_trajectory(self, alpha: np.ndarray, grid_t: np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        """! Get the trajectory of the drone given the flight parameters alpha
        @param alpha The specified flight parameters
        @param grid_t the time grid on which the drone position shall be computed
        @return  Tuple of (position over flight path, corresponding time for each position)
        """
        raise NotImplementedError("Drone.get_trajectory: Needs to be implemented in subclass")

    def measure(self, flightpath, grid_t, state) -> np.ndarray:
        """! Method to take a measurement

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        raise NotImplementedError("Drone.measure: Needs to be implemented in subclass")
