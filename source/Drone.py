import numpy as np
from typing import Optional
from State import State
from Flight import Flight


class Drone:
    """! This is a class for accessing the "public" functionalities navigating and measuring

    The drone class uses the navigation system (class Navigation) and the detector measuring system (class Detector) to
    take measurements of states on the domain $\Omega$. Its main functionality is to inferface between the two systems
    and the outside world to ensure that everything remains in their place. Ideally, the user does not need to alter
    this class in any way, but only passes an instance of "Navigation" and "Detector" -- both are problem specific.
    """

    def __init__(self, navigation: "Navigation", detector: "Detector"):
        """! Initialization for the drone class
        In most cases, the drone is already defined by the navigation system and the detector, nothing else needs to
        be passed. The navigation system and the detector will both be told that they have been equipped by this drone

        @param navigation: for how the drone flies, type Navigation
        @param detector: for how the drone measures, type Detector
        """
        self.navigation = navigation
        self.detector = detector
        self.grid_t = navigation.grid_t

        # tell the detector and the navigation system that they've just been equipped
        self.detector.attach_to_drone(self)
        self.navigation.attach_to_drone(self)
        # this gives the two systems the possiblilty to see beyond their own scope (caution advised), and can ensure
        # that any single navigation / detector system is only equipped by one drone at a time if necessary
        # (see setup parameter bool_allow_multiple_attachments in Detector and Navigation)

    def get_position(self, t: float | np.ndarray, flight : "Flight"):
        """! Get the position of the drone given the time and flying parameters

        @param t  The time at which to evaluate the position of the drone
        @param flight  the
        """
        print("In Drone.get_position: Should call flight.d_position_d_control() instead")
        return flight.get_position(t)

    def get_trajectory(
        self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """! Get the trajectory of the drone given the flight parameters alpha
        @param alpha The specified flight parameters
        @param grid_t the time grid on which the drone position shall be computed
        @return  Tuple of (position over flight path, corresponding time for each position)
        """
        raise DeprecationWarning("Drone.get_trajectory is depricated: should get flight and call flight.get_trajectory instead")
        # return self.navigation.get_trajectory(alpha, grid_t)

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
        print("In drone.d_position_d_control: Should call flight.d_position_d_control() instead")
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
        # # derivative of the measurement with respect to the position
        # dmdp = self.d_measurement_d_position(flight=flight, state=state)
        #
        # # derivative of the position with respect to the control
        # dpdc = flight.d_position_d_control()
        #
        # # chain rule
        # dmdc = np.zeros((dmdp.shape[0], dpdc.shape[0]))
        # for k in range(dmdp.shape[0]):
        #     dmdc[k, :] = dmdp[k, :] @ dpdc[:, :, k].T
        #     # TODO: the transpose here is increadibly error prone. We should change the order of computations such that
        #     #  we can get rid of it.
        # # TODO: This loop slows everything down. We should change the structure of dpdc such that we can compute dmdc
        # #  as a single tensor multiplication
        #
        # return dmdc

        # TODO: we should switch to the code above (which applies the chain rule for getting the derivative of the
        #   measurements w.r.t. the control by computing the derivatives of the measurements w.r.t. the position, and
        #  the derivative of the position w.r.t. the control separately). However, right now the functions
        #  "d_measurement_d_position" for our 4 measurement types don't have these functions (they are on Thomas'
        #  computer). Once we've transferred them over, we can put in the code above and take out
        #  "d_measurement_d_control" in all four classes. By doing it this way, we avoid for Detector to compute the
        #  derivatives in two serparate scenarios.

        return self.detector.d_measurement_d_control(flight=flight, state=state, navigation=self.navigation)
    
    def d_measurement_d_position(self, flight : "Flight", state):
        """
        derivative of the measurement function for a given flight in direction of the flight's positions flightpath.
        For measurements of the form
        $$
        d(t; p) = \int_{\Omega} \Phi(x, p(t)) u(x,t) dx
        $$
        this function returns
        $$
        \frac{\partial d(t;, p)}{\partial p}
        = \int_{\Omega} D_y \Phi(x, y=p(t)) u(x, t) dx.
        $$

        @param flight: the flight parameterization of the drone. Contains, in particular, the flightpath `flightpath`,
        the flight controls `alpha`, and the time discretization `grid_t`, Flight object
        @param state  The state which the drone shall measure, State object
        @return: np.ndarray of shape (grid_t.shape[0], <spatial dimension>)
        """
        return self.detector.d_measurement_d_position(flight, state)

    def plan_flight(self, alpha) -> Flight:
        """
        creates a Flight object for a given control parameter alpha
        """
        return self.navigation.plan_flight(alpha)
