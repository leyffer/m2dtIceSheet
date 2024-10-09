"""Drone class

The `Drone` class has a `Navigation` and `Detector` class attached to it that
move the measurement location and take measurements respectively.
"""

from typing import Optional, Union

import numpy as np

from Detector import Detector
from Flight import Flight
from Navigation import Navigation
from State import State
from FullOrderModel import FullOrderModel

class Drone:
    r"""! This is a class for accessing the "public" functionalities navigating
    and measuring

    The drone class uses the navigation system (class `Navigation`) and the
    detector measuring system (class `Detector`) to take measurements of states on
    the domain $\Omega$. Its main functionality is to interface between the two
    systems and the outside world to ensure that everything remains in their
    place. Ideally, the user does not need to alter this class in any way, but
    only passes an instance of "Navigation" and "Detector" -- both are problem
    specific.
    """

    def __init__(self, navigation: "Navigation", detector: "Detector", fom: "FullOrderModel"):
        """! Initialization for the drone class

        In most cases, the drone is already defined by the navigation system and
        the detector, nothing else needs to be passed. The navigation system and
        the detector will both be told that they have been equipped by this
        drone

        @param navigation: for how the drone flies
        @param detector: for how the drone measures
        """
        self.fom = fom
        self.navigation = navigation
        self.detector = detector
        self.grid_t = navigation.grid_t

        # Tell the detector navigator that they have been equipped
        self.detector.attach_to_drone(self)
        self.navigation.attach_to_drone(self)
        # this gives the two systems the possibility to see beyond their own
        # scope (caution advised), and can ensure that any single navigation /
        # detector system is only equipped by one drone at a time if necessary
        # (see setup parameter bool_allow_multiple_attachments in Detector and
        # Navigation)

    def get_position(self, t: Union[float, np.ndarray], flight: "Flight"):
        """! Get the position of the drone given the time and flying parameters

        @param t  The time at which to evaluate the position of the drone
        @param flight  The `Flight` object to get the location of at time `t`
        """
        print("In Drone.get_position: Should call flight.d_position_d_control instead")
        return flight.get_position(t)

    def get_trajectory(
        self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """! **DEPRECATED**
        Get the trajectory of the drone given the flight parameters alpha

        Flight control parameters alpha and time grid_t determine the flightpath
        trajectory.

        @param alpha The specified flight parameters
        @param grid_t the time grid on which the drone position shall be
            computed
        @return  Tuple of (position over flight path, corresponding time for
            each position)
        """
        # todo: take out this function and make sure it does not get called anywhere
        raise DeprecationWarning(
            "Drone.get_trajectory is deprecated: should "
            "get flight and call flight.get_trajectory instead"
        )

    def measure(self, flight: "Flight", state: State) -> np.ndarray:
        """! Method to take a measurement

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        return self.detector.measure(flight, state)

    def d_position_d_control(self, flight: "Flight"):
        """
        computes the derivative of the flightpath with respect to the control parameters in alpha.
        This class is problem specific and needs to be written by the user.

        @param alpha:
        @param flightpath:
        @param grid_t:
        @return:
        """
        print(
            "In drone.d_position_d_control: Should call flight.d_position_d_control instead"
        )
        return flight.d_position_d_control

    def d_measurement_d_control(self, flight: "Flight", state: "State") -> np.ndarray:
        """
        Derivative of the measurement function for a given flightpath in control
        direction alpha

        This derivative is computed using the chain rule such that:
        d measurement/d alpha = (d measurement/d position) @ (d position/d alpha)

        @param flight: `Flight` to measure along
        @param state: `State` to measure
        @return: np.ndarray of shape (grid_t.shape[0], self.n_parameters)
        """
        # Derivative of the measurement with respect to the position
        d_meas_d_pos = self.d_measurement_d_position(flight=flight, state=state)
        # Shape $<n_timesteps> \times <n_spatial * n_timesteps>$

        # Derivative of the position with respect to the control
        d_pos_d_cont = flight.d_position_d_control
        # Shape $<n_spatial * n_timesteps> \times <n_controls>$

        # Apply chain rule
        d_meas_d_control = d_meas_d_pos @ d_pos_d_cont

        return d_meas_d_control

    def d_measurement_d_position(self, flight: "Flight", state: "State") -> np.ndarray:
        r"""
        Derivative of the measurement function for a given flight in direction
        of the flight's positions flightpath.

        Passes this functionality to the attached `Detector` class.

        For measurements of the form
            $$
            d(t; p) = \int_{\Omega} \Phi(x, p(t)) u(x,t) dx
            $$
        this function returns
            $$
            \frac{\partial d(t;, p)}{\partial p} = \int_{\Omega} D_y \Phi(x, y=p(t)) u(x, t) dx.
            $$

        @param flight: the flight parameterization of the drone. Contains, in
            particular, the flightpath `flightpath`, the flight controls
            `alpha`, and the time discretization `grid_t`, Flight object
        @param state  The state which the drone shall measure, State object
        @return: np.ndarray of shape (grid_t.shape[0], <spatial dimension>)
        """
        return self.detector.d_measurement_d_position(flight, state)

    def plan_flight(
        self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None
    ) -> "Flight":
        """
        Create a Flight object using the attached `Navigation` system for a
        given control parameter alpha

        The parameters alpha and the time grid grid_t define the flightpath. If
        the time grid is not provided, the default for the attached `Navigation`
        class is used instead.

        @param alpha  flight control parameters, e.g., velocity, angular velocity
        @param grid_t  time grid of the flight path
        @return  the flight object specified by the navigation system, the
            control parameters, and time grid
        """
        return Flight(alpha=alpha, navigation=self.navigation, grid_t=grid_t)
