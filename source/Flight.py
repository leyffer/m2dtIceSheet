"""!Flight class

Holds information about an individual flight
"""
from functools import cached_property

import numpy as np


class Flight():
    """
    A flight is the realization of given control parameters alpha. It is specific to it - any alpha should have its
    own associated flight. And no changing alpha after the fact. The flight stores all information that is known about
    it:

    - the control parameterization alpha
    - the time grid
    - the navigation system
    - the flightpath it takes on this time grid (produced by the navigation system)
    - the derivative of its path with respect to its position (not computed at initialization)

    Since the flight outsources the computations of these quantities to the Navigation system, the user probably doesn't
    need to change this class unless in very special cases. If so, please make a child class for compatibility.
    """

    def __init__(self, navigation : "Navigation", alpha, grid_t: np.ndarray = None):
        """ ! Initialization of Flight class

        Creates the flight associated to the control parameters alpha. The navigation system describes how to get
        the flightpath from it.

        @param navigation: Navigation system, for interpreting control parameter alpha
        @param alpha: control parameter, likely np.ndarray but could also be user-specific
        @param grid_t: time grid for the flight
        """
        # Default time grid if not provided
        if grid_t is None:
            grid_t = np.arange(0, 4 + 1e-2, 1e-2)

        self.navigation = navigation
        self.alpha = alpha

        self.flightpath, self.grid_t = navigation.get_trajectory(alpha, grid_t=grid_t)
        
    def get_position(self, t: float | np.ndarray):
        """! Get the position of the drone at a requested time(s)

        The reasons to call this function instead of evaluating flightpath at some index k are that
        - evaluating flight.flightpath[k, :] requires to know for which k we have k delta_t = t
        - the provided t might be between time steps
        - if the position shall be evaluated for a finer or coarser time discretization

        @param t  The time(s) at which to evaluate the position of the drone
        @return  spatial position of the drone at time(s) t
        """
        if isinstance(t, float):  # Only providing a single time
            t = t * np.ones((1,))
        if np.shape(t)[0] == 1:  # Only providing a single time
            pos, __ = self.navigation.get_trajectory(alpha=self.alpha, grid_t=t)
            return pos[0, :]  # Return only a single location
        pos, __ = self.navigation.get_trajectory(alpha=self.alpha, grid_t=t)
        return pos

    @cached_property
    def d_position_d_control(self):
        """
        Computes the derivative of the flight path with respect to the control parameters alpha by calling the
        corresponding function in flight.navigation system. Since this derivative is alpha dependent, but alpha is
        fixed for any flight, we store this derivative for future use within this flight (flight.d_pos_d_con). This is why
        for getting this derivative,
        flight.d_position_d_control
        should be called, and **not**
        flight.navigation.d_position_d_control(flight)
        The latter is more expensive when called frequently and generally more error prone.

        @return derivative of self.flightpath w.r.t. self.alpha, np.ndarray of shape
        <self.grid_t.shape[0]> times <number of control parameters>
        """
        # if it was not yet computed, we get it from the navigation system, and store in self.d_pos_d_con for future use
        return self.navigation.d_position_d_control(flight=self)