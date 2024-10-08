"""Class for converting flight controls to positions"""
from typing import Optional
from functools import cached_property

import numpy as np
import scipy.sparse as sparse

# from Drone import Drone
from Flight import Flight


class Navigation:
    """
    Converts control parameters alpha into a flightpath

    The Navigation system describes how the drone flies for any given control
    parameterization. It creates the flight as an instance of the Flight class
    (plan_flight). The navigation system is more general than a specific flight,
    alpha never gets set. Instead, it offers all the understanding of how a
    flight is created from the control alpha, and how it reacts to changes
    (d_position_d_control).

    Since these questions depend on the implementation of the FOM and the
    specific choice of measurement integration kernel function, the user has to
    fill out this class by themselves (as a child class please, don't change
    this file). The descriptions below tell the user which functions have to be
    implemented for compatibility with the rest of the code: Search for
    "NotImplementedError"
    """

    # properties to be defined in user child class
    # spatial dimension, likely n_spatial = 2 or 3
    n_spatial: int = None
    # number of control dimensions, can be fixed or dependent on number of time steps
    n_controls: int = None

    # properties set at runtime
    drone: "Drone" = None

    def __init__(self, grid_t: np.ndarray, *args, **kwargs):
        """Initialization of the Navigation class

        When writing the child class specific for the application, remember to
        call super().__init__

        The time discretization grid_t will be used for all flights. We are
        assuming uniform time stepping. This might change in the future.

        @param grid_t : the time discretization for the drone, np.ndarray with
        len(grid_t.shape)=1

        Options to be passed in **kwargs:

        bool_allow_multiple_attachments:
            Whenever a drone equips a detector, it will tell the detector that
            it was just equipped through a call to detector.attach_to_drone. By
            default, we allow a single detector to be attached to multiple
            drones. However, if the user expects detector parameters to change,
            they might want to set
        bool_allow_multiple_attachments = False
            to ensure that any instance of Detector can only be equipped by a
            single drone. This avoids copy-issues and saves time on debugging.
            It will, however, make code testing harder within notebooks when
            running blocks out of order, which is why, per default, we are
            enabling multiple attachments.
        """

        # time discretization (this one will be used for all flights)
        self.grid_t = grid_t
        self.n_timesteps = grid_t.shape[0]

        # check if the user wants to allow or disallow the use of the same
        # navigation system in multiple drones
        self.bool_allow_multiple_attachments = kwargs.get(
            "bool_allow_multiple_attachments", True
        )

    def attach_to_drone(self, drone: "Drone"):
        """
        The drone tells the navigation system that it was just equipped. The
        navigation system has now the ability to communicate to the drone and
        from there interact with its environment. However, this ability should
        be used with care, we expect that the navigation for most applications
        is stand-alone.

        TODO - avoid circular code (reference a class that references this
        class) by editing or eliminating this

        @param drone  the drone that is attached to the navigator
        """
        if self.drone is not None and not self.bool_allow_multiple_attachments:
            raise RuntimeWarning(
                "Navigation system was attached to a new drone. Was this intentional? If attaching "
                "the navigation system to several drones, make sure they'll not accidentally change "
                "each other (make an appropriate copy or make sure they have no changeable "
                "parameters)"
            )
        self.drone = drone
        self.n_spatial = drone.fom.n_spatial

    def get_trajectory(
        self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """! Get the trajectory of the drone given the flight parameters alpha

        @param alpha The specified flight parameters
        @param grid_t the time grid on which the drone position shall be computed
        @return  Tuple of (position over flight path, corresponding time for each position), grid_t
        """
        raise NotImplementedError(
            "Navigation.get_trajectory: Needs to be implemented in subclass"
        )

    def d_position_d_control(self, flight: Flight) -> np.ndarray:
        """
        computes the derivative of the flightpath with respect to the control
        parameters in alpha. This class is problem specific and needs to be
        written by the user.

        @param flight: Flight object
        @return: gradient vector
        """
        raise NotImplementedError(
            "Navigation.d_position_d_control: Needs to be implemented in subclass"
        )

    def d_position_d_position_and_control(self, flight: Flight) -> np.ndarray:
        """
        computes the derivative of the flightpath with respect to the its positions
        and the control parameters in alpha. If all positions are computed independently
        from each other, this is just going to be the zero matrix stacked next to
        the d_position_d_control output. However, in some cases the position at time $t_k$
        may be computed as an adjustment to the position at time $t_{k-1}$ (for example),
        in which case the derivative of position w.r.t. position is not the identity. These
        special cases need to be implemented in the subclass.

        @param flight: Flight object
        @return: gradient vector
        """
        zero = sparse.coo_matrix(([], ([], [])), shape=(2 * self.n_timesteps, 2 * self.n_timesteps))
        deriv = self.d_position_d_control(flight)
        deriv = sparse.hstack([zero, sparse.coo_matrix(deriv)])

        return deriv

    def d_positional_constraint(self, flight: Flight):
        identity = sparse.eye(self.n_positions)
        extension = sparse.coo_matrix(([], ([], [])), shape=(self.n_positions, self.navigation.n_controls))
        jacobian = sparse.hstack([identity, extension])

        flight = Flight(navigation=self.navigation, alpha=flight.alpha)
        deriv = self.navigation.d_position_d_position_and_control(flight=flight)
        deriv = sparse.coo_matrix(deriv)

        jacobian = jacobian - deriv

        return jacobian

    def evaluate_positional_constraints(self, flightpath_1d, alpha):
        flightpath_alpha, __, __ = self.get_trajectory(alpha=alpha)
        flightpath_alpha = np.hstack([flightpath_alpha[:, i] for i in range(self.n_spatial)])

        return flightpath_1d - flightpath_alpha

    @cached_property
    def positional_constraint_sparsity_pattern(self):
        raise NotImplementedError("Navigation.positional_constraint_sparsity_pattern: still needs to be implemented")

    def regularize_control(self, alpha):
        return 0

    def d_regularize_control(self, alpha):
        return np.zeros((self.n_controls))
