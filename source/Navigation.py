import numpy as np
from typing import Optional
from Flight import Flight

class Navigation:
    """
    Converts control parameters alpha into a flightpath

    The Navigation system describes how the drone flies for any given control parameterization. It creates the flight
    as an instance of the Flight class (plan_flight). The navigation system is more general than a specific flight,
    alpha never gets set. Instead, it offers all the understanding of how a flight is created from the control alpha,
    and how it reacts to changes (d_position_d_control).

    Since these questions depend on the implementation of the FOM and the specific choice of measurement integration
    kernel function, the user has to fill out this class by themselves (as a child class please, don't change this
    file). The descriptions below tell the user which functions have to be implemented for compatibility with the rest
    of the code: Search for "NotImplementedError"
    """
    # properties to be defined in user child class
    n_spatial = None  # spatial dimension, likely n_spatial = 2 or 3
    n_controls = None  # number of control dimensions, can be fixed or dependent on number of time steps

    # properties set at runtime
    drone = None

    def __init__(self, grid_t : np.ndarray, *args, **kwargs):
        """Initialization of the Navigation class

        When writing the child class specific for the application, remember to call super().__init__

        The time discretization grid_t will be used for all flights. We are assuming uniform time stepping. This might
        change in the future.

        @param grid_t : the time discretization for the drone, np.ndarray with len(grid_t.shape)=1

        Options to be passed in **kwargs:

        bool_allow_multiple_attachments:
        Whenever a drone equips a detector, it will tell the detector that it was just
        equipped through a call to detector.attach_to_drone. By default, we allow a single detector to be attached to
        multiple drones. However, if the user expects detector parameters to change, they might want to set
        bool_allow_multiple_attachments = False
        to ensure that any instance of Detector can only be equipped by a single drone. This avoids copy-issues and
        saves time on debugging. It will, however, make code testing harder within notebooks when running blocks out of
        order, which is why, per default, we are enabling multiple attachments.
        """

        # time discretization (this one will be ued for all flights)
        self.grid_t = grid_t
        self.n_timesteps = grid_t.shape[0]

        # check if the user wants to allow or disallow the use of the same navigation system in multiple drones
        self.bool_allow_multiple_attachments = kwargs.get("bool_allow_multiple_attachments", True)

    def attach_to_drone(self, drone):
        """
        The drone tells the navigation system that it was just equipped. The navigation system has now the ability to
        communicate to the drone and from there interact with its environment. However, this ability should be used with
        care, we expect that the navigation for most applications is stand-alone.
        """
        if self.drone is not None and not self.bool_allow_multiple_attachments:
            raise RuntimeWarning("Navigation system was attached to a new drone. Was this intentional? If attaching "
                                 "the navigation system to several drones, make sure they'll not accidentally change "
                                 "each other (make an appropriate copy or make sure they have no changable "
                                 "parameters)")
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
