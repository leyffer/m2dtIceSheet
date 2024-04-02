import numpy as np


class Detector:
    """
    Takes measurements of a state at locations

    The detector describes how to take a measurement of a state $u$ for a given flight $p$. In general words, we model
    measurements to be of the form
    $$
    d(t; p) = \int_{\Omega} \Phi(x, p(t)) u(x,t) dx
    $$
    where
    - $u$ is the state (may be time dependent, but does not have to be)
    - $p : [0, T] \rightarrow \Omega$ is the flight path
    - $\Phi : \Omega \times \Omega \rightarrow \mathbb{R}$ is the measurement integration kernel function.

    The detector class describes $\Phi$:
    - what it is (e.g., dirac delta function, Gaussian distribution, etc.)
    - how to compute the measurement (e.g., how to compute the integral)
    - how to take a derivative w.r.t. $p$ (e.g., automatic differentiation, closed form derivative, etc.)
    - how it interacts with the state (e.g., FE library, interpolation, etc.)

    Since these questions depend on the implementation of the FOM and the specific choice of measurement integration
    kernel function, the user has to fill out this class by themselves (as a child class please, don't change this
    file). The descriptions below tell the user which functions have to be implemented for compatibility with the rest
    of the code: Search for "NotImplementedError"
    """
    drone = None
    # The detector will always be attached to a Drone object.

    def __init__(self, *args, **kwargs):
        """Initialization of the Detector class

        When writing the child class specific for the application, remember to call super().__init__

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
        self.bool_allow_multiple_attachments = kwargs.get("bool_allow_multiple_attachments", True)

    def attach_to_drone(self, drone : "Drone"):
        """
        The drone tells the detector that it was just equipped. The detector has now the ability to communicate to the
        drone and from there interact with its environment. However, this ability should be used with care - according
        to the measurement formula
        $$
        d(t; p) = \int_{\Omega} \Phi(x, p(t)) u(x,t) dx
        $$
        we don't expect the detector to actually need to communicate to the drone or beyond.
        """
        if self.drone is not None and not self.bool_allow_multiple_attachments:
            raise RuntimeWarning("Detector system was attached to a new drone. Was this intentional? If attaching "
                                 "the navigation system to several drones, make sure they'll not accidentally change "
                                 "each other (make an appropriate copy or make sure they have no changable "
                                 "parameters)")
        self.drone = drone

    def measure(
        self,
        flight : "Flight",
        state: "State"
    ) -> np.ndarray:
        """! Method to take a measurement

        @param flight  the flight parameterization of the drone. Contains, in particular, the flightpath `flightpath`,
        the flight controls `alpha`, and the time discretization `grid_t`, Flight object
        @param state  The state which the drone shall measure, State object
        """
        raise NotImplementedError("Drone.measure: Needs to be implemented in subclass")

    def d_measurement_d_position(self, flight, state):
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
        raise NotImplementedError(
            "Drone.d_measurement_d_control: Needs to be implemented in subclass"
        )
