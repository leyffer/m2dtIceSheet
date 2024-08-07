"""Detector class

Component of the `Drone` class that performs measurements of a provided `State`
"""
import numpy as np

# from .Drone import Drone
from .Flight import Flight
from .State import State


class Detector:
    r"""
    Takes measurements of a `State` at specified positions and times

    This is a component to a `Drone` class (which consists of a `Detector` for
    measuring and a `Navigator` for determining where to measure).

    The detector describes how to take a measurement of a state $u$ for a given
    flight $p$. In general words, we model measurements to be of the form
        $$
        d(t; p) = \int_{\Omega} \Phi(x, p(t)) u(x,t) dx
        $$
    where
        - $u$ is the state (may be time dependent, but does not have to be)
        - $p : [0, T] \rightarrow \Omega$ is the flight path
        - $\Phi : \Omega \times \Omega \rightarrow \mathbb{R}$ is the
          measurement integration kernel function.

    The detector class describes $\Phi$:
        - what it is (e.g., dirac delta function, Gaussian distribution, etc.)
        - how to compute the measurement (e.g., how to compute the integral)
        - how to take a derivative w.r.t. $p$ (e.g., automatic differentiation,
          closed form derivative, etc.)
        - how it interacts with the state (e.g., FE library, interpolation,
          etc.)

    Since these questions depend on the implementation of the FOM and the
    specific choice of measurement integration kernel function, the user has to
    fill out this class by themselves (as a child class please, don't change
    this file). The descriptions below tell the user which functions have to be
    implemented for compatibility with the rest of the code: Search for
    "NotImplementedError"
    """

    drone = None
    # The detector will always be attached to a Drone object.

    def __init__(self, *args, **kwargs):
        """Initialization of the Detector class

        When writing the child class specific for the application, remember to
        call super().__init__

        Options to be passed in **kwargs:

        bool_allow_multiple_attachments:
        Whenever a drone equips a detector, it will tell the detector that it
        was just equipped through a call to detector.attach_to_drone. By
        default, we allow a single detector to be attached to multiple drones.
        However, if the user expects detector parameters to change, they might
        want to set
        ```
        bool_allow_multiple_attachments = False
        ```
        to ensure that any instance of Detector can only be equipped by a single
        drone. This avoids copy-issues and saves time on debugging. It will,
        however, make code testing harder within notebooks when running blocks
        out of order, which is why, per default, we are enabling multiple
        attachments.
        """
        self.bool_allow_multiple_attachments = kwargs.get(
            "bool_allow_multiple_attachments", True
        )

    def attach_to_drone(self, drone: "Drone"):
        r"""
        The drone tells the detector that it was just equipped. The detector has
        now the ability to communicate to the drone and from there interact with
        its environment. However, this ability should be used with care -
        according to the measurement formula
        $$
        d(t; p) = \int_{\Omega} \Phi(x, p(t)) u(x,t) dx
        $$
        we don't expect the detector to actually need to communicate to the
        drone or beyond.
        """
        if self.drone is not None and not self.bool_allow_multiple_attachments:
            raise RuntimeWarning(
                "Detector system was attached to a new drone. Was this intentional? If attaching "
                "the navigation system to several drones, make sure they will not accidentally "
                "change each other (make an appropriate copy or make sure they have no changeable "
                "parameters)"
            )
        self.drone = drone

    def measure(self, flight: "Flight", state: "State") -> np.ndarray:
        """! Method to take a measurement

        Given a flight with a flightpath and a state, measure the state along
        the flightpath. This returns a vector of measured data, one measurement
        for each time step in the flightpath.

        @param flight  the flight parameterization of the drone. Contains, in
            particular, the flightpath `flightpath`, the flight controls
            `alpha`, and the time discretization `grid_t`, Flight object
        @param state  The state which the drone shall measure, State object
        @returns  a vector of measurements corresponding to the flightpath
        """
        flightpath = flight.flightpath
        grid_t = flight.grid_t

        # initialization
        n_steps = flightpath.shape[0]
        data = np.NaN * np.ones((n_steps,))

        # Loop over the flightpath and measure the state at the positions from
        # the flightpath
        for k in range(n_steps):
            try:
                data[k] = self.measure_at_position(
                    pos=flightpath[k, :], t=grid_t[k], state=state
                )
            # We expect a RuntimeError if the position is outside of the
            # computational domain
            except RuntimeError:
                # If we are outside of the computational domain, the data
                # returned is zero. If another behavior is needed, please
                # overwrite this function in the child class
                data[k] = 0.0

        return data

    def measure_at_position(
        self, pos: np.ndarray, t: float, state: "State", **kwargs
    ) -> float:
        """Measure the state at a particular position

        @param pos  position to measure at
        @param t  time to measure at
        @param state  `State` to measure
        @returns  the measured value at the position
        """
        raise NotImplementedError("Needs to be implemented in subclass")

    def d_measurement_d_position(self, flight: "Flight", state: "State") -> np.ndarray:
        r"""
        Derivative of the measurement function for a given flight in direction
        of the flight's positions flightpath. For measurements of the form
        $$
        d(t; p) = \int_{\Omega} \Phi(x, p(t)) u(x,t) dx
        $$
        this function returns
        $$
        \frac{\partial d(t;, p)}{\partial p} = \int_{\Omega} D_y \Phi(x, y=p(t)) u(x, t) dx.
        $$

        Since the position is determined by <spatial dimension>*<number of time
        steps> parameters, and a measurement has <number of time steps> entries,
        the return of this function has to have shape
        $$ <number of time steps> \times <spatial dimension>*<number of time steps> $$

        The columns should be ordered such that the first <number of time steps>
        columns are for the first spatial dimension (x direction), the next
        <number of time steps> columns for the second (y-direction), etc.

        @param flight: the flight parameterization of the drone. Contains, in
            particular, the flightpath `flightpath`, the flight controls
            `alpha`, and the time discretization `grid_t`, Flight object
        @param state  The state which the drone shall measure, State object
        @return: np.ndarray of shape (grid_t.shape[0], <spatial dimension>)
        """
        # Get the flightpath and time grid from the `Flight` object
        flightpath, grid_t = flight.flightpath, flight.grid_t

        # The number of spatial points measured
        n_spatial = flightpath.shape[1]

        # Initialization of the derivative
        D_data_d_position = np.zeros((grid_t.shape[0], n_spatial))  # (time, (dx,dy))

        for time_step in range(grid_t.shape[0]):
            # Evaluate the derivative at the considered position, e.g., [d/dx, d/dy]
            D_data_d_position[time_step, :] = self.derivative_at_position(
                pos=flightpath[time_step, :], t=grid_t[time_step], state=state
            )

        # Stack derivatives next to each other horizontally
        D_data_d_position = np.hstack(
            [
                np.diag(D_data_d_position[:, spatial_dimension])
                for spatial_dimension in range(n_spatial)
            ]
        )
        return D_data_d_position

    def derivative_at_position(
        self, pos: np.ndarray, t: float, state: "State", **kwargs
    ):
        """
        Compute the state's measurement derivative (w.r.t. the position) at time
        `t` and position `pos`

        @param pos  position to measure at
        @param t  time to measure at
        @param state  `State` to measure
        @returns  the measured derivative at the position
        """
        raise NotImplementedError(
            "derivative_at_position needs to be implemented in subclass"
        )
