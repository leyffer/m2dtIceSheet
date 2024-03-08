import sys

sys.path.insert(0, "../source/")

from typing import Optional, Tuple

import numpy as np
from Drone import Drone
from myState import State
from Path import CirclePath

FlightPath = np.dtype([("position", "<f8", 2), ("time", "<f8")])


class MyDrone(Drone):
    center = np.array([0.75 / 2, 0.55 / 2])

    def __init__(
        self,
        eval_mode: str = "point-eval",
        grid_t: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """! Initializer for the drone class
        @param fom  Full-order-model (FOM) object. The drone takes
        measurements from this
        @param eval_mode  Evaluation mode of the drone's measurements:

            - `"gaussian, truncated"`: The drone takes a measurement that is dispersed
              over a 2D truncated gaussian
            - `"gaussian"`: The drone takes a measurement that is dispersed
              over a 2D gaussian
            - `"uniform"`: The drone takes a measurement that is dispersed
              uniformly over a circle
            - `"point-eval"`: The drone takes a measurement at its exact
              location

        @param grid_t the time grid the drone should fly in
        @param **kwargs  Keyword arguments including `sigma_gaussian`
        (Gaussian radius) and `radius_uniform`
        """
        super().__init__(eval_mode=eval_mode, grid_t=grid_t)

        self.grid_t = grid_t if grid_t is not None else np.arange(0, 4 + 1e-2, 1e-2)

        self.path_class = CirclePath

        # TODO: get parameterization for other eval modes, in particular give them a common name, not individual ones:
        # self.sigma_gaussian = kwargs.get("sigma_gaussian", 0.1)
        # self.radius_uniform = kwargs.get("radius_uniform", 0.1)

    def path(self, alpha: np.ndarray) -> CirclePath:
        """Instantiate the path class with alpha"""
        return self.path_class(alpha=alpha, center=self.center)

    def get_trajectory(
        self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """! Get the trajectory of the drone given the flight parameters alpha
        @param alpha  The specified flight parameters
        @param grid_t  the time grid on which the drone position shall be computed
        @return  Position over flight path
        """
        # default time grid if None is provided
        if grid_t is None:
            grid_t = self.grid_t

        return self.path(alpha).position(grid_t), grid_t

    def d_position_d_control(
        self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None
    ):
        """
        Computes the derivative of the flightpath with respect to the control parameters in alpha.
        This class is problem specific and needs to be written by the user.

        @param alpha:
        @param grid_t:
        @return:
        """
        # for the Drone class
        if grid_t is None:
            grid_t = self.grid_t

        d_speed = self.path(alpha).d_position_d_velocity(grid_t).T
        d_radius = self.path(alpha).d_position_d_radius(grid_t).T

        return np.array([d_radius, d_speed])

    def measure(
        self, flightpath: np.ndarray, grid_t: np.ndarray, state: State
    ) -> np.ndarray:
        """! Method to take a measurement

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        mode = self.eval_mode

        if mode == "point-eval":
            return self.measure_pointwise(flightpath, grid_t, state)

        # TODO: copy over functions for other eval-modes
        # if mode == "gaussian, truncated":
        #     return self.measure_gaussian(flightpath, state)
        #
        # if mode == "uniform":
        #     return self.measure_uniform(flightpath, state)

        raise RuntimeError(f"invalid eval_mode={mode} encountered in Drone.measure")

    def measure_pointwise(
        self, flightpath: np.ndarray, grid_t: np.ndarray, state: State
    ) -> np.ndarray:
        """! Get measurements along the flight path at the drone location

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        if isinstance(state, State):
            if state.bool_is_transient:
                raise NotImplementedError(
                    "In MyDrone.measure_pointwise: still need to bring over code for transient measurements"
                )
                # old code:
                # return [state[k].at(*flightpath[k, :]) for k in range(flightpath.shape[0])]
            out = np.zeros((flightpath.shape[0],))
            for k in range(flightpath.shape[0]):
                try:
                    out[k] = state.state(flightpath[k, :])
                except RuntimeError:
                    out[k] = 0
            return out
            # return np.array(
            #     [state.state(flightpath[k, :]) for k in range(flightpath.shape[0])]
            # )
        out = np.zeros((flightpath.shape[0],))
        for k in range(flightpath.shape[0]):
            try:
                out[k] = state(flightpath[k, :])
            except RuntimeError:
                out[k] = 0
        return out
        # return np.array([state(flightpath[k, :]) for k in range(flightpath.shape[0])])

    def d_measurement_d_control(
        self,
        alpha: np.ndarray,
        flightpath: np.ndarray,
        grid_t: np.ndarray,
        state: State,
    ) -> np.ndarray:
        """
        derivative of the measurement function for a given flightpath in control direction alpha

        @param alpha:
        @param flightpath:
        @param grid_t:
        @param state:
        @return np.ndarray of shape (grid_t.shape[0], self.n_parameters)
        """

        # parts of the chain rule (only compute once)
        grad_p = self.d_position_d_control(alpha, grid_t)  # derivative of position
        # TODO: optimize this computation such that we don't repeat it as often
        Du = state.get_derivative()  # spatial derivative of the state

        # initialization
        D_data_d_alpha = np.zeros((grid_t.shape[0], alpha.shape[0]))

        if self.eval_mode == "point-eval":
            for i in range(grid_t.shape[0]):
                # the FEniCS evaluation of the Du at a position unfortunately doesn't work with multiple positions
                # that's why we can't get rid of this loop

                # apply chain rule
                try:
                    D_data_d_alpha[i, :] = Du(flightpath[i, :]) @ grad_p[:, :, i].T
                except RuntimeError:
                    pass

                # TODO: make compatible with transient setting

            return D_data_d_alpha

        # TODO: put in other measurement types

        raise NotImplementedError(
            "still need to do the maths for other measurement types"
        )
