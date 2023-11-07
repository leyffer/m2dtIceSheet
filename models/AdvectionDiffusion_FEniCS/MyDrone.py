import sys
sys.path.insert(0, "../source/")

import numpy as np

from Drone import Drone

class MyDrone(Drone):

    center = np.array([0.75/2, 0.55/2])

    def __init__(self, fom, eval_mode, grid_t = None, **kwargs):
        """! Initializer for the drone class
        @param fom  Full-order-model (FOM) object. The drone takes
        measurements from this
        @param eval_mode  Evaluation mode of the drone's measurements:

            - `"gaussian, truncated"`: The drone takes a measurement that is dispersed
              over a 2D truncted gaussian
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
        super().__init__(fom)

        self.eval_mode = eval_mode
        self.grid_t = grid_t if grid_t is not None else np.arange(0, 4 + 1e-2, 1e-2)

        # todo: get parameterization for other eval modes, in partiuclar give them a common name, not individual ones:
        # self.sigma_gaussian = kwargs.get("sigma_gaussian", 0.1)
        # self.radius_uniform = kwargs.get("radius_uniform", 0.1)

    def get_trajectory(self, alpha: np.ndarray, grid_t: np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        """! Get the trajectory of the drone given the flight parameters alpha
        @param alpha The specified flight parameters
        @param grid_t the time grid on which the drone position shall be computed
        @return  Tuple of (position over flight path, corresponding time for each position)
        """
        # interpret the flying parameter
        center = self.center
        radius = alpha[0]
        speed = alpha[1]

        # default to your own time grid if none is provided
        if grid_t is None:
            grid_t = self.grid_t

        # put in the closed-form description of the circular flight path
        round_trip_time = 2 * np.pi * radius / speed
        angles = (grid_t * 2 * np.pi) / round_trip_time
        pos = radius * np.vstack([np.cos(angles), np.sin(angles)]).T
        pos = pos + center

        return pos, grid_t

    def d_position_d_control(self, alpha, flightpath, grid_t):
        """
        computes the derivative of the flightpath with respect to the control parameters in alpha.
        This class is problem specific and needs to be written by the user.

        @param alpha:
        @param flightpath:
        @param grid_t:
        @return:
        """
        # for the Drone class

        radius = alpha[0]
        speed = alpha[1]

        round_trip_time = 2 * np.pi * radius / speed
        angles = (grid_t * 2 * np.pi) / round_trip_time

        d_speed = (np.vstack([-np.sin(angles), np.cos(angles)]) * grid_t)

        d_radius = np.vstack([np.cos(angles), np.sin(angles)])
        d_radius = d_radius - (np.vstack([-np.sin(angles), np.cos(angles)]) * (grid_t * speed)) / radius

        return np.array([d_radius, d_speed])

    def measure(self, flightpath, grid_t, state) -> np.ndarray:
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

    def measure_pointwise(self, flightpath, grid_t, state):
        """! Get measurements along the flight path at the drone location

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        if state.bool_is_transient:
            raise NotImplementedError("In MyDrone.measure_pointwise: still need to bring over code for transient measurements")
            # old code:
            # return [state[k].at(*flightpath[k, :]) for k in range(flightpath.shape[0])]

        return np.array([state.state(flightpath[k, :]) for k in range(flightpath.shape[0])])

    def d_measurement_d_control(self, alpha, flightpath, grid_t, state):
        """
        derivative of the measurement function for a given flightpath in control direction alpha

        @param alpha:
        @param flightpath:
        @param grid_t:
        @param state:
        @return: np.ndarray of shape (grid_t.shape[0], self.n_parameterss)
        """

        # parts of the chain rule (only compute once)
        grad_p = self.d_position_d_control(alpha, flightpath, grid_t)  # derivative of position
        # todo: optimize this computation such that we don't repeat it as often
        Du = state.get_derivative()  # spatial derivative of the state

        # initialization
        D_data_d_alpha = np.empty((grid_t.shape[0], alpha.shape[0]))

        if self.eval_mode == "point-eval":

            for i in range(grid_t.shape[0]):
                # the FEniCS evaluation of the Du at a position unfortunately doesn't work with multiple positions
                # that's why we can't get rid of this loop

                # apply chain rule
                D_data_d_alpha[i, :] = Du(flightpath[i, :]) @ grad_p[:, :, i].T

                # todo: make compatible with transient setting

            return D_data_d_alpha

        # todo: put in other measurement types

        raise NotImplementedError("still need to do the maths for other measurement types")







