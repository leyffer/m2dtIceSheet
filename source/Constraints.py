from Navigation import Navigation
from Flight import Flight

import numpy as np
import scipy.sparse as sparse
from functools import cached_property


class Constraints():
    lower_bounds = None
    upper_bounds = None

    def __init__(self, navigation: Navigation):

        self.navigation = navigation

        self.n_positions = navigation.n_timesteps * navigation.n_spatial
        self.n_timesteps = navigation.n_timesteps
        self.n_controls = navigation.n_controls
        self.n_dofs = self.n_positions + self.n_controls
        self.n_spatial = navigation.n_spatial

        self.n_constraints = self.n_positions
        self.constraints_upper = np.zeros((self.n_constraints,))
        self.constraints_lower = np.zeros((self.n_constraints,))
        self.initialize_constraints()

    def set_bounds(self, given_bounds, bool_lower) -> np.ndarray:
        """Define bounds on all variables, first n_spatial for the positions, then the controls
        """
        if not given_bounds.shape[0] == self.n_spatial + self.n_controls:
            if given_bounds.shape[0] == self.n_dofs:
                if bool_lower:
                    self.lower_bounds = given_bounds
                else:
                    self.upper_bounds = given_bounds
            else:
                raise RuntimeError("Invalid bounds provided")

        new_bounds = np.array([given_bounds[0]] * self.n_timesteps)
        for i in range(1, self.n_spatial):
            new_bounds = np.hstack([new_bounds, np.array([given_bounds[i]] * self.n_timesteps)])

        new_bounds = np.hstack([new_bounds, given_bounds[self.n_spatial:]])
        if bool_lower:
            self.lower_bounds = new_bounds
        else:
            self.upper_bounds = new_bounds

    def circle_obstacle(
            self, x: np.ndarray, y: np.ndarray, cx: float, cy: float, rx: float, ry: float
    ) -> np.ndarray:
        """Circular (ellipse) obstacle

        L2 norm:
        ((x - xc)/rx)**2 + ((y - yc)/ry)**2 >= 1

        @param x: np.ndarray. Trajectory x-positions
        @param y: np.ndarray. Trajectory y-positions
        @param cx: float. Circle center x
        @param cy: float. Circle center y
        @param rx: float. Circle (ellipse) radius x
        @param ry: float. Circle (ellipse) radius y
        @return np.ndarray. Distance outside of circle (should be non-negative)
        """
        return (
                ((x - cx) / (rx + self.obstacle_buffer)) ** 2
                + ((y - cy) / (ry + self.obstacle_buffer)) ** 2
                - 1
        )

    def diamond_obstacle(
            self, x: np.ndarray, y: np.ndarray, cx: float, cy: float, rx: float, ry: float
    ) -> np.ndarray:
        """Diamond obstacle

        L1 norm:
        |(x - xc)/rx| + |(y - yc)/ry| >= 1

        @param x: np.ndarray. Trajectory x-positions
        @param y: np.ndarray. Trajectory y-positions
        @param cx: float. Diamond center x
        @param cy: float. Diamond center y
        @param rx: float. Diamond radius x
        @param ry: float. Diamond radius y
        @return np.ndarray. Distance outside of diamond (should be
            non-negative)
        """
        return (
                np.abs((x - cx) / (rx + self.obstacle_buffer))
                + np.abs((y - cy) / (ry + self.obstacle_buffer))
                - 1
        )

    def square_obstacle(
            self, x: np.ndarray, y: np.ndarray, cx: float, cy: float, rx: float, ry: float
    ) -> np.ndarray:
        """Square (rectangle) obstacle

        Rotated 1-norm
        |(x - xc)/rx + (y - yc)/ry|/2 + |(x - xc)/rx - (y - yc)/ry|/2 >= 1
        or
        Infinity norm
        max(|x - xc|/rx, |y - yc|/ry) >= 1

        @param x: np.ndarray. Trajectory x-positions
        @param y: np.ndarray. Trajectory y-positions
        @param cx: float. Square center x
        @param cy: float. Square center y
        @param rx: float. Square (rectangle) radius x
        @param ry: float. Square (rectangle) radius y
        @return np.ndarray. Distance outside of square (should be non-negative)
        """
        return (
                np.abs(((x - cx) / rx + (y - cy) / (ry + self.obstacle_buffer)) / 2)
                + np.abs(((x - cx) / rx - (y - cy) / (ry + self.obstacle_buffer)) / 2)
                - 1
        )

    def initialize_constraints(self):
        return

    def evaluate_constraints(self, flightpath_1d, alpha):
        flightpath_alpha, __, __ = self.navigation.get_trajectory(alpha=alpha)
        flightpath_alpha = np.hstack([flightpath_alpha[:, i] for i in range(self.n_spatial)])

        return flightpath_1d - flightpath_alpha

    def evaluate_jacobian(self, flightpath_1d, alpha):

        jacobian = sparse.eye(self.n_positions)
        flight = Flight(navigation=self.navigation, alpha=alpha)
        deriv = self.navigation.d_position_d_control(flight=flight)
        deriv = sparse.coo_matrix(deriv)

        jacobian = sparse.hstack([jacobian, -deriv])
        jacobian = sparse.coo_matrix(jacobian)

        return jacobian.data

    @cached_property
    def memorized_jac_structure(self) -> tuple[np.ndarray, np.ndarray]:

        jacobian = sparse.eye(self.n_positions)
        flight = Flight(navigation=self.navigation, alpha=np.ones((self.n_controls,)))
        deriv = self.navigation.d_position_d_control(flight=flight)
        deriv = sparse.coo_matrix(deriv)

        jacobian = sparse.hstack([jacobian, -deriv])
        jacobian = sparse.coo_matrix(jacobian)

        return (jacobian.row, jacobian.col)
