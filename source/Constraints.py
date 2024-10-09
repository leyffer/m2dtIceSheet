from Navigation import Navigation
from Flight import Flight

import numpy as np
import scipy.sparse as sparse
from functools import cached_property


class Constraints():
    lower_bounds = None
    upper_bounds = None

    n_constraints = 0
    constraints_upper = None
    constraints_lower = None

    obstacle_buffer = 1e-2

    def __init__(self, navigation: Navigation):

        self.navigation = navigation

        self.n_positions = navigation.n_timesteps * navigation.n_spatial
        self.n_timesteps = navigation.n_timesteps
        self.n_controls = navigation.n_controls
        self.n_dofs = self.n_positions + self.n_controls + navigation.n_timesteps
        # todo: formalize, this is specific for DAE right now
        self.n_spatial = navigation.n_spatial

        self.initialize_constraints()

        self.n_obstacles = 0
        self.obstacles = np.zeros((0, 3), dtype=object)

        self.jacobian_sparsity_structure = self.navigation.positional_constraint_sparsity_pattern

    def set_bounds(self, given_bounds, bool_lower) -> np.ndarray:
        """Define bounds on all variables, first n_spatial for the positions, then the controls
        """
        if given_bounds.shape[0] == self.n_dofs:
            if bool_lower:
                self.lower_bounds = given_bounds
            else:
                self.upper_bounds = given_bounds
        else:
            raise RuntimeError("Invalid bounds provided")

    def initialize_constraints(self):
        self.n_constraints = self.navigation.n_constraints
        self.constraints_upper = np.zeros((self.n_constraints,))
        self.constraints_lower = np.zeros((self.n_constraints,))

    def evaluate_constraints(self, flightpath_1d, alpha):
        positional = self.navigation.evaluate_positional_constraints(flightpath_1d, alpha)
        obstacles = self.evaluate_obstacles(flightpath_1d)
        return np.hstack([positional, obstacles])

    def evaluate_jacobian(self, flightpath_1d, alpha):
        flight = Flight(navigation=self.navigation, alpha=alpha)
        positional = self.navigation.d_positional_constraint(flight).data
        obstacles = self.d_evaluate_obstacles(flightpath_1d)

        return np.hstack([positional, obstacles])

    def add_obstacle(self, type, specs):
        new_obstacle = np.zeros((1, 3), dtype=object)
        new_obstacle[0, 0] = type
        new_obstacle[0, 1] = specs

        rows = [*range(self.n_constraints, self.n_constraints + self.n_timesteps)]
        rows = rows + rows
        cols = [*range(2 * self.n_timesteps)]
        new_obstacle[0, 2] = (np.array(rows), np.array(cols))

        self.obstacles = np.vstack([self.obstacles, new_obstacle])
        self.n_obstacles += 1
        self.n_constraints = self.n_constraints + self.n_timesteps
        self.constraints_lower = np.hstack([self.constraints_lower, np.zeros((self.n_timesteps,))])
        self.constraints_upper = np.hstack([self.constraints_upper, np.infty * np.ones((self.n_timesteps,))])

        a = np.hstack([self.jacobian_sparsity_structure[0], np.array(rows)])
        b = np.hstack([self.jacobian_sparsity_structure[1], np.array(cols)])
        self.jacobian_sparsity_structure = (a, b)

    def evaluate_obstacles(self, flightpath_1d):
        """
        evaluates the constraint function for each obstacle
        """
        # split up input data to get the positional information
        pos_x = flightpath_1d[:self.n_timesteps]
        pos_y = flightpath_1d[self.n_timesteps:2 * self.n_timesteps]

        # initialize return data
        data_obstacles = np.zeros(self.n_obstacles, dtype=object)

        # loop over all obstacles
        for i in range(self.n_obstacles):

            # identify which obstacle function we need
            type = self.obstacles[i, 0]
            if type == "circle":
                eval_fct = self.circle_obstacle
            elif type == "diamond":
                eval_fct = self.diamond_obstacle
            elif type == "square":
                eval_fct = self.square_obstacle
            else:
                raise RuntimeError("Invalid obstacle type: {}".format(type))

            # evaluate obstacle function
            data_obstacles[i] = eval_fct(x=pos_x,
                                         y=pos_y,
                                         cx=self.obstacles[i, 1][0],
                                         cy=self.obstacles[i, 1][1],
                                         rx=self.obstacles[i, 1][2],
                                         ry=self.obstacles[i, 1][3])
        return np.hstack(data_obstacles)

    def d_evaluate_obstacles(self, flightpath_1d):
        """
        computes all non-zero entries in teh derivatives of each obstacle
        """
        # split up input data to get the positional information
        pos_x = flightpath_1d[:self.n_timesteps]
        pos_y = flightpath_1d[self.n_timesteps:2 * self.n_timesteps]

        # initialize return data
        data_obstacles = np.zeros(self.n_obstacles, dtype=object)
        for i in range(self.n_obstacles):

            # get characterizing information for this obstacle
            type = self.obstacles[i, 0]
            cx = self.obstacles[i, 1][0]
            cy = self.obstacles[i, 1][1]
            rx = self.obstacles[i, 1][2]
            ry = self.obstacles[i, 1][3]

            # compute derivative data according to obstacle typle
            if type == "circle":
                a = 2 * (pos_x - cx) / (rx ** 2)
                b = 2 * (pos_y - cy) / (ry ** 2)
            elif type == "diamond":
                a = np.sign(pos_x - cx) / rx
                b = np.sign(pos_y - cy) / ry
            elif type == "square":
                a = np.sign((pos_x - cx) / rx + (pos_y - cy) / ry) / rx / 2 + np.sign(
                    (pos_x - cx) / rx - (pos_y - cy) / ry) / rx / 2
                b = np.sign((pos_x - cx) / rx + (pos_y - cy) / ry) / ry / 2 - np.sign(
                    (pos_x - cx) / rx - (pos_y - cy) / ry) / ry / 2
            else:
                raise RuntimeError("Invalid obstacle type: {}".format(type))
            data_obstacles[i] = np.hstack([a, b])

        # stack derivative data according to sparsity structure
        return np.hstack(data_obstacles)

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
