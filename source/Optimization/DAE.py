"""
Functions and 
"""

import sys

sys.path.insert(0, "../source/")

from functools import cached_property
from typing import Literal

import cyipopt
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from InverseProblem import InverseProblem
from OEDUtility import OEDUtility

from collections import namedtuple

# from scipy.integrate import RK45


class polygon_obstacle:
    """
    A polygon from vertices listed clockwise.

    Constraints for avoiding the polygon are of the form:

    v_i^T x >= c_i

    where v_i are orthogonal vectors from each polygonal edge and c_i are
    intercept values.

    Constraints probably will only work well for convex polygons.
    """

    def __init__(self, vertices):
        """
        :param vertices:  polygon vertices in clockwise order
        """
        self.vertices = np.array(vertices)

    def unit_vec(self, vector: np.ndarray):
        """Unit vector

        :param vector: np.ndarray. vector to convert to unit vector
        :return: np.ndarray. unit vector in the direction of vector
        """
        return vector / np.linalg.norm(vector)

    @property
    def side_vectors(self):
        """Unit vectors along perimeter of polygon"""
        return np.array(
            [
                self.unit_vec(self.vertices[i] - self.vertices[i - 1])
                for i in range(len(self.vertices))
            ]
        )

    @property
    def orthogonal_vectors(self):
        """Unit normal vectors for polygon sides"""
        return np.array([np.array([-v[1], v[0]]) for v in self.side_vectors])

    @property
    def constants(self):
        """
        orthogonal . x >= constant
        """
        return np.array(
            [
                np.dot(vertex, v)
                for vertex, v in zip(self.vertices, self.orthogonal_vectors)
            ]
        )


# %%

DAE_vars = namedtuple("DAE_vars", ["x", "y", "theta", "v", "acc", "omega", "omega_acc"])


class Objective:
    """Objective class to work with cyipopt"""

    def __init__(
        self,
        grid_t: np.ndarray,
        oed_utility: OEDUtility,
        inversion: InverseProblem,
        obstacle_shape: Literal["circle", "diamond", "square"] = "square",
    ):
        self.utility = oed_utility
        self.inversion = inversion

        # Initial position
        self.x0 = 0.7
        self.y0 = 0.3
        # Initial heading
        self.theta0 = 0.0

        # Final position
        self.x_final = 0.1
        self.y_final = 0.1
        # Final heading
        self.theta_final = np.pi

        # Obstacles
        # Centers
        self.cxs = [(0.5 + 0.25) / 2, (0.75 + 0.6) / 2]
        self.cys = [(0.4 + 0.15) / 2, (0.85 + 0.6) / 2]
        # Radii
        self.rxs = [(0.5 - 0.25) / 2, (0.75 - 0.6) / 2]
        self.rys = [(0.4 - 0.15) / 2, (0.85 - 0.6) / 2]
        # Shape
        self.obstacle_shape = obstacle_shape

        # Bound constraints
        # x
        self.x_lower = 0
        self.x_upper = 1
        # y
        self.y_lower = 0
        self.y_upper = 1
        # theta (unbounded)
        self.theta_lower = -np.inf
        self.theta_upper = np.inf
        # v
        self.v_lower = 0.1
        self.v_upper = 3.0
        # acc
        self.acc_lower = -10
        self.acc_upper = 10
        # omega
        self.omega_lower = -20 * np.pi
        self.omega_upper = 20 * np.pi
        # omega_acc
        self.omega_acc_lower = -200
        self.omega_acc_upper = 200

        self.T = grid_t[-1] - grid_t[0]  # Total time

        self.NK = grid_t.shape[0]  # Number of control variables
        self.N_x = self.NK  # Number of x values
        self.N_y = self.NK  # Number of y values
        self.N_theta = self.NK  # Number of theta values
        self.N_v = self.NK  # Number of v values
        self.N_acc = self.NK  # Number of acc values
        self.N_omega = self.NK  # Number of omega values
        self.N_omega_acc = self.NK  # Number of omega_acc values

        self.grid_t = grid_t  # Time grid
        self.h = self.T / self.NK  # Time grid spacing
        self.reg = 0.000001  # Regularization strength

        self.x_shift = 0
        self.y_shift = self.x_shift + self.N_x
        self.theta_shift = self.y_shift + self.N_y
        self.v_shift = self.theta_shift + self.N_theta
        self.acc_shift = self.v_shift + self.N_v
        self.omega_shift = self.acc_shift + self.N_acc
        self.omega_acc_shift = self.omega_shift + self.N_omega

        self.var_lengths = DAE_vars(
            self.N_x,
            self.N_y,
            self.N_theta,
            self.N_v,
            self.N_acc,
            self.N_omega,
            self.N_omega_acc,
        )
        self.cum_var_lengths = DAE_vars(*tuple(np.cumsum(self.var_lengths)))
        self.indices = DAE_vars(
            *tuple(
                np.arange(cum_length - length, cum_length, 1, dtype=int)
                for cum_length, length in zip(self.cum_var_lengths, self.var_lengths)
            )
        )

        self.n = self.omega_acc_shift + self.N_omega_acc  # number of variables

        self.num_equality_constraints = None
        self.num_inequality_constraints = None
        # Do an initialization of the constraints to get constraint numbers
        self.constraints(np.zeros((self.n,)))

    def OED_objective(self, combined_vars: np.ndarray) -> float:
        """OED objective function"""
        (x, y, _theta, _v, _acc, _omega, _omega_acc) = self.var_splitter(combined_vars)
        return self.utility.eval_utility_A(
            self.inversion.compute_posterior(
                alpha=np.concatenate((x, y), axis=0), grid_t=self.grid_t
            )
        )

    def OED_gradient(self, combined_vars: np.ndarray) -> np.ndarray:
        """Gradient of OED objective function"""
        (x, y, _theta, _v, _acc, _omega, _omega_acc) = self.var_splitter(combined_vars)
        out = np.zeros((self.n,))
        out[self.x_shift : self.theta_shift] = self.utility.d_utilA_d_position(
            self.inversion.compute_posterior(
                alpha=np.concatenate((x, y), axis=0), grid_t=self.grid_t
            )
        )
        return out

    @property
    def m(self) -> int:
        """Number of constraints

        :return: int. Number of constraints
        """
        return self.num_equality_constraints + self.num_inequality_constraints

    def var_joiner(
        self,
        *args,
    ) -> np.ndarray:
        """Join variables into a single vector

        :param args: tuple[np.ndarray]. Variables to join
        :return: np.ndarray. Concatenated vector containing all of the DAE
            variables
        """
        return np.concatenate(args, axis=0)

    def var_splitter(self, combined_vars: np.ndarray) -> tuple[np.ndarray]:
        """Split variables from a single vector"""
        # x = combined_vars[self.x_shift : self.x_shift + self.N_x]
        # y = combined_vars[self.y_shift : self.y_shift + self.N_y]
        # theta = combined_vars[self.theta_shift : self.theta_shift + self.N_theta]
        # v = combined_vars[self.v_shift : self.v_shift + self.N_v]
        # acc = combined_vars[self.acc_shift : self.acc_shift + self.N_acc]
        # omega = combined_vars[self.omega_shift : self.omega_shift + self.N_omega]
        # omega_acc = combined_vars[
        #     self.omega_acc_shift : self.omega_acc_shift + self.N_omega_acc
        # ]
        # return (x, y, theta, v, acc, omega, omega_acc)
        return DAE_vars(*tuple(np.split(combined_vars, self.cum_var_lengths[:-1])))

    @property
    def lb(self) -> np.ndarray:
        """Lower bounds on variables

        :return: np.ndarray. Vector of lower bounds on all discretized variables
        """
        x_lb = np.ones((self.N_x,)) * self.x_lower
        y_lb = np.ones((self.N_y,)) * self.y_lower
        theta_lb = np.ones((self.N_theta,)) * self.theta_lower
        v_lb = np.ones((self.N_v,)) * self.v_lower
        acc_lb = np.ones((self.N_acc,)) * self.acc_lower
        omega_lb = np.ones((self.N_omega,)) * self.omega_lower
        omega_acc_lb = np.ones((self.N_omega_acc,)) * self.omega_acc_lower
        return self.var_joiner(
            x_lb, y_lb, theta_lb, v_lb, acc_lb, omega_lb, omega_acc_lb
        )

    @property
    def ub(self) -> np.ndarray:
        """Upper bound on variables

        :return: np.ndarray. Vector of upper bounds on all discretized variables
        """
        x_ub = self.x_upper * np.ones((self.N_x,))
        y_ub = self.y_upper * np.ones((self.N_y,))
        theta_ub = self.theta_upper * np.ones((self.N_theta,))
        v_ub = self.v_upper * np.ones((self.N_v,))
        acc_ub = self.acc_upper * np.ones((self.N_acc,))
        omega_ub = self.omega_upper * np.ones((self.N_omega,))
        omega_acc_ub = self.omega_acc_upper * np.ones((self.N_omega_acc,))
        return self.var_joiner(
            x_ub, y_ub, theta_ub, v_ub, acc_ub, omega_ub, omega_acc_ub
        )

    def regularization_objective(self, combined_vars: np.ndarray) -> float:
        """Regularization objective function

        :param vars: np.ndarray. Combined variables
        :return: np.ndarray. Regularization objective value
        """
        (_x, _y, _theta, _v, acc, _omega, omega_acc) = self.var_splitter(combined_vars)
        # return (np.sum(v**2) + np.sum(omega**2))
        return np.sum(acc**2) + np.sum(omega_acc**2)

    def regularization_gradient(self, combined_vars: np.ndarray):
        """Regularization objective function gradient

        :param vars: np.ndarray. Combined variables
        :return: np.ndarray. Gradient w.r.t. the regularization objective
        """
        (x, y, theta, v, acc, omega, omega_acc) = self.var_splitter(combined_vars)

        x = np.zeros(x.shape)
        y = np.zeros(y.shape)
        theta = np.zeros(theta.shape)
        v = np.zeros(v.shape)
        # v = 2*v
        # acc = np.zeros(acc.shape)
        acc = 2 * acc
        omega = np.zeros(omega.shape)
        # omega = 2*omega
        # omega_acc = np.zeros(omega_acc.shape)
        omega_acc = 2 * omega_acc

        return self.var_joiner(x, y, theta, v, acc, omega, omega_acc)

    def objective(self, combined_vars: np.ndarray) -> float:
        """Objective function

        Combination of OED objective and regularization objective

        :param vars: np.ndarray. Combined variables
        :return: np.ndarray. Objective value
        """
        return self.OED_objective(
            combined_vars
        ) + self.reg * self.regularization_objective(combined_vars)

    def gradient(self, combined_vars: np.ndarray) -> np.ndarray:
        """Objective function gradient

        Combination of OED objective and regularization gradient

        :param vars: np.ndarray. Combined variables
        :return: np.ndarray. Gradient of objective w.r.t. variables
        """
        return self.OED_gradient(
            combined_vars
        ) + self.reg * self.regularization_gradient(combined_vars)

    def circle_obstacle(
        self, x: np.ndarray, y: np.ndarray, cx: float, cy: float, rx: float, ry: float
    ) -> np.ndarray:
        """Circular (ellipse) obstacle

        ((x - xc)/rx)**2 + ((y - yc)/ry)**2 >= 1

        :param x: np.ndarray. Trajectory x-positions
        :param y: np.ndarray. Trajectory y-positions
        :param cx: float. Circle center x
        :param cy: float. Circle center y
        :param rx: float. Circle (ellipse) radius x
        :param ry: float. Circle (ellipse) radius y
        :return: np.ndarray. Distance outside of circle (should be non-negative)
        """
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 - 1

    def diamond_obstacle(
        self, x: np.ndarray, y: np.ndarray, cx: float, cy: float, rx: float, ry: float
    ) -> np.ndarray:
        """Diamond obstacle

        |(x - xc)/rx| + |(y - yc)/ry| >= 1

        :param x: np.ndarray. Trajectory x-positions
        :param y: np.ndarray. Trajectory y-positions
        :param cx: float. Diamond center x
        :param cy: float. Diamond center y
        :param rx: float. Diamond radius x
        :param ry: float. Diamond radius y
        :return: np.ndarray. Distance outside of diamond (should be
            non-negative)
        """
        return np.abs((x - cx) / rx) + np.abs((y - cy) / ry) - 1

    def square_obstacle(
        self, x: np.ndarray, y: np.ndarray, cx: float, cy: float, rx: float, ry: float
    ) -> np.ndarray:
        """Square (rectangle) obstacle

        Rotated 1-norm
        |(x - xc)/rx + (y - yc)/ry|/2 + |(x - xc)/rx - (y - yc)/ry|/2 >= 1
        or
        Infinity norm
        max(|x - xc|/rx, |y - yc|/ry) >= 1

        :param x: np.ndarray. Trajectory x-positions
        :param y: np.ndarray. Trajectory y-positions
        :param cx: float. Square center x
        :param cy: float. Square center y
        :param rx: float. Square (rectangle) radius x
        :param ry: float. Square (rectangle) radius y
        :return: np.ndarray. Distance outside of square (should be non-negative)
        """
        return (
            np.abs(((x - cx) / rx + (y - cy) / ry) / 2)
            + np.abs(((x - cx) / rx - (y - cy) / ry) / 2)
            - 1
        )

    # @jax.jit
    def constraints(self, combined_vars):
        """Values to constrain between cl and cu"""
        # Equality constraints
        (x, y, theta, v, acc, omega, omega_acc) = self.var_splitter(combined_vars)
        dae_x = (x[1:] - x[:-1]) - self.h * np.cos(theta[:-1]) * v[:-1]  # == 0
        dae_y = (y[1:] - y[:-1]) - self.h * np.sin(theta[:-1]) * v[:-1]  # == 0
        dae_theta = (theta[1:] - theta[:-1]) - self.h * omega[:-1]  # == 0
        dae_acc = (v[1:] - v[:-1]) - self.h * acc[:-1]  # == 0
        dae_omega_acc = (omega[1:] - omega[:-1]) - self.h * omega_acc[:-1]  # == 0
        initial_x = x[0] - self.x0  # == 0
        initial_y = y[0] - self.y0  # == 0
        final_x = x[-1] - self.x_final  # == 0
        final_y = y[-1] - self.y_final  # == 0
        cons = np.concatenate(
            (
                dae_x,
                dae_y,
                dae_theta,
                dae_acc,
                dae_omega_acc,
                np.array((initial_x, initial_y, final_x, final_y)),
            ),
            axis=0,
        )
        self.num_equality_constraints = len(cons)

        # Inequality constraints (Obstacles)
        for cx, cy, rx, ry in zip(self.cxs, self.cys, self.rxs, self.rys):
            if self.obstacle_shape == "circle":
                cons = np.concatenate(
                    (cons, self.circle_obstacle(x, y, cx, cy, rx, ry)), axis=0
                )  # >= 0
            elif self.obstacle_shape == "square":
                cons = np.concatenate(
                    (cons, self.square_obstacle(x, y, cx, cy, rx, ry)), axis=0
                )  # >= 0
            elif self.obstacle_shape == "diamond":
                cons = np.concatenate(
                    (cons, self.diamond_obstacle(x, y, cx, cy, rx, ry)), axis=0
                )  # >= 0

        self.num_inequality_constraints = len(cons) - self.num_equality_constraints
        return cons

    @property
    def cl(self) -> np.ndarray:
        """Constraint lower bounds

        Constraints are written to all have zero lower bounds

        :return: np.ndarray. Constraint lower bounds.
        """
        return np.zeros((self.m,))

    @property
    def cu(self) -> np.ndarray:
        """Constraint upper bounds

        Equality constraints have an upper bound of zero. Inequality constraints
        have upper bound of infinity

        :return: np.ndarray. Constraint upper bounds.
        """
        upper_bound = np.zeros((self.m,))
        upper_bound[self.num_equality_constraints :] = np.inf
        return upper_bound

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        """Structure of the constraint jacobian

        :return: tuple[np.ndarray,np.ndarray]. Rows (constraint number) and
            columns (variable number) of the corresponding Jacobian value
        """
        return self.memoized_jac_structure

    @cached_property
    def memoized_jac_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """Cached jacobian structure

        Because the Jacobian structure does not change, we simply save the
        structure

        :return: tuple[np.ndarray,np.ndarray]. Rows (constraint number) and
            columns (variable number) of the corresponding Jacobian value
        """
        rows = []
        columns = []
        # values = []
        row = 0
        for i in range(self.N_x - 1):
            columns += [
                i + 1 + self.x_shift,
                i + self.x_shift,
                i + self.theta_shift,
                i + self.v_shift,
            ]
            rows += [row] * 4
            # values += [1, -1, self.h * np.sin(theta[i]) * v[i], -self.h * np.cos(theta[i])]
            row += 1
        for i in range(self.N_y - 1):
            columns += [
                i + 1 + self.y_shift,
                i + self.y_shift,
                i + self.theta_shift,
                i + self.v_shift,
            ]
            rows += [row] * 4
            # values += [1, -1, -self.h * np.cos(theta[i]) * v[i], -self.h * np.sin(theta[i])]
            row += 1
        for i in range(self.N_theta - 1):
            columns += [
                i + 1 + self.theta_shift,
                i + self.theta_shift,
                i + self.omega_shift,
            ]
            rows += [row] * 3
            # values += [1, -1, -self.h]
            row += 1
        for i in range(self.N_v - 1):
            columns += [i + 1 + self.v_shift, i + self.v_shift, i + self.acc_shift]
            rows += [row] * 3
            # values += [1, -1, -self.h]
            row += 1
        for i in range(self.N_omega - 1):
            columns += [
                i + 1 + self.omega_shift,
                i + self.omega_shift,
                i + self.omega_acc_shift,
            ]
            rows += [row] * 3
            # values += [1, -1, -self.h]
            row += 1

        # Initial x
        columns += [0 + self.x_shift]
        rows += [row]
        # values += [1]
        row += 1

        # Initial y
        columns += [0 + self.y_shift]
        rows += [row]
        # values += [1]
        row += 1

        # Final x
        columns += [self.NK - 1 + self.x_shift]
        rows += [row]
        # values += [1]
        row += 1

        # Final y
        columns += [self.NK - 1 + self.y_shift]
        rows += [row]
        # values += [1]
        row += 1

        for _cx, _cy, _rx, _ry in zip(self.cxs, self.cys, self.rxs, self.rys):
            for i in range(self.NK):
                columns += [i + self.x_shift, i + self.y_shift]
                rows += [row, row]
                # values += [dx[i], dy[i]]
                row += 1
        return (np.array(rows, dtype=int), np.array(columns, dtype=int))

    def jacobian(self, combined_vars: np.ndarray) -> np.ndarray:
        """Jacobian values of constraints

        :param vars: np.ndarray. Combined discretized variables
        :return: np.ndarray. Non-zero values of the Jacobian corresponding to
            the row and column of the jacobian structure
        """
        (x, y, theta, v, _acc, _omega, _omega_acc) = self.var_splitter(combined_vars)
        rows, _columns = self.jacobianstructure()
        values = np.zeros(rows.shape)
        index = 0

        # DAE x integration (forward Euler)
        # x[i+1] - x[i] - h * cos(theta[i]) * v[i] = 0
        for i in range(self.N_x - 1):
            values[index : index + 4] = [
                1,
                -1,
                self.h * np.sin(theta[i]) * v[i],
                -self.h * np.cos(theta[i]),
            ]
            index += 4

        # DAE y integration (forward Euler)
        # y[i+1] - y[i] - h * sin(theta[i]) * v[i] = 0
        for i in range(self.N_y - 1):
            values[index : index + 4] = [
                1,
                -1,
                -self.h * np.cos(theta[i]) * v[i],
                -self.h * np.sin(theta[i]),
            ]
            index += 4

        # DAE theta integration (forward Euler)
        # theta[i+1] - theta[i] - h * omega[i] = 0
        for i in range(self.N_theta - 1):
            values[index : index + 3] = [1, -1, -self.h]
            index += 3

        # DAE v integration (forward Euler)
        # v[i+1] - v[i] - h * acc[i] = 0
        for i in range(self.N_v - 1):
            values[index : index + 3] = [1, -1, -self.h]
            index += 3

        # DAE omega integration (forward Euler)
        # omega[i+1] - omega[i] - h * omega_acc[i] = 0
        for i in range(self.N_omega - 1):
            values[index : index + 3] = [1, -1, -self.h]
            index += 3

        # Initial x
        values[index] = 1
        index += 1
        # Initial y
        values[index] = 1
        index += 1
        # Final x
        values[index] = 1
        index += 1
        # Final y
        values[index] = 1
        index += 1

        for cx, cy, rx, ry in zip(self.cxs, self.cys, self.rxs, self.rys):
            for i in range(self.N_x):
                if self.obstacle_shape == "circle":
                    values[index : index + 2] = [
                        2 * (x[i] - cx) / (rx**2),
                        2 * (y[i] - cy) / (ry**2),
                    ]
                elif self.obstacle_shape == "square":
                    values[index : index + 2] = [
                        np.sign((x[i] - cx) / rx + (y[i] - cy) / ry) / rx / 2
                        + np.sign((x[i] - cx) / rx - (y[i] - cy) / ry) / rx / 2,
                        np.sign((x[i] - cx) / rx + (y[i] - cy) / ry) / ry / 2
                        - np.sign((x[i] - cx) / rx - (y[i] - cy) / ry) / ry / 2,
                    ]
                elif self.obstacle_shape == "diamond":
                    values[index : index + 2] = [
                        np.sign(x[i] - cx) / rx,
                        np.sign(y[i] - cy) / ry,
                    ]
                index += 2
        return values


# problem_obj = Objective()

# problem = cyipopt.Problem(
#     n=problem_obj.n,  # Number of decision variables
#     m=problem_obj.m,  # Number of constraints
#     problem_obj=problem_obj,  # Objective object with "objective" and "gradient" functions
#     lb=problem_obj.lb,  # Lower bounds for x
#     ub=problem_obj.ub,  # Upper bounds for x
#     cl=problem_obj.cl,  # Lower bound for constraints
#     cu=problem_obj.cu,  # Lower bound for constraints
# )
