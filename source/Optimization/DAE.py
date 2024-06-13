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
# import jax.numpy as jnp
import numpy as np

jnp = np
# import numpy as jnp

from InverseProblem import InverseProblem
from OEDUtility import OEDUtility

from collections import namedtuple

# from scipy.integrate import RK45

# TODO - add "movie" output

DAE_vars = namedtuple("DAE_vars", ["x", "y", "theta", "v", "acc", "omega", "omega_acc"])


class Objective(cyipopt.Problem):
    """Objective class to work with cyipopt"""

    def __init__(
        self,
        grid_t: np.ndarray,
        oed_utility: OEDUtility,
        inversion: InverseProblem,
        **kwargs,
    ):
        self.utility = oed_utility
        self.inversion = inversion

        # Initial position
        self.enforce_initial_position = kwargs.get("enforce_initial_position", True)
        self.x0 = kwargs.get("x0", 0.7)
        self.y0 = kwargs.get("y0", 0.3)
        # Initial heading
        self.enforce_initial_heading = kwargs.get("enforce_initial_heading", False)
        self.theta0 = kwargs.get("theta0", np.pi / 2)

        # Final position
        self.enforce_final_position = kwargs.get("enforce_final_position", False)
        self.x_final = kwargs.get("x_final", 0.1)
        self.y_final = kwargs.get("y_final", 0.1)
        # Final heading
        self.theta_final = kwargs.get("theta_final", jnp.pi)

        # Use the exact DAE (otherwise just use finite differences)
        self.use_exact_DAE = kwargs.get("use_exact_DAE", False)
        self.linear_tolerance = kwargs.get("linear_tolerance", 1e-3)

        # Build a video of the optimization process
        self.build_video = kwargs.get("build_video", False)
        if self.build_video:
            self.video_combined_vars = []
            self.video_constraint_violation = []
            self.video_objective = []
            self.video_frames = []

        # Obstacles
        # Centers
        self.cxs = kwargs.get("cxs", [(0.5 + 0.25) / 2, (0.75 + 0.6) / 2])
        self.cys = kwargs.get("cys", [(0.4 + 0.15) / 2, (0.85 + 0.6) / 2])
        # Radii
        self.rxs = kwargs.get("rxs", [(0.5 - 0.25) / 2, (0.75 - 0.6) / 2])
        self.rys = kwargs.get("rys", [(0.4 - 0.15) / 2, (0.85 - 0.6) / 2])
        # Shape
        self.obstacle_shape = kwargs.get("obstacle_shape", "square")

        # Bound constraints
        # x
        self.x_lower = kwargs.get("x_lower", 0.0)
        self.x_upper = kwargs.get("x_upper", 1.0)
        # y
        self.y_lower = kwargs.get("y_lower", 0.0)
        self.y_upper = kwargs.get("y_upper", 1.0)
        # theta (unbounded)
        self.theta_lower = kwargs.get("theta_lower", -jnp.inf)
        self.theta_upper = kwargs.get("theta_upper", jnp.inf)
        # v
        self.v_lower = kwargs.get("v_lower", 0.1)
        self.v_upper = kwargs.get("v_upper", 3.0)
        # acc
        self.acc_lower = kwargs.get("acc_lower", -10.0)
        self.acc_upper = kwargs.get("acc_upper", 10.0)
        # omega
        self.omega_lower = kwargs.get("omega_lower", -20.0 * jnp.pi)
        self.omega_upper = kwargs.get("omega_upper", 20.0 * jnp.pi)
        # omega_acc
        self.omega_acc_lower = kwargs.get("omega_acc_lower", -200.0)
        self.omega_acc_upper = kwargs.get("omega_acc_upper", 200.0)

        # Can limit the number of changes for acceleration to get piecewise constant controls
        # TODO - separate the time grid and control grid (grids should be aligned)
        self.piecewise_constant = kwargs.get("piecewise_constant", None)

        self.T = kwargs.get("T", grid_t[-1] - grid_t[0])  # Total time

        self.NK = kwargs.get("NK", grid_t.shape[0])  # Number of control variables
        self.N_x = kwargs.get("N_x", self.NK)  # Number of x
        self.N_y = kwargs.get("N_y", self.NK)  # Number of y
        self.N_theta = kwargs.get("N_theta", self.NK)  # Number of theta
        self.N_v = kwargs.get("N_v", self.NK)  # Number of v
        self.N_acc = kwargs.get("N_acc", self.NK)  # Number of acc
        self.N_omega = kwargs.get("N_omega", self.NK)  # Number of omega
        self.N_omega_acc = kwargs.get("N_omega_acc", self.NK)  # Number of omega_acc

        self.grid_t = grid_t  # Time grid
        self.h = self.T / self.NK  # Time grid spacing
        self.reg_strength = kwargs.get("reg_strength", 0.000001)
        # Regularization strength

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
        self.cum_var_lengths = DAE_vars(*tuple(jnp.cumsum(jnp.array(self.var_lengths))))
        self.indices = DAE_vars(
            *tuple(
                jnp.arange(cum_length - length, cum_length, 1, dtype=int)
                for cum_length, length in zip(self.cum_var_lengths, self.var_lengths)
            )
        )

        self.number_of_variables = self.omega_acc_shift + self.N_omega_acc  # number of variables

        # Simplified two parameter mode
        self.circle_mode = kwargs.get("circle_mode", False)
        self.circle_center_x = kwargs.get("circle_center_x", 0.75 / 2)
        self.circle_center_y = kwargs.get("circle_center_y", 0.55 / 2)
        if self.circle_mode:
            self.enforce_final_position = False
            self.enforce_initial_position = False
            self.enforce_initial_heading = True
            self.acc_lower = 0.0
            self.acc_upper = 0.0
            self.omega_acc_lower = 0.0
            self.omega_acc_upper = 0.0

        self.num_equality_constraints = None
        self.num_inequality_constraints = None
        # Do an initialization of the constraints to get constraint numbers
        self.objective_value = 0.0
        self.constraints(jnp.ones((self.number_of_variables,)))
        super().__init__(
            n=self.number_of_variables,
            m=self.number_of_constraints,
            lb=self.lower_bounds,
            ub=self.upper_bounds,
            cl=self.constraint_lower_bounds,
            cu=self.constraint_upper_bounds,
        )

    def OED_objective(self, combined_vars: jnp.ndarray) -> float:
        """OED objective function"""
        (x, y, _theta, _v, _acc, _omega, _omega_acc) = self.var_splitter(combined_vars)
        return self.utility.eval_utility_A(
            self.inversion.compute_posterior(
                alpha=jnp.concatenate((x, y), axis=0), grid_t=self.grid_t
            )
        )

    def OED_gradient(self, combined_vars: jnp.ndarray) -> jnp.ndarray:
        """Gradient of OED objective function"""
        (x, y, _theta, _v, _acc, _omega, _omega_acc) = self.var_splitter(combined_vars)
        out = self.utility.d_utilA_d_position(
            self.inversion.compute_posterior(
                alpha=jnp.concatenate((x, y), axis=0), grid_t=self.grid_t
            )
        )
        out = jnp.concatenate((out, jnp.zeros((self.number_of_variables - self.N_x - self.N_y,))))
        return out

    def regularization_objective(self, combined_vars: jnp.ndarray) -> float:
        """Regularization objective function

        :param vars: jnp.ndarray. Combined variables
        :return: jnp.ndarray. Regularization objective value
        """
        (_x, _y, _theta, _v, acc, _omega, omega_acc) = self.var_splitter(combined_vars)
        # return (jnp.sum(v**2) + jnp.sum(omega**2))
        return jnp.sum(acc**2) + jnp.sum(omega_acc**2)

    def regularization_gradient(self, combined_vars: jnp.ndarray):
        """Regularization objective function gradient

        :param vars: jnp.ndarray. Combined variables
        :return: jnp.ndarray. Gradient w.r.t. the regularization objective
        """
        (x, y, theta, v, acc, omega, omega_acc) = self.var_splitter(combined_vars)

        x = jnp.zeros(x.shape)
        y = jnp.zeros(y.shape)
        theta = jnp.zeros(theta.shape)
        v = jnp.zeros(v.shape)
        # v = 2*v
        # acc = jnp.zeros(acc.shape)
        acc = 2 * acc
        omega = jnp.zeros(omega.shape)
        # omega = 2*omega
        # omega_acc = jnp.zeros(omega_acc.shape)
        omega_acc = 2 * omega_acc

        return self.var_joiner(x, y, theta, v, acc, omega, omega_acc)

    def objective(self, combined_vars: jnp.ndarray) -> float:
        """Objective function

        Combination of OED objective and regularization objective

        :param vars: jnp.ndarray. Combined variables
        :return: jnp.ndarray. Objective value
        """
        self.objective_value = self.OED_objective(
            combined_vars
        ) + self.reg_strength * self.regularization_objective(combined_vars)
        return self.objective_value

    def gradient(self, combined_vars: jnp.ndarray) -> jnp.ndarray:
        """Objective function gradient

        Combination of OED objective and regularization gradient

        :param vars: jnp.ndarray. Combined variables
        :return: jnp.ndarray. Gradient of objective w.r.t. variables
        """

        return self.OED_gradient(
            combined_vars
        ) + self.reg_strength * self.regularization_gradient(combined_vars)

    @property
    def number_of_constraints(self) -> int:
        """Number of constraints

        :return: int. Number of constraints
        """
        return self.num_equality_constraints + self.num_inequality_constraints

    def var_joiner(
        self,
        *args,
    ) -> jnp.ndarray:
        """Join variables into a single vector

        :param args: tuple[jnp.ndarray]. Variables to join
        :return: jnp.ndarray. Concatenated vector containing all of the DAE
            variables
        """
        return jnp.concatenate(args, axis=0)

    def var_splitter(self, combined_vars: jnp.ndarray) -> tuple[jnp.ndarray]:
        """Split variables from a single vector"""
        return DAE_vars(*tuple(jnp.split(combined_vars, self.cum_var_lengths[:-1])))

    @property
    def lower_bounds(self) -> jnp.ndarray:
        """Lower bounds on variables

        :return: jnp.ndarray. Vector of lower bounds on all discretized variables
        """
        x_lb = jnp.ones((self.N_x,)) * self.x_lower
        y_lb = jnp.ones((self.N_y,)) * self.y_lower
        theta_lb = jnp.ones((self.N_theta,)) * self.theta_lower
        v_lb = jnp.ones((self.N_v,)) * self.v_lower
        if self.piecewise_constant is not None:
            acc_lb = np.zeros((self.N_acc,))
            acc_lb[:: self.N_acc // self.piecewise_constant] = self.acc_lower
        else:
            acc_lb = jnp.ones((self.N_acc,)) * self.acc_lower
        omega_lb = jnp.ones((self.N_omega,)) * self.omega_lower
        if self.piecewise_constant is not None:
            omega_acc_lb = np.zeros((self.N_omega_acc,))
            omega_acc_lb[:: self.N_omega_acc // self.piecewise_constant] = (
                self.omega_acc_lower
            )
        else:
            omega_acc_lb = jnp.ones((self.N_omega_acc,)) * self.omega_acc_lower
        return self.var_joiner(
            x_lb, y_lb, theta_lb, v_lb, acc_lb, omega_lb, omega_acc_lb
        )

    @property
    def upper_bounds(self) -> jnp.ndarray:
        """Upper bound on variables

        :return: jnp.ndarray. Vector of upper bounds on all discretized variables
        """
        x_ub = self.x_upper * jnp.ones((self.N_x,))
        y_ub = self.y_upper * jnp.ones((self.N_y,))
        theta_ub = self.theta_upper * jnp.ones((self.N_theta,))
        v_ub = self.v_upper * jnp.ones((self.N_v,))
        if self.piecewise_constant is not None:
            acc_ub = np.zeros((self.N_acc,))
            acc_ub[:: self.N_acc // self.piecewise_constant] = self.acc_upper
        else:
            acc_ub = jnp.ones((self.N_acc,)) * self.acc_upper
        omega_ub = self.omega_upper * jnp.ones((self.N_omega,))
        if self.piecewise_constant is not None:
            omega_acc_ub = np.zeros((self.N_omega_acc,))
            omega_acc_ub[:: self.N_omega_acc // self.piecewise_constant] = (
                self.omega_acc_upper
            )
        else:
            omega_acc_ub = jnp.ones((self.N_omega_acc,)) * self.omega_acc_upper
        return self.var_joiner(
            x_ub, y_ub, theta_ub, v_ub, acc_ub, omega_ub, omega_acc_ub
        )

    def circle_obstacle(
        self, x: jnp.ndarray, y: jnp.ndarray, cx: float, cy: float, rx: float, ry: float
    ) -> jnp.ndarray:
        """Circular (ellipse) obstacle

        ((x - xc)/rx)**2 + ((y - yc)/ry)**2 >= 1

        :param x: jnp.ndarray. Trajectory x-positions
        :param y: jnp.ndarray. Trajectory y-positions
        :param cx: float. Circle center x
        :param cy: float. Circle center y
        :param rx: float. Circle (ellipse) radius x
        :param ry: float. Circle (ellipse) radius y
        :return: jnp.ndarray. Distance outside of circle (should be non-negative)
        """
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 - 1

    def diamond_obstacle(
        self, x: jnp.ndarray, y: jnp.ndarray, cx: float, cy: float, rx: float, ry: float
    ) -> jnp.ndarray:
        """Diamond obstacle

        |(x - xc)/rx| + |(y - yc)/ry| >= 1

        :param x: jnp.ndarray. Trajectory x-positions
        :param y: jnp.ndarray. Trajectory y-positions
        :param cx: float. Diamond center x
        :param cy: float. Diamond center y
        :param rx: float. Diamond radius x
        :param ry: float. Diamond radius y
        :return: jnp.ndarray. Distance outside of diamond (should be
            non-negative)
        """
        return jnp.abs((x - cx) / rx) + jnp.abs((y - cy) / ry) - 1

    def square_obstacle(
        self, x: jnp.ndarray, y: jnp.ndarray, cx: float, cy: float, rx: float, ry: float
    ) -> jnp.ndarray:
        """Square (rectangle) obstacle

        Rotated 1-norm
        |(x - xc)/rx + (y - yc)/ry|/2 + |(x - xc)/rx - (y - yc)/ry|/2 >= 1
        or
        Infinity norm
        max(|x - xc|/rx, |y - yc|/ry) >= 1

        :param x: jnp.ndarray. Trajectory x-positions
        :param y: jnp.ndarray. Trajectory y-positions
        :param cx: float. Square center x
        :param cy: float. Square center y
        :param rx: float. Square (rectangle) radius x
        :param ry: float. Square (rectangle) radius y
        :return: jnp.ndarray. Distance outside of square (should be non-negative)
        """
        return (
            jnp.abs(((x - cx) / rx + (y - cy) / ry) / 2)
            + jnp.abs(((x - cx) / rx - (y - cy) / ry) / 2)
            - 1
        )

    # @jax.jit
    def constraints(self, combined_vars):
        """Values to constrain between cl and cu"""
        # Equality constraints
        (x, y, theta, v, acc, omega, omega_acc) = self.var_splitter(combined_vars)

        # Finite difference
        dae_x = (x[1:] - x[:-1]) - self.h * jnp.cos(theta[:-1]) * v[:-1]  # == 0
        dae_y = (y[1:] - y[:-1]) - self.h * jnp.sin(theta[:-1]) * v[:-1]  # == 0

        if self.use_exact_DAE:
            # Piecewise constant undefined when omega -> 0)
            dae_x_exact = (x[1:] - x[:-1]) - (v[:-1] / omega[:-1]) * (
                jnp.sin(theta[:-1] + omega[:-1] * self.h) - jnp.sin(theta[:-1])
            )  # == 0
            dae_y_exact = (y[1:] - y[:-1]) - (v[:-1] / omega[:-1]) * (
                -jnp.cos(theta[:-1] + omega[:-1] * self.h) + jnp.cos(theta[:-1])
            )  # == 0

            indicator = jnp.abs(omega[:-1]) >= self.linear_tolerance
            for i, ind in enumerate(indicator):
                if ind:
                    dae_x[i] = dae_x_exact[i]
                    dae_y[i] = dae_y_exact[i]
                    # dae_x.at[i].set(dae_x_exact[i])
                    # dae_y.at[i].set(dae_y_exact[i])

        dae_theta = (theta[1:] - theta[:-1]) - self.h * omega[:-1]  # == 0
        # Average acceleration
        dae_acc = (v[1:] - v[:-1]) - self.h * acc[:-1]  # == 0
        dae_omega_acc = (omega[1:] - omega[:-1]) - self.h * omega_acc[:-1]  # == 0

        cons = jnp.concatenate(
            (
                dae_x,
                dae_y,
                dae_theta,
                dae_acc,
                dae_omega_acc,
            ),
            axis=0,
        )
        if self.enforce_initial_position:
            initial_x = x[0] - self.x0  # == 0
            initial_y = y[0] - self.y0  # == 0
            cons = jnp.concatenate((cons, jnp.array((initial_x, initial_y))), axis=0)
        if self.enforce_final_position:
            final_x = x[-1] - self.x_final  # == 0
            final_y = y[-1] - self.y_final  # == 0
            cons = jnp.concatenate((cons, jnp.array((final_x, final_y))), axis=0)
        if self.enforce_initial_heading:
            initial_heading = theta[0] - self.theta0  # ==0
            cons = jnp.concatenate((cons, jnp.array((initial_heading,))), axis=0)

        if self.circle_mode:
            initial_x_circle = x[0] - (
                self.circle_center_x
                + v[0] / omega[0] * jnp.cos(self.theta0 - jnp.pi / 2)
            )  # == 0
            initial_y_circle = y[0] - (
                self.circle_center_y
                + v[0] / omega[0] * jnp.sin(self.theta0 - jnp.pi / 2)
            )  # == 0
            cons = jnp.concatenate((cons, jnp.array((initial_x_circle, initial_y_circle))), axis=0)

        self.num_equality_constraints = len(cons)

        # Inequality constraints (Obstacles)
        for cx, cy, rx, ry in zip(self.cxs, self.cys, self.rxs, self.rys):
            if self.obstacle_shape == "circle":
                cons = jnp.concatenate(
                    (cons, self.circle_obstacle(x, y, cx, cy, rx, ry)), axis=0
                )  # >= 0
            elif self.obstacle_shape == "square":
                cons = jnp.concatenate(
                    (cons, self.square_obstacle(x, y, cx, cy, rx, ry)), axis=0
                )  # >= 0
            elif self.obstacle_shape == "diamond":
                cons = jnp.concatenate(
                    (cons, self.diamond_obstacle(x, y, cx, cy, rx, ry)), axis=0
                )  # >= 0

        self.num_inequality_constraints = len(cons) - self.num_equality_constraints
        return cons

    @property
    def constraint_lower_bounds(self) -> jnp.ndarray:
        """Constraint lower bounds

        Constraints are written to all have zero lower bounds

        :return: jnp.ndarray. Constraint lower bounds.
        """
        return jnp.zeros((self.number_of_constraints,))

    @property
    def constraint_upper_bounds(self) -> jnp.ndarray:
        """Constraint upper bounds

        Equality constraints have an upper bound of zero. Inequality constraints
        have upper bound of infinity

        :return: jnp.ndarray. Constraint upper bounds.
        """
        upper_bound = jnp.concatenate(
            (
                jnp.zeros((self.num_equality_constraints,)),
                jnp.inf * jnp.ones((self.num_inequality_constraints,)),
            ),
            axis=0,
        )
        return upper_bound

    def jacobianstructure(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Structure of the constraint jacobian

        :return: tuple[jnp.ndarray,jnp.ndarray]. Rows (constraint number) and
            columns (variable number) of the corresponding Jacobian value
        """
        return self.memoized_jac_structure

    @cached_property
    def memoized_jac_structure(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Cached jacobian structure

        Because the Jacobian structure does not change, we simply save the
        structure

        :return: tuple[jnp.ndarray,jnp.ndarray]. Rows (constraint number) and
            columns (variable number) of the corresponding Jacobian value
        """
        rows = []
        columns = []
        # values = []
        row = 0
        for i in range(self.N_x - 1):
            if self.use_exact_DAE:
                columns += [
                    i + 1 + self.x_shift,
                    i + self.x_shift,
                    i + self.theta_shift,
                    i + self.v_shift,
                    i + self.omega_shift,
                ]
                rows += [row] * 5
            else:
                columns += [
                    i + 1 + self.x_shift,
                    i + self.x_shift,
                    i + self.theta_shift,
                    i + self.v_shift,
                ]
                rows += [row] * 4
            # values += [1, -1, self.h * jnp.sin(theta[i]) * v[i], -self.h * jnp.cos(theta[i])]
            row += 1
        for i in range(self.N_y - 1):
            if self.use_exact_DAE:
                columns += [
                    i + 1 + self.y_shift,
                    i + self.y_shift,
                    i + self.theta_shift,
                    i + self.v_shift,
                    i + self.omega_shift,
                ]
                rows += [row] * 5
            else:
                columns += [
                    i + 1 + self.y_shift,
                    i + self.y_shift,
                    i + self.theta_shift,
                    i + self.v_shift,
                ]
                rows += [row] * 4
            # values += [1, -1, -self.h * jnp.cos(theta[i]) * v[i], -self.h * jnp.sin(theta[i])]
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

        if self.enforce_initial_position:
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

        if self.enforce_final_position:
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

        if self.enforce_initial_heading:
            # Initial theta
            columns += [0 + self.theta_shift]
            rows += [row]
            # values += [1]
            row += 1

        if self.circle_mode:
            # Initial x
            columns += [0 + self.x_shift, 1 + self.v_shift, 1 + self.omega_shift]
            rows += [row] * 3
            # values += [1, 1/omega[1], -v[1]/omega[1]**2]
            row += 1

            # Initial y
            columns += [0 + self.y_shift, 1 + self.v_shift, 1 + self.omega_shift]
            rows += [row] * 3
            # values += [1, 1/omega[1], -v[1]/omega[1]**2]
            row += 1

        for _cx, _cy, _rx, _ry in zip(self.cxs, self.cys, self.rxs, self.rys):
            for i in range(self.NK):
                columns += [i + self.x_shift, i + self.y_shift]
                rows += [row, row]
                # values += [dx[i], dy[i]]
                row += 1
        return (jnp.array(rows, dtype=int), jnp.array(columns, dtype=int))

    def jacobian(self, combined_vars: jnp.ndarray) -> jnp.ndarray:
        """Jacobian values of constraints

        :param vars: jnp.ndarray. Combined discretized variables
        :return: jnp.ndarray. Non-zero values of the Jacobian corresponding to
            the row and column of the jacobian structure
        """
        (x, y, theta, v, _acc, omega, _omega_acc) = self.var_splitter(combined_vars)
        rows, _columns = self.jacobianstructure()
        values = np.zeros(rows.shape)
        index = 0

        # DAE x integration (forward Euler)
        # x[i+1] - x[i] - h * cos(theta[i]) * v[i] = 0
        for i in range(self.N_x - 1):
            if self.use_exact_DAE:
                if jnp.abs(omega[i]) >= self.linear_tolerance:
                    values[index : index + 5] = [
                        1,
                        -1,
                        -(v[i] / omega[i])
                        * (jnp.cos(theta[i] + omega[i] * self.h) - jnp.cos(theta[i])),
                        -(1 / omega[i])
                        * (jnp.sin(theta[i] + omega[i] * self.h) - jnp.sin(theta[i])),
                        (v[i] / (omega[i] ** 2))
                        * (jnp.sin(theta[i] + omega[i] * self.h) - jnp.sin(theta[i]))
                        - (v[i] / omega[i])
                        * (self.h * jnp.cos(theta[i] + omega[i] * self.h)),
                    ]
                    index += 5
                else:
                    values[index : index + 5] = [
                        1,  # x[i + 1]
                        -1,  # x[i]
                        self.h * jnp.sin(theta[i]) * v[i],  # theta[i]
                        -self.h * jnp.cos(theta[i]),  # v[i]
                        0,  # omega[i]
                    ]
                    index += 5
            else:
                values[index : index + 4] = [
                    1,
                    -1,
                    self.h * jnp.sin(theta[i]) * v[i],
                    -self.h * jnp.cos(theta[i]),
                ]
                index += 4

        # DAE y integration (forward Euler)
        # y[i+1] - y[i] - h * sin(theta[i]) * v[i] = 0
        for i in range(self.N_y - 1):
            if self.use_exact_DAE:
                if jnp.abs(omega[i]) >= self.linear_tolerance:
                    values[index : index + 5] = [
                        1,
                        -1,
                        -(v[i] / omega[i])
                        * (jnp.sin(theta[i] + omega[i] * self.h) - jnp.sin(theta[i])),
                        -(1 / omega[i])
                        * (-jnp.cos(theta[i] + omega[i] * self.h) + jnp.cos(theta[i])),
                        v[i]
                        / (omega[i] ** 2)
                        * (-jnp.cos(theta[i] + omega[i] * self.h) + jnp.cos(theta[i]))
                        - (v[i] / omega[i])
                        * (self.h * jnp.sin(theta[i] + omega[i] * self.h)),
                    ]
                    index += 5
                else:
                    values[index : index + 5] = [
                        1,
                        -1,
                        -self.h * jnp.cos(theta[i]) * v[i],
                        -self.h * jnp.sin(theta[i]),
                        0,
                    ]
                    index += 5
            else:
                values[index : index + 4] = [
                    1,
                    -1,
                    -self.h * jnp.cos(theta[i]) * v[i],
                    -self.h * jnp.sin(theta[i]),
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

        if self.enforce_initial_position:
            # Initial x
            values[index] = 1
            index += 1
            # Initial y
            values[index] = 1
            index += 1

        if self.enforce_final_position:
            # Final x
            values[index] = 1
            index += 1
            # Final y
            values[index] = 1
            index += 1

        if self.enforce_initial_heading:
            # Initial theta
            values[index] = 1
            index += 1

        if self.circle_mode:
            # Initial x
            values[index : index + 3] = [
                1,
                -jnp.cos(self.theta0 - jnp.pi / 2) / omega[1],
                jnp.cos(self.theta0 - jnp.pi / 2) * v[1] / (omega[1] ** 2),
            ]
            index += 3
            # Initial y
            values[index : index + 3] = [
                1,
                -jnp.sin(self.theta0 - jnp.pi / 2) / omega[1],
                jnp.sin(self.theta0 - jnp.pi / 2) * v[1] / (omega[1] ** 2),
            ]
            index += 3

        for cx, cy, rx, ry in zip(self.cxs, self.cys, self.rxs, self.rys):
            for i in range(self.N_x):
                if self.obstacle_shape == "circle":
                    values[index : index + 2] = [
                        2 * (x[i] - cx) / (rx**2),
                        2 * (y[i] - cy) / (ry**2),
                    ]
                elif self.obstacle_shape == "square":
                    values[index : index + 2] = [
                        jnp.sign((x[i] - cx) / rx + (y[i] - cy) / ry) / rx / 2
                        + jnp.sign((x[i] - cx) / rx - (y[i] - cy) / ry) / rx / 2,
                        jnp.sign((x[i] - cx) / rx + (y[i] - cy) / ry) / ry / 2
                        - jnp.sign((x[i] - cx) / rx - (y[i] - cy) / ry) / ry / 2,
                    ]
                elif self.obstacle_shape == "diamond":
                    values[index : index + 2] = [
                        jnp.sign(x[i] - cx) / rx,
                        jnp.sign(y[i] - cy) / ry,
                    ]
                index += 2
        return values

    def video(self, combined_vars, constraints=None, objective=None):
        """
        Video hook

        Records variables, constraint violations, and objective over the course of an optimization.
        """
        if self.build_video:
            self.video_combined_vars.append(combined_vars)
            if constraints is None:
                constraints = self.constraints(combined_vars)
            self.video_constraint_violation.append(
                jnp.maximum(self.constraint_lower_bounds - constraints, 0.0)
                + jnp.maximum(constraints - self.constraint_upper_bounds, 0.0)
            )
            if objective is None:
                objective = self.objective(combined_vars)
            self.video_objective.append(objective)

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        """
        Intermediate callback is run at the end of every IPOPT iteration

        In this callback, we have additional access to IPOPT values by calling:
        self.get_current_iterate():
        Dict:
            "x" is variable values (primal)
            "mult_x_L" is multipliers for lower variable bounds
            "mult_x_U" is multipliers for upper variable bounds
            "mult_g" is multipliers for constraints

        self.get_current_violations()
        Dict:
            "x_L_violation" is violation of original lower bounds on variables
            "x_U_violation" is violation of original upper bounds on variables
            "compl_x_L" is violation of complementarity for lower bounds on variables
            "compl_x_U" is violation of complementarity for upper bounds on variables
            "grad_lag_x" is gradient of Lagrangian w.r.t. variables x
            "g_violation" is violation of constraints
            "compl_g" is complementarity of constraints

        Args:
            :param alg_mod: Algorithm phase: 0 is for regular, 1 is restoration.
            :param iter_count: The current iteration count.
            :param obj_value: The unscaled objective value at the current point
            :param inf_pr: The scaled primal infeasibility at the current point.
            :param inf_du: The scaled dual infeasibility at the current point.
            :param mu: The value of the barrier parameter.
            :param d_norm: The infinity norm (max) of the primal step.
            :param regularization_size: The value of the regularization term for the Hessian of the Lagrangian in the augmented system.
            :param alpha_du: The stepsize for the dual variables.
            :param alpha_pr: The stepsize for the primal variables.
            :param ls_trials: The number of backtracking line search steps.
        """
        if self.build_video:
            iterate = self.get_current_iterate()
            violations = self.get_current_violations()
            (x, y, theta, v, acc, omega, omega_acc) = self.var_splitter(iterate["x"])

            self.video_frames.append(
                {
                    "iterate": iterate,
                    "violations": violations,
                    "alg_mod": alg_mod,
                    "iter_count": iter_count,
                    "obj_value": obj_value,
                    "inf_pr": inf_pr,
                    "inf_du": inf_du,
                    "mu": mu,
                    "d_norm": d_norm,
                    "regularization_size": regularization_size,
                    "alpha_du": alpha_du,
                    "alpha_pr": alpha_pr,
                    "ls_trials": ls_trials,
                    "variables": {
                        "x": x,
                        "y": y,
                        "theta": theta,
                        "v": v,
                        "acc": acc,
                        "omega": omega,
                        "omega_acc": omega_acc,
                    },
                    "OED_objective": self.OED_objective(iterate["x"]),
                    "regularization_objective": self.regularization_objective(
                        iterate["x"]
                    ),
                }
            )


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
        self.vertices = jnp.array(vertices)

    def unit_vec(self, vector: jnp.ndarray):
        """Unit vector

        :param vector: jnp.ndarray. vector to convert to unit vector
        :return: jnp.ndarray. unit vector in the direction of vector
        """
        return vector / jnp.linalg.norm(vector)

    @property
    def side_vectors(self):
        """Unit vectors along perimeter of polygon"""
        return jnp.array(
            [
                self.unit_vec(self.vertices[i] - self.vertices[i - 1])
                for i in range(len(self.vertices))
            ]
        )

    @property
    def orthogonal_vectors(self):
        """Unit normal vectors for polygon sides"""
        return jnp.array([jnp.array([-v[1], v[0]]) for v in self.side_vectors])

    @property
    def constants(self):
        """
        orthogonal . x >= constant
        """
        return jnp.array(
            [
                jnp.dot(vertex, v)
                for vertex, v in zip(self.vertices, self.orthogonal_vectors)
            ]
        )