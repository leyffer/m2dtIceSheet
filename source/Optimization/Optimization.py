import cyipopt

import numpy as np
from InverseProblem import InverseProblem
from OEDUtility import OEDUtility
from Flight import Flight


class Optimization(cyipopt.Problem):
    """! Optimization class
    In this class we solve the flight path optimization problem. In particular, we:

    - set up the optimization problem in <optimization library, e.g., IPOPT>
    - apply our sequential (partially) linearized algorithm
    - set up the E-OED variant

    """

    reg_strength = 1

    def __init__(
            self, utility: OEDUtility, inverse_problem: InverseProblem, constraints,
    ) -> None:
        """! Initialization for Optimization class

        @param utility: OEDUtility object, includes the information about the utility function, its gradient, etc.
        """
        self.inversion = inverse_problem
        self.utility = utility
        self.navigation = inverse_problem.drone.navigation
        self.my_constraints = constraints

        self.n_positions = self.navigation.n_timesteps * self.navigation.n_spatial
        self.n_auxiliary = self.navigation.n_timesteps
        # todo: formalize, this is for DAE right now
        self.n_controls = self.navigation.n_controls
        self.n_dofs = self.n_positions + self.n_controls + self.n_auxiliary

        super().__init__(
            n=self.n_dofs,
            m=self.my_constraints.n_constraints,
            lb=self.lower_bounds,
            ub=self.upper_bounds,
            cl=self.my_constraints.constraints_lower,
            cu=self.my_constraints.constraints_upper,
        )

    def OED_objective(self, combined_vars: np.ndarray) -> float:
        """OED objective function

        Given variables, split them into (x,y) discrete observations to create
        the data observation locations. Compute the posterior using those
        positions and estimated measurements using the `InverseProblem` instance
        `self.inversion`. Using the posterior, compute the specified OED utility
        function using the `OEDUtility` instance `self.utility`.

        Arguments:
            @param combined_vars  Combined variable vector
        """
        # Split the combined variable vector into the component variables
        (flightpath_1d, controls) = self.var_splitter(combined_vars)

        # initialize flight
        flight = Flight(navigation=self.navigation, flightpath=flightpath_1d)

        # compute posterior
        posterior = self.inversion.compute_posterior(flight=flight)
        # todo: ideally we would want to store this posterior to reuse it when computing the derivatives

        # evaluate utility function
        return self.utility.eval_utility(posterior=posterior)

    def OED_gradient(self, combined_vars: np.ndarray) -> np.ndarray:
        """Gradient of OED objective function

        Takes a combined variable vector and returns the gradient of the OED
        objective function. The gradient is computed as a derivative with
        respect to position such that d obj/dx and d obj/dy are returned.

        Arguments:
            @param combined_vars  Combined variable vector
        """
        # Split the combined variable vector into the component variables
        (flightpath_1d, controls) = self.var_splitter(combined_vars)

        # initialize flight
        flight = Flight(navigation=self.navigation, flightpath=flightpath_1d)

        # compute posterior
        posterior = self.inversion.compute_posterior(flight=flight)
        # todo: ideally we would want to store this posterior to reuse it when computing the derivatives

        # compute the derivative
        derivative = self.utility.d_utility_d_position(posterior=posterior)

        # Since only derivatives w.r.t. x and y were computed, we need to fill
        # out the rest of gradient for the other variables as zeros
        out = np.concatenate(
            (derivative, np.zeros((self.n_auxiliary + self.n_controls,)))
        )
        return out

    def regularization_objective(self, combined_vars: np.ndarray) -> float:
        """Regularization objective function

        This can take a variety of forms. Here we regularize on the sum-squared
        values of acc and omega_acc (if normalized by self.h, this would be the
        integral of acc**2 and omega_acc**2)

        @param vars  np.ndarray. Combined variables
        @return np.ndarray  Regularization objective value
        """
        (flightpath_1d, controls) = self.var_splitter(combined_vars)
        # todo: use navigation class to define a regularization term
        return 0

    def regularization_gradient(self, combined_vars: np.ndarray):
        """Regularization objective function gradient

        This is the gradient of the above regularization objective

        @param vars np.ndarray  Combined variables
        @return np.ndarray  Gradient w.r.t. the regularization objective
        """
        (flightpath_1d, controls) = self.var_splitter(combined_vars)
        # todo: use navigation class to define a regularization term
        return np.zeros((self.n_dofs,))

    def objective(self, combined_vars: np.ndarray) -> float:
        """Objective function

        Combination of OED objective and regularization objective

        @param vars: np.ndarray. Combined variables
        @return np.ndarray. Objective value
        """
        self.objective_value = self.OED_objective(
            combined_vars
        ) + self.reg_strength * self.regularization_objective(combined_vars)
        return self.objective_value

    def gradient(self, combined_vars: np.ndarray) -> np.ndarray:
        """Objective function gradient

        Combination of OED objective and regularization gradient

        @param vars: np.ndarray. Combined variables
        @return np.ndarray. Gradient of objective w.r.t. variables
        """

        return self.OED_gradient(
            combined_vars
        ) + self.reg_strength * self.regularization_gradient(combined_vars)

    def var_joiner(
            self,
            *args,
    ) -> np.ndarray:
        """Join variables into a single vector

        Take separate variables and join them into a single vector

        @param args: tuple[np.ndarray]. Variables to join
        @return np.ndarray. Concatenated vector containing all of the DAE
            variables
        """
        return np.concatenate(args, axis=0)

    def var_splitter(self, combined_vars: np.ndarray) -> tuple[np.ndarray]:
        """Split variables from a single vector

        Split the combined variable vector into individual variables

        @param combined_vars  Combined variable vector
        @return  A namedtuple object as defined at the top of this file
        """
        return combined_vars[:self.n_positions + self.n_auxiliary], combined_vars[self.n_positions + self.n_auxiliary:]

    @property
    def lower_bounds(self) -> np.ndarray:
        """Lower bounds on variables

        Takes lower bounds on variables from provided options, creates vectors
        with those values, and combines them into a single vector (may need to
        be more complex for different treatment of the discretization)

        @return np.ndarray. Vector of lower bounds on all discretized variables
        """
        return self.my_constraints.lower_bounds

    @property
    def upper_bounds(self) -> np.ndarray:
        """Upper bound on variables

        Takes upper bounds on variables from provided options, creates vectors
        with those values, and combines them into a single vector (may need to
        be more complex for different treatment of the discretization)

        @return np.ndarray. Vector of upper bounds on all discretized variables
        """
        return self.my_constraints.upper_bounds

    def constraints(self, combined_vars):
        (flightpath_1d, controls) = self.var_splitter(combined_vars)
        return self.my_constraints.evaluate_constraints(flightpath_1d=flightpath_1d, alpha=controls)

    def jacobian(self, combined_vars):
        (flightpath_1d, controls) = self.var_splitter(combined_vars)
        return self.my_constraints.evaluate_jacobian(flightpath_1d=flightpath_1d, alpha=controls)

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        return self.my_constraints.memorized_jac_structure
