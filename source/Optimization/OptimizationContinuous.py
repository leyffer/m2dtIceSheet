from typing import Callable, Optional

import numpy as np
from InverseProblem import InverseProblem
from OEDUtility import OEDUtility
from scipy import optimize
import Optimization

# from torch import inverse


class OptimizationContinuous(Optimization):
    """! Optimization class
    In this class we solve the flight path optimization problem. In particular, we:

    - set up the optimization problem in <optimization library, e.g., IPOPT>
    - apply our sequential (partially) linearized algorithm
    - set up the E-OED variant

    """

    def minimize(
        self,
        alpha_initial: np.ndarray,
        method: str = "Nelder-Mead",
        mask: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Wrapper for scipy.optimize.minimize"""

        if mask is not None:
            fixed_alpha = alpha_initial[~mask]
            unfixed_alpha = alpha_initial[mask]

            def fun_masked(alpha):
                posterior_ref = self.inversion.compute_posterior(
                    np.hstack((fixed_alpha, alpha))
                )
                return self.utility.eval_utility(posterior_ref, mode=self.mode)

            def jac_masked(alpha):
                posterior_ref = self.inversion.compute_posterior(
                    np.hstack((fixed_alpha, alpha))
                )
                return self.utility.d_utility_d_control(posterior_ref, mode=self.mode)[
                    mask
                ]

            a = optimize.minimize(
                fun_masked, unfixed_alpha, jac=jac_masked, method=method, **kwargs
            )
            a.x = np.hstack((fixed_alpha, a.x))

            return a

        else:

            def fun(alpha):
                posterior_ref = self.inversion.compute_posterior(alpha)
                return self.utility.eval_utility(posterior_ref, mode=self.mode)

            def jac(alpha):
                posterior_ref = self.inversion.compute_posterior(alpha)
                return self.utility.d_utility_d_control(posterior_ref, mode=self.mode)

            return optimize.minimize(
                fun, alpha_initial, jac=jac, method=method, **kwargs
            )
