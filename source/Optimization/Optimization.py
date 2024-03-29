from typing import Callable, Optional

import numpy as np
from InverseProblem import InverseProblem
from OEDUtility import OEDUtility
from scipy import optimize

# from torch import inverse


class Optimization:
    """! Optimization class
    In this class we solve the flight path optimization problem. In particular, we:

    - set up the optimization problem in <optimization library, e.g., IPOPT>
    - apply our sequential (partially) linearized algorithm
    - set up the E-OED variant

    """

    def __init__(
        self, utility: OEDUtility, inverse_problem: InverseProblem, mode: str = "A"
    ) -> None:
        """! Initialization for Optimization class

        @param utility: OEDUtility object, includes the information about the utility function, its gradient, etc.
        """
        # self.fom = utility.fom
        # self.drone = utility.drone
        self.inversion = inverse_problem
        self.utility = utility
        self.mode = mode
        # note (Nicole, Oct 31, 2023):
        # not sure if the Optimization class really needs to know all these objects or just the utility
        # I'm including them for now for consistency. Sven and Thomas can take them out if they are not needed

    # TODO: write the other functions required for this class

    def minimize(self, *args, **kwargs):
        raise NotImplementedError("Implement in subclass")
