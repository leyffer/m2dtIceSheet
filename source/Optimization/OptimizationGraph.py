from typing import Callable, Optional

import numpy as np
from InverseProblem import InverseProblem
from OEDUtility import OEDUtility
from scipy import optimize
import Optimization

# from torch import inverse


class OptimizationGraph(Optimization):
    """! Optimization class
    In this class we solve the flight path optimization problem. In particular, we:

    - set up the optimization problem in <optimization library, e.g., IPOPT>
    - apply our sequential (partially) linearized algorithm
    - set up the E-OED variant

    """

    def minimize():
        raise NotImplementedError("TODO")
