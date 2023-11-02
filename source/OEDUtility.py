from InverseProblem import InverseProblem
import numpy as np
import scipy.linalg as la

class OEDUtility():
    """! OEDUtility class
    In this class we provide all information about the OED utility function (for A-, D-, and E-optimal experimental
    design). In particular, we provide:

    - evaluate the utility of a given flight path design (A-, D-, and E)
    - compute the eigenvalues of the posterior covariance matrix for a given flight path design
    - evaluate the gradient of the utility function
    """

    # todo: decide if we should outsource the different OED modes into subclasses

    def __init__(self, inversion : InverseProblem, default_mode : str = None) -> None:
        """! initialization for OEDUtility class

        @param inversion: inverse problem for which the OED utility shall be computed
        """

        self.fom = inversion.fom
        self.drone = inversion.drone
        self.inversion = inversion

        self.default_mode = default_mode if default_mode is not None else "D"

    # TODO: specifiy all function call names that this class needs to have to interact with the other classes
    # TODO: from the old source files, copy over all computations

    def eval_utility(self, posterior, mode=None):
        mode = mode if mode is not None else self.default_mode

        if mode == "A":
            return self.eval_utility_A(posterior)

        if mode == "D":
            return self.eval_utility_D(posterior)

        if mode == "E":
            return self.eval_utility_E(posterior)

        raise RuntimeError("Invalid oed_mode encountered: {}".format(mode))

    def eval_utility_A(self, posterior):
        return sum(posterior.get_eigenvalues())

    def eval_utility_D(self, posterior):
        # todo: should we compute the inverse or the log instead? The values get very big
        return np.prod(posterior.get_eigenvalues())

    def eval_utility_E(self, posterior):
        return np.max(posterior.get_eigenvalues())