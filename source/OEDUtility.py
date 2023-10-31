from InverseProblem import InverseProblem

class OEDUtility():
    """! OEDUtility class
    In this class we provide all information about the OED utility function (for A-, D-, and E-optimal experimental
    design). In particular, we provide:

    - evaluate the utility of a given flight path design (A-, D-, and E)
    - compute the eigenvalues of the posterior covariance matrix for a given flight path design
    - evaluate the gradient of the utility function
    """

    def __init__(self, inversion : InverseProblem) -> None:
        """! initialization for OEDUtility class

        @param inversion: inverse problem for which the OED utility shall be computed
        """

        self.fom = inversion.fom
        self.drone = inversion.drone
        self.inversion = inversion

    # TODO: specifiy all function call names that this class needs to have to interact with the other classes
    # TODO: from the old source files, copy over all computations