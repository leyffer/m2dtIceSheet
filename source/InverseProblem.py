from FullOrderModel import FullOrderModel as FOM
from Drone import Drone

class InverseProblem():
    """! InverseProblem class
    In this class we provide all functions needed for handling the inverse problem, starting from its setup to its
    solution. In particular, for the OED problem, we provide:

    - a call that applies the inverse posterior covariance matrix for given flight path parameters
    - a call to compute the posterior mean
    - the option to apply a reduction in parameter space (e.g., with active subspaces)

    Note: the details on the last part are not clear yet
    """

    def __init__(self, fom: FOM, drone: Drone) -> None:
        """! Initialization for InverseProblem class instance

        @param fom: Forward model, also specifies the prior
        @param drone: specifies how the measurements are taken (for given flight parameters)
        """
        self.fom = fom
        self.drone = drone

    # TODO: write other functions required for this class
    # TODO: from the old source files, copy over all computations
    # TODO: set up connection to hIppylib

