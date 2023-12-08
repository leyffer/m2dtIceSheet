from OEDUtility import OEDUtility


class Optimization:
    """! Optimization class
    In this class we solve the flight path optimization problem. In particular, we:

    - set up the optimization problem in <optimization library, e.g., IOPT>
    - apply our sequential (partially) linearized algorithm
    - set up the E-OED variant

    """

    def __init__(self, utility: OEDUtility) -> None:
        """! Initialization for Optimization class

        @param utility: OEDUtility object, includes the information about the utility function, its gradient, etc.
        """
        self.fom = utility.fom
        self.drone = utility.drone
        self.inversion = utility.inversion
        self.utility = utility
        # note (Nicole, Oct 31, 2023):
        # not sure if the Optimization class really needs to know all these objects or just the utility
        # I'm including them for now for consistency. Sven and Thomas can take them out if they are not needed

    # TODO: write the other functions required for this class
