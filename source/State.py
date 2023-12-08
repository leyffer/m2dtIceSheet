class State:
    """! State class

    The state class is a generic way of passing state solutions of the full-order model around without accounting for
    the structure the solution might have (e.g., it might have several state variables). The implemented functions below
    are set up for the general case (only one state, can be transient or stationary), but the user might want to
    overwrite this class for their specific needs.
    """

    state = None  # where the state variable gets saved, e.g. as np.ndarray
    parameter = None  # the parameter sample for which the solution was obtained, if any

    other_identifiers = None
    # other setup parameters involved in obtaining this state (e.g., nuisance parameters)
    # We don't anticipate this attribute to be used by us, but it might be helpful for reproducibility

    bool_is_transient = None  # whether the solution is time dependent
    grid_t = None  # time discretization of the state solution
    Du = None

    def __init__(
        self, fom, state, bool_is_transient, parameter, other_identifiers, *kwargs
    ) -> None:
        """! Initialization for State class

        @param fom: FullOrderModel instance
        @param state: encodes the solution, e.g., as np.ndarray or FEniCS Function, Firedrake object, etc.
        @param bool_is_transient: bool
        @param parameter: parameter for which the solution was obtained
        @param other_identifiers: solver setup parameters (for reproducibility)
        @param kwargs: should contain:
            "grid_t" if the solution is transient
        """
        # background information
        self.fom = fom  # the model where this state came from
        self.bool_is_transient = bool_is_transient
        if bool_is_transient:
            self.grid_t = kwargs.get("grid_t")

        # information about the state solution itself
        self.state = state
        self.parameter = parameter
        self.other_identifiers = other_identifiers

    def get_derivative(self):
        """
        computes and saves the spatial derivative of the state
        @return:
        """
        raise NotImplementedError(
            "user must implement State.get_derivative in subclass"
        )
