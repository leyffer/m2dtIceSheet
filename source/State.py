"""Dataclass class for the PDE state and its descriptors"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class State:
    """! State class

    The state class is a generic way of passing state solutions of the
    full-order model around without accounting for the structure the solution
    might have (e.g., it might have several state variables). The implemented
    functions below are set up for the general case (only one state, can be
    transient or stationary), but the user might want to overwrite this class
    for their specific needs.

    TODO - clean up: as a dataclass, much of the __init__ is redundant
    TODO - confusing State attribute also named state
    """

    # the PDE state, e.g. as np.ndarray of dl.Function
    state: np.ndarray

    # the parameter sample for which the solution was obtained, if any
    parameter: np.ndarray

    other_identifiers: Optional[Any] = None
    # other setup parameters involved in obtaining this state (e.g., nuisance
    # parameters)
    # We don't anticipate this attribute to be used by us, but it might be
    # helpful for reproducibility

    # whether the solution is time dependent
    bool_is_transient: bool = None

    # time discretization of the state solution (may be different from the time
    # grid for the measurements of the state)
    grid_t: np.ndarray = None

    # spatial derivative of the PDE state
    Du = None

    measurement_memory = None

    def __init__(
        self,
        fom: "FullOrderModel",
        state: np.ndarray,
        bool_is_transient: bool,
        parameter: np.ndarray,
        other_identifiers: Optional[Any],
        **kwargs
    ) -> None:
        """! Initialization for State class

        @param fom: FullOrderModel instance
        @param state: a PDE solution state, e.g., as np.ndarray or FEniCS Function, Firedrake object, etc.
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
            if self.grid_t is None:
                raise RuntimeError(
                    "No time discretization was passed for transient state"
                )
            self.n_steps = self.grid_t.shape[0]

        # information about the state solution itself
        self.state = state
        self.parameter = parameter
        self.other_identifiers = other_identifiers

    def get_derivative(self, t: Optional[float] = None):
        """
        Computes and saves the spatial derivative of the state

        @param t  time to get the derivative at (for a transient state)
        @return: spatial derivative (at time t for a transient state)
        """
        raise NotImplementedError(
            "user must implement State.get_derivative in subclass"
        )

    def get_state(self, t: Optional[float] = None):
        """
        Returns the state of the modelled system at a given time

        @param t  time to get the PDE state at (for a transient state)
        @return: PDE solution state (at time t for a transient state)
        """
        raise NotImplementedError("user must implement State.get_state in subclass")

    def remember_measurement(self, pos: np.ndarray, t: float):
        """
        If a measurement has been take at a position pos and time t, retain that
        information for later.

        This is useful when taking a measurement is costly.

        @param pos  position the measurement is taken at
        @param t  time the measurement is taken at
        """
        raise NotImplementedError(
            "user must implement State.remember_measurement in subclass"
        )
