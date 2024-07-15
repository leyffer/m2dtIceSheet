import sys

sys.path.insert(0, "../source/")
import fenics as dl
import numpy as np
from myState import myState


class myState_stationary(myState):
    state: dl.Function
    convolution = None
    grid_t = [0]

    def __init__(
            self,
            fom: "FOM",
            state: dl.Function,
            bool_is_transient: bool,
            parameter: np.ndarray,
            other_identifiers: dict,
            **kwargs
    ) -> None:

        if bool_is_transient:
            raise RuntimeError("myState_stationary.__init__ called for transient state")

        super().__init__(
            fom, state, bool_is_transient, parameter, other_identifiers, **kwargs
        )

    def get_derivative(self, t=None, x=None):

        """
        computes and saves the spatial derivative of the state
        @return:
        """
        if self.Du is None:
            Du = dl.grad(self.state)
            self.Du = dl.project(Du, self.gradient_space)

        if x is not None:
            return self.Du(x)

        return self.Du

    def get_state(self, t=None, x=None):
        """
        returns the state of the modelled system at a given time. In the stationary setting, the same state is returned
        for all possible times

        @return:
        """
        if x is not None:
            return self.state(x)
        return self.state
