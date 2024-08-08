import sys

sys.path.insert(0, "../source/")
import fenics as dl
import numpy as np
from myState import myState


class myState_stationary(myState):
    state: dl.Function
    convolution = None
    grid_t = np.array([0])
    n_steps = 1

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
            try:
                out = self.Du(x)
                return out
            except RuntimeError:
                # TODO - find a way to get the dimension without using protected members
                return self.fom.n_spatial * [0.0]

        return self.Du

    def get_state(self, t=None, x=None):
        """
        returns the state of the modelled system at a given time. In the stationary setting, the same state is returned
        for all possible times

        @return:
        """
        if x is None:
            return self.state
        return self.state(x)

    def apply_interpolation_rule(self, states, t, x=None):
        """
        applies this state's interpolation rule for t to the passed collection of states. The ordering in the argument
        states needs to correspond to self.grid_t, such that states[0] is for t=0.

        In the stationary setting, we always return the evaluation of the very first entry of states, corresponding
        to t=0.
        """
        if x is None:
            return states[0]
        return states[0](x)
