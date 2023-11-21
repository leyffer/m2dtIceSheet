import sys
sys.path.insert(0, "../source/")
from State import State

import fenics as dl

class myState(State):

    def get_derivative(self):
        """
        computes and saves the spatial derivative of the state
        @return:
        """
        if self.Du is None:
            if self.bool_is_transient:
                # TODO: implement derivative for time dependent states (it's easy, I'm just lazy)
                raise RuntimeError("spatial derivative for transient state still needs to be implemented")

            else:
                Du = dl.grad(self.state)
                self.Du = dl.project(Du)

        return self.Du