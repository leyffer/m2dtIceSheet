import sys
sys.path.insert(0, "../source/")
from State import State

import fenics as dl

class myState(State):

    convolution = None

    def get_derivative(self):
        """
        computes and saves the spatial derivative of the state
        @return:
        """
        if self.Du is None:
            if self.bool_is_transient:
                # todo: implement derivative for time dependent states (it's easy, I'm just lazy)
                raise RuntimeError("spatial derivative for transient state still needs to be implemented")

            else:
                Du = dl.grad(self.state)
                self.Du = dl.project(Du)

        return self.Du

    def set_convolution(self, convolution, key):
        """
        this function is intended for the MyDroneGaussianEval class to save the convoluted state computed in
        MyDroneGaussianEval.measure such that it does not need to get re-computed for other flight paths or when
        taking the derivative. The "key" parameter is there to distinguish between different drones measuring this
        state.

        Discussion:
        Right now, we only consider one drone at a time, so the key is probably not strictly necessary, but I think
        it's probably good practice to build it in already. In particular, we can probably think of other use cases
        beyond the MyDroneGaussianEval class

        @param convolution:
        @param key: unique identifier (string) of the drone
        @return:
        """
        if self.convolution is None:
            self.convolution = {}
        self.convolution[key] = convolution

    def get_convolution(self, key):
        if self.convolution is None:
            return None
        return self.convolution[key]

