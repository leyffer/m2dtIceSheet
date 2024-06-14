import sys

sys.path.insert(0, "../source/")
import fenics as dl
import numpy as np
from State import State


class myState(State):
    state: dl.Function
    convolution = None

    def __init__(
        self,
        fom: "FOM",
        state: dl.Function,
        bool_is_transient: bool,
        parameter: np.ndarray,
        other_identifiers: dict,
        **kwargs
    ) -> None:
        super().__init__(
            fom, state, bool_is_transient, parameter, other_identifiers, **kwargs
        )
        # Gradient space is Vector space of one dimension lower than FOM dimension
        self.gradient_space = dl.VectorFunctionSpace(fom.mesh, 'DG', fom.polyDim - 1) # Discontinuous Galerkin

    def get_derivative(self):

        """
        computes and saves the spatial derivative of the state
        @return:
        """
        if self.Du is None:
            if self.bool_is_transient:
                # TODO: implement derivative for time dependent states (it's easy, I'm just lazy)
                raise RuntimeError(
                    "spatial derivative for transient state still needs to be implemented"
                )

            else:
                Du = dl.grad(self.state)
                self.Du = dl.project(Du, self.gradient_space)

        return self.Du

    def measure_pointwise(
        self, position: np.ndarray, time: float | np.ndarray
    ) -> np.ndarray:
        """
        Given positions of type [x,y], return the value of the state at the positions
        """
        if self.bool_is_transient:
            # implement some sort of time interpolation here
            raise NotImplementedError
        else:
            return np.array(
                [self.state(x, y) for x, y in zip(position[:, 0], position[:, 1])]
            )

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
