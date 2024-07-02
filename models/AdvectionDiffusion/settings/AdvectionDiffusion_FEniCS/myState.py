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
        self.gradient_space = dl.VectorFunctionSpace(fom.mesh, 'DG', fom.polyDim)

        self.final_time = self.grid_t[-1]

    def get_derivative(self, t: float = None, x=None):

        """
        computes and saves the spatial derivative of the state
        @return:
        """
        # make sure we don't recompute the derivative every time it gets called
        if self.Du is None:

            self.Du = np.zeros(self.n_steps, dtype=object)
            for k in range(self.n_steps):
                du = dl.grad(self.state[k])
                self.Du[k] = dl.project(du, self.gradient_space)

        if t is None:
            raise RuntimeError(
                "transient state myState.get_derivative called without specifying a time at which to evaluate")

        if x is not None:
            return self.apply_interpolation_rule(states=self.Du, t=t, x=x)
        def evaluate_Du(x):
            return self.apply_interpolation_rule(states=self.Du, t=t, x=x)

        # note: returning a function that evaluates the derivative does cause additional overhead and is not the most
        # elegant solution. However, the way the addition of the derivatives works in FEniCS, I couldn't find a simple
        # way to evaluate them at a given point. It works much nicer for the states. (Nicole, May 28, 2024)

        return evaluate_Du

    def get_state(self, t=None, x=None):
        if t is None:
            raise RuntimeError(
                "transient state myState.get_state called without specifying a time at which to evaluate")

        return self.apply_interpolation_rule(states=self.state, t=t, x=x)

    def apply_interpolation_rule(self, states, t, x=None):

        if t > self.final_time:
            raise RuntimeError("transient state called for time step beyond simulation horizon")

        k_right = np.argmax(self.grid_t >= t)

        if self.grid_t[k_right] == t:
            if x is None:
                return states[k_right]
            else:
                return states[k_right](x)

        if k_right == 0:
            raise RuntimeError(
                "In myState.apply_interpolation_rule: encountered k_right = 0. This happened for time={}".format(t))

        t_left = self.grid_t[k_right - 1]
        t_right = self.grid_t[k_right]
        t_diff = t_right - t_left

        state_left = states[k_right - 1]
        state_right = states[k_right]
        state_diff = state_right - state_left

        if x is not None:
            eval_left = state_left(x)
            return eval_left + (t - t_left) * (state_right(x) - eval_left) / t_diff

        # apply linear interpolation rule
        return state_left + (t - t_left) * state_diff / t_diff

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

    def measure_pointwise(
            self, position: np.ndarray, t: float | np.ndarray
    ) -> np.ndarray:
        """
        Given positions of type [x,y], return the value of the state at the positions

        # todo: does this function get called? Intuitively this feels redundant with the code in the pointwise drone
        """
        print(
            "myState_stationary.measure_pointwise got called. Why? Shouldn't the pointwise drone class take care of this?")

        state = self.get_state(t=t)
        return np.array(
            [state(x, y) for x, y in zip(position[:, 0], position[:, 1])]
        )


# todo: the functions below - why are they here? What are they used for? They seems out of place to me (Nicole, May 28, 2024)
def make_circle_kernel(radius: float, dx: float) -> np.ndarray:
    """!
    Make a circular uniform kernel

    @param radius  the radius of the kernel
    @param dx  the grid spacing of the space that the kernel will be applied to
    @return  a 2D kernel centered at zero with 1's everywhere within the radius
        of zero

    TODO - allow for values between 0-1 on the boundary of the circle
        (anti-aliasing based on fraction of a grid square filled)
    """
    w = int(np.ceil(radius / dx))
    x = np.linspace(-w * dx, w * dx, 2 * w + 1)
    y = np.linspace(-w * dx, w * dx, 2 * w + 1)
    X, Y = np.meshgrid(x, y)
    return (X**2 + Y**2 < radius**2).astype(
        float
    )  # no partial cells, but that would be nice


def make_truncated_gaussian_kernel(
    radius: float, dx: float, sigma: float
) -> np.ndarray:
    """!
    Make a truncated Gaussian kernel

    @param radius  the radius of the kernel (truncation)
    @param dx  the grid spacing of the space that the kernel will be applied to
    @param sigma  the sigma parameter of the Gaussian
    """
    w = int(np.ceil(radius / dx))
    x = np.linspace(-w * dx, w * dx, 2 * w + 1)
    y = np.linspace(-w * dx, w * dx, 2 * w + 1)
    X, Y = np.meshgrid(x, y)
    r_squared = X**2 + Y**2
    truncation = (r_squared < radius**2).astype(
        float
    )  # no partial cells, but that would be nice
    return (
        np.exp(-0.5 * r_squared / (sigma**2))
        / sigma
        / np.sqrt(2 * np.pi)
        * truncation
    )
