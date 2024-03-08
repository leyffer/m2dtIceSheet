import sys

sys.path.insert(0, "../source/")
import fenics as dl
import numpy as np
from State import State


class myState(State):
    state: dl.Function

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

    def get_derivative(self) -> dl.MultiMeshFunction | dl.Function:
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
                self.Du = dl.project(Du)

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
