"""Approximations to other detectors

Instead of computing a local convolution for a single measurement, this attempts
to compute the convolution of the entires state (numerically) and sample
pointwise from that. This relies on an equivalent definition of the data
measurement kernel operation for stationary kernels (convolutions):

d(p(t)) = ((u * Phi) / (indicator(Omega) * Phi)) (p(t))
"""

import sys

import numpy as np
import scipy.interpolate as scipy_interpolate
from scipy.ndimage import convolve  # , gaussian_filter

sys.path.insert(0, "../source/")
from Detector import Detector


def anti_alias(x: np.ndarray, scale: int) -> np.ndarray:
    """Downscale by two for each scale"""
    for _ in range(scale):
        x = 0.25 * (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2])
    return x


def make_circle_kernel(
    radius: float, dx: float, dy: float = None, anti_alias_scale: int = 4
) -> np.ndarray:
    """!
    Make a circular uniform kernel

    @param radius  the radius of the kernel
    @param dx  the grid spacing of the space that the kernel will be applied to
    @param anti_alias_scale  scale up the kernel by 2**anti_alias_scale and then scale down to get smoother kernel
    @return  a 2D kernel centered at zero with 1's everywhere within the radius
    """
    if dy is None:
        dy = dx
    wx = int(np.ceil(radius / dx))
    wy = int(np.ceil(radius / dy))
    x = np.linspace(-wx * dx, wx * dx, (2**anti_alias_scale) * (2 * wx + 1))
    y = np.linspace(-wy * dy, wy * dy, (2**anti_alias_scale) * (2 * wy + 1))
    X, Y = np.meshgrid(x, y)
    kernel = (X**2 + Y**2 < radius**2).astype(float)
    kernel = anti_alias(kernel, anti_alias_scale)
    kernel /= np.sum(kernel)
    return kernel


def make_truncated_gaussian_kernel(
    radius: float,
    dx: float,
    sigma: float,
    dy: float = None,
    anti_alias_scale: int = 4,
    truncate: bool = True,
) -> np.ndarray:
    """!
    Make a truncated Gaussian kernel

    @param radius  the radius of the kernel (truncation)
    @param dx  the grid spacing of the space that the kernel will be applied to
    @param sigma  the sigma parameter of the Gaussian
    @param anti_alias_scale  scale up the kernel by 2**anti_alias_scale and then scale down to get smoother kernel
    @return  a 2D kernel centered at zero with 0's everywhere outside the radius
    """
    if dy is None:
        dy = dx
    wx = int(np.ceil(radius / dx))
    wy = int(np.ceil(radius / dy))
    x = np.linspace(-wx * dx, wx * dx, (2**anti_alias_scale) * (2 * wx + 1))
    y = np.linspace(-wy * dy, wy * dy, (2**anti_alias_scale) * (2 * wy + 1))
    X, Y = np.meshgrid(x, y, indexing="ij")
    r_squared = X**2 + Y**2
    if sigma > 0:
        kernel = np.exp(-0.5 * r_squared / (sigma**2))
        if truncate:
            truncation = (r_squared < radius**2).astype(float)
            kernel *= truncation
    else:
        kernel = -r_squared + np.min(r_squared)
        kernel[kernel == 0] = 1.0
        kernel[kernel < 0] = 0.0
    kernel = anti_alias(kernel, anti_alias_scale)
    kernel /= np.sum(kernel)
    return kernel


class Convolution:
    """Contains approximation to the convolution of the state"""

    def __init__(
        self,
        fom: "FOM",
        state: "State",
        radius: float = 0.2,
        sigma: float = 0.1,
        mode: str = "apprx_gaussian",
        resolution: int = 100,
        debug: bool = False,
    ):
        """Assemble the approximations to the state

        @param fom  FullOrderModel probably not needed, but is currently used to
            get bounds on the FEM mesh
        @param state  State that we wish to approximate (not working for
            transient states)
        @param radius  radius of the stationary kernel
        @param sigma  the standard deviation of the Gaussian type kernels
        @param mode  the kernel type
        @param resolution  the number of grid points used to construct the
            interpolating spline approximation to the state
        @param debug  Add print statements to help debug
        """
        self.fom = fom
        self.state = state
        self.resolution = resolution
        self.coords = fom.mesh.coordinates()

        xx = np.linspace(
            np.min(self.coords[:, 0]), np.max(self.coords[:, 0]), self.resolution
        )
        dx = xx[1] - xx[0]
        yy = np.linspace(
            np.min(self.coords[:, 1]), np.max(self.coords[:, 1]), self.resolution
        )
        dy = yy[1] - yy[0]

        if mode.lower() in ["apprx_truncgaussian", "truncgaussian"]:
            self.kernel = make_truncated_gaussian_kernel(radius, dx, sigma, dy)
        elif mode.lower() in ["apprx_gaussian", "gaussian"]:
            self.kernel = make_truncated_gaussian_kernel(
                4 * sigma, dx, sigma, dy, truncate=False
            )
        elif mode.lower() in ["apprx_uniform", "uniform"]:
            self.kernel = make_circle_kernel(radius, dx, dy)
        elif mode.lower() in ["apprx_pointwise", "pointwise"]:
            self.kernel = np.array([[1.0]])
        else:
            raise ValueError(
                'Specify "apprx_uniform", "apprx_truncgaussian", "apprx_pointwise",'
                + 'or "apprx_gaussian" for kernel mode'
            )

        if debug:
            print("sampling the state")
        # Sample the state
        X, Y = np.meshgrid(xx, yy, indexing="ij")

        Z = np.zeros(X.reshape((-1,)).shape)
        indicator = np.ones(X.reshape((-1,)).shape)

        for (i, xv), yv in zip(enumerate(X.reshape((-1,))), Y.reshape((-1,))):
            try:
                Z[i] = self.state.state([xv, yv])
            except RuntimeError:
                indicator[i] = 0.0
                continue
        Z = Z.reshape(X.shape)
        indicator = indicator.reshape(X.shape)

        if debug:
            print("done sampling state")

        self.sampled_state = Z
        if debug:
            print("convolving the state")
        # Convolve and normalize for convolutions outside of the domain
        self.weight = np.maximum(
            convolve(indicator, self.kernel, mode="constant"),
            1e-16 * np.ones(indicator.shape),
        )
        self.convolved_state = (
            convolve(self.sampled_state, self.kernel, mode="constant") / self.weight
        ) * indicator

        if debug:
            print("building interpolator")

        self.interp = scipy_interpolate.RectBivariateSpline(
            xx, yy, self.convolved_state
        )

        self.interp_grad_x = self.interp.partial_derivative(1, 0)

        self.interp_grad_y = self.interp.partial_derivative(0, 1)

    def __call__(self, xi: np.ndarray, **kwargs) -> np.ndarray:
        """Get a value of the approximation at point(s) xi

        @param xi  points to get approximated state values at
        """
        return self.interp(xi[:, 0], xi[:, 1], grid=False, **kwargs)

    def grad(self, xi: np.ndarray, **kwargs) -> np.ndarray:
        """
        Return an estimate to the gradient using finite differences
        """
        grad_x = self.interp_grad_x(xi[:, 0], xi[:, 1], grid=False)
        grad_y = self.interp_grad_y(xi[:, 0], xi[:, 1], grid=False)
        return np.hstack((grad_x.reshape((-1, 1)), grad_y.reshape((-1, 1))))


class DetectorApprox(Detector):
    r"""
    This class uses an interpolated approximation to the state to speed up
    computation. A convolution is computed with a kernel depending on the
    provided mode. The Convolution object contains interpolated values using
    splines and provides gradients for those splines as well. Using different
    convolution kernels allows pointwise measurements for all modes, bypassing
    any need for computing convolutions at specific points (slow) whenever new
    points are provided.

    In this drone class, we model measurements at time t of a state u to be of
    the form:
    d(t) = 1/|c(t)| \int_{\Omega} u(x, t) Phi(x, p(t)) dx
    where
    - $Phi(x, y) = \frac{1}{\sqrt( (2 \pi \sigma^2 )^n ) \exp( - \| x-y \|^2 /
    (2 \sigma^2) )}$ is a multivariate Gaussian with covariance matrix sigma * I
    (sigma is given input parameter), and
    - c(t) = \int_{\Omega} Phi(x, p(t)) dx for normalization

    Computing the convolution over the whole domain at each time step is much
    less expensive than computing it per measurement point. By computing the
    convolution over the domain, we can use pointwise measurements on the
    convolved state.
    """

    def __init__(
        self,
        fom: "FOM",
        radius: float = 0.2,
        sigma: float = 0.1,
        resolution: int = 100,
        eval_mode: str = "apprx_gaussian",
        **kwargs
    ):
        """! Initializer for the drone class with point-wise measurements

        @param fom  Full-order-model (FOM) object. The drone takes
            measurements from states produced by this
        @param **kwargs  Keyword arguments including `sigma_gaussian`
            (Gaussian radius) and `radius_uniform`
        """
        super().__init__(**kwargs)
        self.radius = radius
        self.sigma = sigma
        self.fom = fom
        self.convolution = None
        self.resolution = resolution
        self.eval_mode = eval_mode

    def compute_convolution(self, state) -> Convolution:
        """
        We approximate the convolution of the state with the given kernel
        function
        """
        convolution = state.get_convolution(key=self.eval_mode)
        if convolution is not None:
            # don't recompute
            return convolution

        convolution = Convolution(
            self.fom,
            state,
            self.radius,
            self.sigma,
            self.eval_mode,
            self.resolution,
        )

        state.set_convolution(convolution=convolution, key=self.eval_mode)

        return convolution

    def measure(self, flight, state) -> np.ndarray:
        """! Get measurements along the flight path at the drone location

        To get the measurements, we proceed the following way:
        1) we compute the convolution of the state with the Gaussian everywhere
        in the domain. This step is outsourced to compute_convolution. We save
        the convolution as a field, so we never need to compute it again.
        2) We take point-wise evaluations of this convolution field.

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        flightpath = flight.flightpath
        grid_t = flight.grid_t

        # compute convolution with gaussian
        convolution = self.compute_convolution(state=state)

        return convolution(flightpath)

    def d_measurement_d_position(self, flight, state):
        r"""
        derivative of the measurement function for a given flight in direction
        of the flight's positions flightpath.
        For measurements of the form
        $$
        d(t; p) = \int_{\Omega} \Phi(x, p(t)) u(x,t) dx
        $$
        this function returns
        $$
        \frac{\partial d(t;, p)}{\partial p}
        = \int_{\Omega} D_y \Phi(x, y=p(t)) u(x, t) dx.
        $$

        Since the position is determined by <spatial dimension>*<number of time
        steps> parameters, and a measurement has <number of time steps> entries,
        the return of this function has to have shape

        $$ <number of time steps> \times <spatial dimension>*<number of time steps> $$

        The columns should be ordered such that the first <number of time steps>
        columns are for the first spatial dimension (x direction), the next
        <number of time steps> columns for the second (y-direction), etc.

        @param flight: the flight parameterization of the drone. Contains, in
        particular, the flightpath `flightpath`, the flight controls `alpha`,
        and the time discretization `grid_t`, Flight object
        @param state  The state which the drone shall measure, State object
        @return: np.ndarray of shape (grid_t.shape[0], <spatial dimension>)
        """
        if state.bool_is_transient:
            raise NotImplementedError(
                "In DetectorTruncGaussian(approx).d_measurement_d_position:"
                + " still need to bring over code for transient measurements"
            )

        flightpath, _grid_t = flight.flightpath, flight.grid_t
        n_spatial = flightpath.shape[1]

        # compute convolution with gaussian
        convolution = self.compute_convolution(state=state)
        D_data_d_position = convolution.grad(flightpath)  # (time, (dx,dy))

        # stack next to each other horizontally [diag(dx[:]), diag(dy[:])]
        D_data_d_position = np.hstack(
            [np.diag(D_data_d_position[:, i]) for i in range(n_spatial)]
        )
        return D_data_d_position


class DetectorTruncGaussian(DetectorApprox):
    """Truncated Gaussian detector"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, eval_mode="apprx_truncgaussian", **kwargs)


class DetectorGaussian(DetectorApprox):
    """Gaussian detector"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, eval_mode="apprx_gaussian", **kwargs)


class DetectorUniform(DetectorApprox):
    """Uniform detector"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, eval_mode="apprx_uniform", **kwargs)


class DetectorPointwise(DetectorApprox):
    """Pointwise detector"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, eval_mode="apprx_pointwise", **kwargs)
