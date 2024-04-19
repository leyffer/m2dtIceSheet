import sys
import numpy as np
import warnings

sys.path.insert(0, "../source/")

from Detector import Detector

class DetectorPointwise(Detector):
    """
    In this drone class, we model measurements at time t of a state u to be of the form:
    d(t) = u(p(t), t)
    where p(t) is the position of the drone at time t. Compared to the other convolution-type measurements, this is the
    cheapest way to compute measurements. However, in infinite-dimensional function space, the point evaluation is
    not necessarily well defined.
    """
    center = np.array([0.75/2, 0.55/2])

    def __init__(self, grid_t=None, **kwargs):
        """! Initializer for the drone class with point-wise measurements

        @param fom  Full-order-model (FOM) object. The drone takes
        measurements from this
        @param grid_t the time grid the drone should fly in
        @param **kwargs  Keyword arguments including `sigma_gaussian`
        (Gaussian radius) and `radius_uniform`
        """
        super().__init__(grid_t=grid_t, **kwargs)

    def compute_convolution(self, state):
        """Pointwise measurements are taken from the un-convolved state."""
        warnings.warn("For pointwise measurements, measurements are taken from the un-convolved state.")
        return state

    def measure(self, flight, state) -> np.ndarray:
        """! Get measurements along the flight path at the drone location

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        
        flightpath = flight.flightpath
        grid_t = flight.grid_t
        
        if state.bool_is_transient:
            # todo: extend to transient measurements
            raise NotImplementedError("In MyDrone.measure_pointwise: still need to bring over code for transient measurements")
            # old code:
            # return [state[k].at(*flightpath[k, :]) for k in range(flightpath.shape[0])]

        data = np.zeros((flightpath.shape[0],))
        for k in range(flightpath.shape[0]):
            try:
                data[k] = state.state(flightpath[k, :])
            except RuntimeError:
                warnings.warn(f"DetectorPointwise.measure: flightpath goes outside of computational domain")
                pass
        # data = np.array([state.state(flightpath[k, :]) for k in range(flightpath.shape[0])])
        return data

    def d_measurement_d_position(self, flight, state):
        """
        derivative of the measurement function for a given flight in direction of the flight's positions flightpath.
        For measurements of the form
        $$
        d(t; p) = \int_{\Omega} \Phi(x, p(t)) u(x,t) dx
        $$
        this function returns
        $$
        \frac{\partial d(t;, p)}{\partial p}
        = \int_{\Omega} D_y \Phi(x, y=p(t)) u(x, t) dx.
        $$

        Since the position is determined by <spatial dimension>*<number of time steps> parameters, and a measurement
        has <number of time steps> entries, the return of this function has to have shape

        $$ <number of time steps> \times <spatial dimension>*<number of time steps> $$

        The columns should be ordered such that the first <number of time steps> columns are for the first spatial
        dimension (x direction), the next <number of time steps> columns for the second (y-direction), etc.

        @param flight: the flight parameterization of the drone. Contains, in particular, the flightpath `flightpath`,
        the flight controls `alpha`, and the time discretization `grid_t`, Flight object
        @param state  The state which the drone shall measure, State object
        @return: np.ndarray of shape (grid_t.shape[0], <spatial dimension>)
        """
        flightpath, grid_t = flight.flightpath, flight.grid_t
        n_spatial = flightpath.shape[1]
        
        # # compute convolution with delta function (not needed for pointwise)
        # convolution = self.compute_convolution(state=state)
        Du = state.get_derivative()

        # initialization
        D_data_d_position = np.zeros((grid_t.shape[0], n_spatial))  # (time, (dx,dy))

        for i in range(grid_t.shape[0]):
            # the FEniCS evaluation of the Du at a position unfortunately
            # doesn't work with multiple positions that's why we can't get rid
            # of this loop

            # apply chain rule
            if state.bool_is_transient:
                # todo: extend to transient measurements
                raise NotImplementedError(
                    "In MyDronePointEval.d_measurement_d_position: still need to bring over code for transient measurements")
            else:
                # state is time-independent
                try:
                    D_data_d_position[i, :] = Du(flightpath[i, :])
                except RuntimeError:
                    pass

        # stack next to each other horizontally
        D_data_d_position = np.hstack([np.diag(D_data_d_position[:, i]) for i in range(n_spatial)])
        return D_data_d_position

