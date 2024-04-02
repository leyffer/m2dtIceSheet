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

        data = np.array([state.state(flightpath[k, :]) for k in range(flightpath.shape[0])])
        return data

    def d_measurement_d_position(self, flight, state):
        """
        derivative of the measurement function for a given flightpath in position.
        The derivative is the gradient of the  state along the flightpath. We can use FEniCS to get the derivative.

        @param alpha:
        @param flightpath:
        @param grid_t:
        @param state:
        @return: np.ndarray of shape (grid_t.shape[0], self.n_parameters)
        """
        flightpath, grid_t = flight.flightpath, flight.grid_t
        
        # # compute convolution with delta function (not needed for pointwise)
        # convolution = self.compute_convolution(state=state)
        Du = state.get_derivative()

        # initialization
        D_data_d_position = np.empty((grid_t.shape[0], 2))  # (time, (dx,dy))

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
                D_data_d_position[i, :] = Du(flightpath[i, :])

        return D_data_d_position

