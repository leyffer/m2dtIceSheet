import sys
import numpy as np

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

    def d_measurement_d_control(self, flight, state, navigation):
        """
        derivative of the measurement function for a given flightpath in control direction alpha

        The derivative is computed via the chain rule. We use FEniCS functionaly to get the spatial derivative of the
        state and then evaluate it in a point-wise manner.

        @param alpha:
        @param flightpath:
        @param grid_t:
        @param state:
        @return: np.ndarray of shape (grid_t.shape[0], self.n_parameterss)
        """
        
        alpha, flightpath, grid_t = flight.alpha, flight.flightpath, flight.grid_t

        # parts of the chain rule (only compute once)
        grad_p = flight.d_position_d_control()   # derivative of position
        # todo: optimize this computation such that we don't repeat it as often
        Du = state.get_derivative()  # spatial derivative of the state

        # initialization
        D_data_d_alpha = np.empty((grid_t.shape[0], alpha.shape[0]))

        for i in range(grid_t.shape[0]):
            # the FEniCS evaluation of the Du at a position unfortunately doesn't work with multiple positions
            # that's why we can't get rid of this loop

            # apply chain rule
            if state.bool_is_transient:
                # todo: extend to transient measurements
                raise NotImplementedError(
                    "In MyDronePointEval.d_measurement_d_control: still need to bring over code for transient measurements")
            else:
                # state is time-independent
                D_data_d_alpha[i, :] = Du(flightpath[i, :]) @ grad_p[:, :, i].T

        return D_data_d_alpha