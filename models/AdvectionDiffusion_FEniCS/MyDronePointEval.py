import numpy as np

from MyDrone import MyDrone

class MyDronePointEval(MyDrone):

    center = np.array([0.75/2, 0.55/2])

    def __init__(self, fom, grid_t=None, **kwargs):
        """! Initializer for the drone class with point-wise measurements
        @param fom  Full-order-model (FOM) object. The drone takes
        measurements from this
        @param grid_t the time grid the drone should fly in
        @param **kwargs  Keyword arguments including `sigma_gaussian`
        (Gaussian radius) and `radius_uniform`
        """
        super().__init__(fom=fom, eval_mode="point-eval", grid_t=grid_t, **kwargs)

    def measure(self, flightpath, grid_t, state) -> np.ndarray:
        """! Get measurements along the flight path at the drone location

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        if state.bool_is_transient:
            # todo: extend to transient measurements
            raise NotImplementedError("In MyDrone.measure_pointwise: still need to bring over code for transient measurements")
            # old code:
            # return [state[k].at(*flightpath[k, :]) for k in range(flightpath.shape[0])]

        return np.array([state.state(flightpath[k, :]) for k in range(flightpath.shape[0])])

    def d_measurement_d_control(self, alpha, flightpath, grid_t, state):
        """
        derivative of the measurement function for a given flightpath in control direction alpha

        @param alpha:
        @param flightpath:
        @param grid_t:
        @param state:
        @return: np.ndarray of shape (grid_t.shape[0], self.n_parameterss)
        """

        # parts of the chain rule (only compute once)
        grad_p = self.d_position_d_control(alpha, flightpath, grid_t)  # derivative of position
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