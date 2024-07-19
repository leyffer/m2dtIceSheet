import sys
import numpy as np
import warnings

sys.path.insert(0, "../source/")

from Detector import Detector
# from State import State
# from Flight import Flight

class DetectorPointwise(Detector):
    """
    In this drone class, we model measurements at time t of a state u to be of the form:
    d(t) = u(p(t), t)
    where p(t) is the position of the drone at time t. Compared to the other convolution-type measurements, this is the
    cheapest way to compute measurements. However, in infinite-dimensional function space, the point evaluation is
    not necessarily well defined.
    """
    center = np.array([0.75/2, 0.55/2])

    def __init__(self, grid_t:np.ndarray=None, **kwargs):
        """! Initializer for the drone class with point-wise measurements

        @param fom  Full-order-model (FOM) object. The drone takes
        measurements from this
        @param grid_t the time grid the drone should fly in
        @param **kwargs  Keyword arguments including `sigma_gaussian`
        (Gaussian radius) and `radius_uniform`
        """
        super().__init__(grid_t=grid_t, **kwargs)

    def compute_convolution(self, state: "State"):
        """Pointwise measurements are taken from the un-convolved state."""
        warnings.warn("For pointwise measurements, measurements are taken from the un-convolved state.")
        return state

    def measure_at_position(self, pos, t, state):
        try:
            data = state.get_state(t=t, x=pos)
        except RuntimeError:
            warnings.warn(f"DetectorPointwise.measure: flightpath goes outside of computational domain")
            data = 0  # Need to return a value to not crash optimization (value does not matter, but zero is best)
            pass

        return data

    def derivative_at_position(self, pos, t, state):
        """
        computes the state's measurement derivative (w.r.t. the position) at time t and position pos
        """
        try:
            # get derivative at the prescribed position
            data = state.get_derivative(t=t, x=pos)
        except RuntimeError:
            warnings.warn(
                f"DetectorPointwise.measure: flightpath goes outside of computational domain")
            pass

        return data
