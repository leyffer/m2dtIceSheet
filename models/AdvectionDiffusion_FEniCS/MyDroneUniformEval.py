import numpy as np
import fenics as dl
import scipy.linalg as la
import mshr

from MyDrone import MyDrone

class MyDroneUniformEval(MyDrone):

    center = np.array([0.75/2, 0.55/2])

    def __init__(self, fom, grid_t=None, radius=0.1, **kwargs):
        """! Initializer for the drone class with point-wise measurements
        @param fom  Full-order-model (FOM) object. The drone takes
        measurements from this
        @param grid_t the time grid the drone should fly in
        @param **kwargs  Keyword arguments including `sigma_gaussian`
        (Gaussian radius) and `radius_uniform`
        """
        super().__init__(fom=fom, eval_mode="uniform", grid_t=grid_t, **kwargs)
        self.radius = radius

    def measure(self, flightpath, grid_t, state) -> np.ndarray:
        """! Get measurements along the flight path at the drone location

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """

        # initialization
        n_steps = flightpath.shape[0]
        data = np.NaN * np.ones((n_steps,))

        # define subdomain
        ref_domain = mshr.generate_mesh(mshr.Circle(c=dl.Point(0.0, 0.0), r=self.radius), 50)
        V = dl.FunctionSpace(ref_domain, 'P', 1)
        val_integral_ref = dl.assemble(1 * dl.dx(domain=ref_domain))

        for k in range(n_steps):

            coordinates = ref_domain.coordinates()
            vals = np.zeros((coordinates.shape[0],))
            inside = np.ones((coordinates.shape[0],))

            for i in range(coordinates.shape[0]):
                if state.bool_is_transient:
                    vals[i] = state.state[k](flightpath[k, :] + coordinates[i, :])
                else:
                    try:
                        vals[i] = state.state(flightpath[k, :] + coordinates[i, :])
                    except(RuntimeError):
                        inside[i] = 0

            # integrate over reference domain
            u = dl.Function(V)
            u.vector().vec().array = vals
            val = dl.assemble(u * dl.dx(domain=ref_domain))

            #  compute re-weighting
            if sum(inside) == inside.shape[0]:
                val_integral = val_integral_ref
            else:
                u = dl.Function(V)
                u.vector().vec().array = inside
                val_integral = dl.assemble(u * dl.dx(domain=ref_domain))

            data[k] = val / val_integral

        return data

    def d_measurement_d_control(self, alpha, flightpath, grid_t, state):
        """
        derivative of the measurement function for a given flightpath in control direction alpha

        @param alpha:
        @param flightpath:
        @param grid_t:
        @param state:
        @return: np.ndarray of shape (grid_t.shape[0], self.n_parameterss)
        """
        raise NotImplementedError("In MyDroneUniformEval.d_measurement_d_control: measurement derivatives not implemented yet")
