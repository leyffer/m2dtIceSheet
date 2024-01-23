import numpy as np
import fenics as dl
import scipy.linalg as la
import mshr

from MyDrone import MyDrone

class MyDroneTruncGaussianEval(MyDrone):

    center = np.array([0.75/2, 0.55/2])

    def __init__(self, fom, grid_t=None, sigma=0.1, radius=0.2, bool_truncate=True, **kwargs):
        """! Initializer for the drone class with point-wise measurements
        @param fom  Full-order-model (FOM) object. The drone takes
        measurements from this
        @param grid_t the time grid the drone should fly in
        @param **kwargs  Keyword arguments including `sigma_gaussian`
        (Gaussian radius) and `radius_uniform`
        """
        super().__init__(fom=fom, eval_mode="gaussian, truncated", grid_t=grid_t, **kwargs)
        self.sigma = sigma
        self.radius = radius

    def measure(self, flightpath, grid_t, state) -> np.ndarray:
        """! Get measurements along the flight path at the drone location

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        if not self.bool_truncate:
            return self.measure_without_truncation(flightpath, grid_t, state)

        # initialization
        n_steps = flightpath.shape[0]
        data = np.NaN * np.ones((n_steps,))

        # define subdomain
        ref_domain = mshr.generate_mesh(mshr.Circle(c=dl.Point(0.0, 0.0), r=np.min([self.radius, 4*self.sigma])), 50)
        V = dl.FunctionSpace(ref_domain, 'P', 1)
        weight = f'exp(-0.5 * ((x[0]-0)*((x[0]-0)) + (x[1]-0)*(x[1]-0)) / {self.sigma ** 2})'
        weight_fct = dl.Expression(weight, degree=1)
        val_integral_ref = dl.assemble(weight_fct * dl.dx(domain=ref_domain))

        for k in range(n_steps):

            coordinates = ref_domain.coordinates()
            vals = np.zeros((coordinates.shape[0],))
            inside = np.ones((coordinates.shape[0],))

            for i in range(coordinates.shape[0]):
                try:
                    # take measurement if inside the domain
                    if state.bool_is_transient:
                        vals[i] = state.state[k](flightpath[k, :] + coordinates[i, :])
                    else:
                        vals[i] = state.state(flightpath[k, :] + coordinates[i, :])
                except(RuntimeError):
                    # mark point as outside the domain
                    inside[i] = 0

            # integrate over reference domain
            u = dl.Function(V)
            u.vector().vec().array = vals
            val = dl.assemble(u * weight_fct * dl.dx(domain=ref_domain))

            #  compute re-weighting
            if sum(inside) == inside.shape[0]:
                # use reference integral value (save some compute time)
                val_integral = val_integral_ref
                # in this case, we know that the circle didn't overlap with the edgess of the domain
                # so we know that the integral is just the one we computed at the reference
            else:
                # the circle overlapped with the edges of the domain
                # for computing the weight function, we need to exclude those points
                u = dl.Function(V)
                u.vector().vec().array = inside
                val_integral = dl.assemble(u * weight_fct * dl.dx(domain=ref_domain))

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
        raise NotImplementedError("In MyDroneTruncGaussianEval.d_measurement_d_control: measurement derivatives not implemented yet")
