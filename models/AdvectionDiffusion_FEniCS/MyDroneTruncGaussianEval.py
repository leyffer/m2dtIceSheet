import numpy as np
import fenics as dl
import scipy.linalg as la
import mshr

from MyDrone import MyDrone

class MyDroneTruncGaussianEval(MyDrone):
    """
    In this drone class, we model measurements at time t of a state u to be of the form:
    d(t) = 1/|c(t)| \int_{B(t)} u(x, t) Phi(x, p(t)) dx
    where
    - B(t) is a circle of a given radius (input parameter) around the position p(t) of the drone at time t,
    - $Phi(x, y) = \frac{1}{\sqrt( (2 \pi \sigma^2 )^n ) \exp( - \| x-y \|^2 / (2 \sigma^2) )}$ is a multivariate
    Gaussian with covariance matrix sigma * I (sigma is given input parameter), and
    - c(t) = \int_{B(t)} Phi(x, p(t)) dx for normalization
    """

    center = np.array([0.75/2, 0.55/2])

    def __init__(self, fom, grid_t=None, sigma=0.1, radius=0.2, bool_truncate=True, **kwargs):
        """! Initializer for the drone class with truncated Gaussian-type measurements

        @param fom  Full-order-model (FOM) object. The drone takes
        measurements from this
        @param grid_t the time grid the drone should fly in
        @param **kwargs  Keyword arguments including `sigma_gaussian`
        (Gaussian radius) and `radius_uniform`
        """
        super().__init__(fom=fom, eval_mode="gaussian, truncated", grid_t=grid_t, **kwargs)
        self.sigma = sigma
        self.radius = radius
        self.meshDim = kwargs.get("meshDim", 50)

    def measure(self, flightpath, grid_t, state) -> np.ndarray:
        """! Get measurements along the flight path at the drone location

        To compute the measurements on the subdomain B(t), we proceed the following way:
        1) We construct a reference mesh of a circular domain
        2) On the reference mesh, we can easily center our multivariate Gaussian.
        3) We use the shift of the circle's mesh point from the center to evaluate the state at the drone position
        + shift. This way, we project the values of the state in the circle around the drone position onto the reference
        circle.
        4) We use the reference mesh and the projected values to compute the convolution with the gaussian.

        Proceeding this way has the following advantages:
        - We don't need to track the circle on the whole domain Omega. In particular, for evaluating the integral,
        FEniCS doesn't need to iterate over all points in Omega to find the subdomain. It's a lot faster.
        - We can, and should, use a finer discretization than on the mesh for Omega. This way, the measurements are much
        smoother.
        - If there's an overlap with the edge of Omega, or if the circle extends beyond Omega, we can easily exclude
        this section from the circle.

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """

        # initialization
        n_steps = flightpath.shape[0]
        data = np.NaN * np.ones((n_steps,))

        # define subdomain
        ref_domain = mshr.generate_mesh(mshr.Circle(c=dl.Point(0.0, 0.0), r=np.min([self.radius, 4*self.sigma])), self.meshDim)
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

        derivative of the measurement function for a given flightpath in control direction alpha
        The derivative can be computed via the chain rule. For computing the derivative of the integral over the
        circle, we proceed similar as in self.measure, but project the spatial derivative of the state onto the
        reference mesh. Ideally, in the future, we do both within the one function to save on some iterations.

        Note:
        This function throws an error when the circle around the drone extends beyond the domain Omega. This is because
        in this case, the measurements are only evaluating the part of the circle within Omega, and the integral is
        also only normalized w.r.t. this subsection. For the derivative, we then need to also compute hte derivative of
        this section. It's doable, but very problem dependent (unless we approximate it numerically). Therefore, it's
        not implemented yet.

        # todo: already compute the derivative as a part of self.measure
        # todo: accept that the circle might extend beyond Omega and approximate the derivative in this case

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
        n_steps = flightpath.shape[0]
        D_data_d_alpha = np.empty((n_steps, alpha.shape[0]))

        # define subdomain
        ref_domain = mshr.generate_mesh(mshr.Circle(c=dl.Point(0.0, 0.0), r=np.min([self.radius, 4 * self.sigma])),
                                        self.meshDim)
        V = dl.FunctionSpace(ref_domain, 'P', 1)
        weight = f'exp(-0.5 * ((x[0]-0)*((x[0]-0)) + (x[1]-0)*(x[1]-0)) / {self.sigma ** 2})'
        weight_fct = dl.Expression(weight, degree=1)
        val_integral_ref = dl.assemble(weight_fct * dl.dx(domain=ref_domain))

        for k in range(n_steps):

            coordinates = ref_domain.coordinates()
            vals = np.zeros((coordinates.shape[0], alpha.shape[0]))
            inside = np.ones((coordinates.shape[0],))

            for i in range(coordinates.shape[0]):
                try:
                    # take measurement if inside the domain
                    if state.bool_is_transient:
                        vals[i, :] = Du[k](flightpath[k, :] + coordinates[i, :]) @ grad_p[:, :, k].T
                    else:
                        vals[i, :] = Du(flightpath[k, :] + coordinates[i, :]) @ grad_p[:, :, k].T
                except(RuntimeError):
                    # mark point as outside the domain
                    inside[i] = 0
                    # while not implemented, interrupt here already to know early
                    raise NotImplementedError(
                        "In MyDroneTruncGaussianEval.d_measurement_d_control: encountered overlap with edge of domain.")

            # integrate over reference domain
            val = np.zeros(alpha.shape[0])
            for i in range(alpha.shape[0]):
                u = dl.Function(V)
                u.vector().vec().array = vals[:, i]
                val[i] = dl.assemble(u * weight_fct * dl.dx(domain=ref_domain))
                # todo: this loop is very inefficient, there's likely a smarter way to integrate each element of a vector

            #  compute re-weighting
            if sum(inside) == inside.shape[0]:
                # use reference integral value (save some compute time)
                val_integral = val_integral_ref
                # in this case, we know that the circle didn't overlap with the edgess of the domain
                # so we know that the integral is just the one we computed at the reference
            else:
                # the circle overlapped with the edges of the domain
                # for computing the weight function, we need to exclude those points
                raise NotImplementedError(
                    "In MyDroneTruncGaussianEval.d_measurement_d_control: encountered overlap with edge of domain.")

            D_data_d_alpha[k, :] = val / val_integral

        return D_data_d_alpha
