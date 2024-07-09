import sys

import numpy as np
import fenics as dl
import scipy.linalg as la
import mshr

sys.path.insert(0, "../source/")

from Detector import Detector

class DetectorUniform(Detector):
    """
    In this drone class, we model measurements at time t of a state u to be of the form:
    d(t) = 1/|C(t)| \int_{C(t)} u(x, t) dx
    where C(t) is a circle of a given radius (input parameter) around the position of the drone at time t.
    """

    center = np.array([0.75/2, 0.55/2])

    def __init__(self, grid_t=None, radius=0.05, **kwargs):
        """! Initializer for the drone class with uniform measurements

        @param fom  Full-order-model (FOM) object. The drone takes
        measurements from this
        @param grid_t the time grid the drone should fly in
        @param **kwargs  Keyword arguments including `sigma_gaussian`
        (Gaussian radius) and `radius_uniform`
        """
        super().__init__(grid_t=grid_t, **kwargs)

        # define subdomain
        self.radius = radius
        self.meshDim = kwargs.get("meshDim", 10)
        self.ref_domain = mshr.generate_mesh(mshr.Circle(c=dl.Point(0.0, 0.0), r=self.radius), self.meshDim)
        self.V = dl.FunctionSpace(self.ref_domain, 'P', 1)
        self.val_integral_ref = dl.assemble(1 * dl.dx(domain=self.ref_domain))

        self.bool_remember_measurements = kwargs.get("bool_remember_measurements", True)

    def measure_at_position(self, pos, t, state):
        """! Get measurements along the flight path at the drone location

        To compute the measurements on the subdomain B(t), we proceed the following way:
        1) We construct a reference mesh of a circular domain
        2) We use the shift of the circle's mesh point from the center to evaluate the state at the drone position
        + shift. This way, we project the values of the state in the circle around the drone position onto the reference
        circle.
        3) We use the reference mesh and the projected values to compute the mean over the domain.

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

        coordinates = self.ref_domain.coordinates()
        vals = np.zeros((coordinates.shape[0],))
        inside = np.ones((coordinates.shape[0],))

        for i in range(coordinates.shape[0]):
            try:
                vals[i] = state.get_state(t=t, x=pos + coordinates[i, :])
            except(RuntimeError):
                inside[i] = 0

        # integrate over reference domain
        u = dl.Function(self.V)
        u.vector().vec().array = vals
        val = dl.assemble(u * dl.dx(domain=self.ref_domain))

        #  compute re-weighting
        if sum(inside) == inside.shape[0]:
            val_integral = self.val_integral_ref
        else:
            u = dl.Function(self.V)
            u.vector().vec().array = inside
            val_integral = dl.assemble(u * dl.dx(domain=self.ref_domain))

        return val / val_integral

    def d_measurement_d_position(self, flight, state):
        """
        TODO - update docstring

        derivative of the measurement function for a given flightpath in position
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
        
        alpha, flightpath, grid_t = flight.alpha, flight.flightpath, flight.grid_t
        n_spatial = flightpath.shape[1]
        
        # parts of the chain rule (only compute once)
        grad_p = flight.d_position_d_control  # derivative of position
        # todo: optimize this computation such that we don't repeat it as often
        # Du = state.get_derivative()  # spatial derivative of the state

        # initialization
        n_steps = flightpath.shape[0]
        D_data_d_position = np.empty((n_steps, 2))  # (time, (dx,dy))

        # define subdomain
        ref_domain = mshr.generate_mesh(mshr.Circle(c=dl.Point(0.0, 0.0), r=self.radius), self.meshDim)
        V = dl.FunctionSpace(ref_domain, 'P', 1)
        val_integral_ref = dl.assemble(1 * dl.dx(domain=ref_domain))

        for k in range(n_steps):

            coordinates = ref_domain.coordinates()
            vals = np.zeros((coordinates.shape[0], 2))  # (time, (dx,dy))
            inside = np.ones((coordinates.shape[0],))

            for i in range(coordinates.shape[0]):
                try:
                    vals[i, :] = state.get_derivative(t=grid_t[k], x=flightpath[k, :] + coordinates[i, :])
                except(RuntimeError):
                    inside[i] = 0
                    # while not implemented, interrupt here already to know early
                    raise NotImplementedError(
                        "In MyDroneUniformEval.d_measurement_d_position: encountered overlap with edge of domain.")

            # integrate over reference domain
            val = np.zeros((vals.shape[1],))
            for i in range(val.shape[0]):
                u = dl.Function(V)
                u.vector().vec().array = vals[:, i]
                val[i] = dl.assemble(u * dl.dx(domain=ref_domain))
                # todo: this loop is very inefficient, there's likely a smarter way to integrate each element of a vector

            #  compute re-weighting
            if sum(inside) == inside.shape[0]:
                val_integral = val_integral_ref
            else:
                raise NotImplementedError("In MyDroneUniformEval.d_measurement_d_position: encountered overlap with edge of domain.")

            D_data_d_position[k, :] = val / val_integral

        # bring into correct shape format
        D_data_d_position = np.hstack([np.diag(D_data_d_position[:, i]) for i in range(n_spatial)])
        return D_data_d_position
