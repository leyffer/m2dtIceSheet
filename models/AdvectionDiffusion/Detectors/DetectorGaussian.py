import sys

import numpy as np
import fenics as dl
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

sys.path.insert(0, "../source/")

from Detector import Detector
from myState import myState

class DetectorGaussian(Detector):
    """
    In this drone class, we model measurements at time t of a state u to be of the form:
    d(t) = 1/|c(t)| \int_{\Omega} u(x, t) Phi(x, p(t)) dx
    where
    - $Phi(x, y) = \frac{1}{\sqrt( (2 \pi \sigma^2 )^n ) \exp( - \| x-y \|^2 / (2 \sigma^2) )}$ is a multivariate
    Gaussian with covariance matrix sigma * I (sigma is given input parameter), and
    - c(t) = \int_{\Omega} Phi(x, p(t)) dx for normalization

    Unfortunately, computing the convolution over the whole domain at each time step is very expensive. We use the
    "Weierstrass trick" to approximate the convolution (explained in compute_convolution below).
    Note that currently it is only an approximation because we are not setting the correct boundary conditions. These,
    in turn, would again require evaluating the convolution everywhere.
    """
    
    eval_mode = "gaussian"

    center = np.array([0.75/2, 0.55/2])

    def __init__(self, fom, sigma=0.1, **kwargs):
        """! Initializer for the drone class with point-wise measurements
        @param fom  Full-order-model (FOM) object. The drone takes
        measurements from this
        @param grid_t the time grid the drone should fly in
        @param **kwargs  Keyword arguments including `sigma_gaussian`
        (Gaussian radius) and `radius_uniform`
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.fom=fom

    def compute_convolution(self, state):  # -> myState:
        """
        In this function we approximate the convolution of the state $v$ with the multivariate Gaussian $Psi$,
        $Psi(x, y) = \frac{1}{\sqrt( (2 \pi \sigma^2 )^n ) \exp( - \| x-y \|^2 / (2 \sigma^2) )}$
        following the "Weierstrass-trick":

        For an infinitely large domain, we know from theory that
        $\Phi(x, t) = \frac{1}{\sqrt((4\pi kt)^n)} \exp( -\| x \|_2^2 / (4 k t) )$
        is the fundamental solution of the heat equation
        $\dot{u} = \Delta u$
        for initial condition u(x,0) = \delta_0(x). From linearity of the heat equation, we get that for initial condition
        $u(x,0)=g$, we'll have
        $u(x, t) = \int \Phi(x-y, t) g(y) dy$.

        In our case we want to compute
        $\int Psi(x-y) v(y) dy$
        where $v$ is our state, and $Psi$ the multivariate Gaussian defined above. We can obtain this convolution by
        solving the heat equation with t=0.5, $k=\sigma^2$.

        There is one caviat to this procedure, which is that Phi was the solution on the infinite-dimensional domain,
        but our state is typically defined on a bounded domain $\Omega$. The correct boundary condition to apply on
        a bounded domain would require evaluating the convolution that we want to get in the first place. At least,
        I didn't find a way around yet.

        In the code below we are applying 0-Neumann boundary conditions. That's not correct, so the state we obtain
        is going to be an approximation. However, as long as sigma is small enough and we are evaluating the convolution
        sufficiently far away from the boundary, then it shouldn't be too bad.

        # todo: quantify the error we are introducing through the wrong boundary condition

        @param state:
        @return:
        """
        # todo: the code below does not apply the correct boundary conditions. If anything, this is an approximation
        #  at best.

        # check if this convolution has already been computed before, use self.eval_mode as unique identifier
        convolution = state.get_convolution(key=self.eval_mode)
        if convolution is not None:
            # don't recompute
            return convolution

        if state.bool_is_transient:
            raise NotImplementedError("In MyDroneGaussianEval: time-dependent convolution with Gaussian not implemented yet")

        # set diffusion coefficient to be the variance we want
        diffusion = self.sigma ** 2

        # choose the final time to bring the coefficient 1/(4*pi*D*t) into the form 1/(2*pi*sigma**2)
        t_final = 0.5

        # specify time discretization for heat equation
        dt = 0.01
        n_steps = int(t_final / dt)
        # todo: evaluate how fine the time discretization actually needs to be

        # initialize variables
        u_old = state.state  # last time step
        u_old = u_old.vector().vec().array

        # precompute matrices for faster solve
        u = dl.TrialFunction(self.fom.V)
        v = dl.TestFunction(self.fom.V)
        w = dl.Function(self.fom.V)

        # variational form for the heat equation, implicit Euler
        F = dl.inner(u, v) * dl.dx \
            + dl.Constant(dt * diffusion) * dl.inner(dl.nabla_grad(u), dl.nabla_grad(v)) * dl.dx
        A = dl.assemble(F)
        A = dl.as_backend_type(A).mat()  # PETSc matrix
        A = sparse.csr_matrix(A.getValuesCSR()[::-1], shape=A.size)
        LU_solver = sparse.linalg.splu(sparse.csc_matrix(A))

        # mass matrix
        m = dl.inner(u, v) * dl.dx
        M = dl.assemble(m)
        M = dl.as_backend_type(M).mat()  # PETSc matrix
        M = sparse.csr_matrix(M.getValuesCSR()[::-1], shape=M.size)

        # Neumann boundary condition
        # Note from Nicole (Feb 13, 2024):
        # I've commented out the code for the Neumann boundary condition below because it is not correct. Specifically,
        # in the definition of phi below, we need to set the convolution of the initial condition with the Gaussian.
        # but that's exactly what we want to compute! Unless we find a good way of approximating the boundary condition,
        # I don't think that's feasible.

        # n = dl.FacetNormal(self.fom.mesh)
        # y = self.fom.mesh.coordinates()
        # y = y[dl.dof_to_vertex_map(self.fom.V), :]
        #
        # g0 = dl.inner(dl.inner(u, n[0]), v) * dl.ds
        # g1 = dl.inner(dl.inner(u, n[1]), v) * dl.ds
        #
        # G0 = dl.assemble(g0)
        # G0 = dl.as_backend_type(G0).mat()  # PETSc matrix
        # G0 = sparse.csr_matrix(G0.getValuesCSR()[::-1], shape=G0.size)
        #
        # G1 = dl.assemble(g1)
        # G1 = dl.as_backend_type(G1).mat()  # PETSc matrix
        # G1 = sparse.csr_matrix(G1.getValuesCSR()[::-1], shape=G1.size)

        for k in range(n_steps):
            # the Neumann boundary conditions change in each time step

            # t = (k + 1) * dt
            # phi = np.exp(-((y[:, 0] - 0) * ((y[:, 0] - 0)) + (y[:, 1] - 0) * (y[:, 1] - 0)) / (4 * t * diffusion))
            # note: is there / np.sqrt(4*np.pi*sigma**2*t) missing here?
            # grad_0 = -phi * y[:, 0] / (2 * t * diffusion)
            # grad_1 = -phi * y[:, 1] / (2 * t * diffusion)
            # rhs = dt * G0 @ grad_0 + dt * G1 @ grad_1 + M @ u_old

            # compute rhs
            rhs = M @ u_old  # from implicit Euler time discretization

            # solve for next time step
            w = LU_solver.solve(rhs)

            # we don't actually need to remember the whole trajectory, only the final state
            u_old = w

        # bring back into FEniCS format
        w = dl.Function(self.fom.V)
        w.vector().vec().array = u_old

        # choose some identifiers to describe the computed object.
        # This is primarily to not get it confused with the original state object in the future
        my_identifiers = {
            "state": state,
            "history" : "computed by MyDroneGaussianEval.compute_convolution"
        }

        # bring into state format
        convolution_state = myState(fom=state.fom,
                                    state=w,
                                    bool_is_transient=False,
                                    parameter=state.parameter,
                                    other_identifiers=my_identifiers)
        # we are saving the convolution in myState-format such that we can easily access its derivative in the future
        # without repeat computations

        # save for future use
        state.set_convolution(convolution=convolution_state, key=self.eval_mode)

        return convolution_state

    def measure(self, flight, state) -> np.ndarray:
        """! Get measurements along the flight path at the drone location

        To get the measurements, we proceed the following way:
        1) we compute the convolution of the state with the Gaussian everywhere in the domain. This step is outsourced
        to compute_convolution. We save the convolution as a field, so we never need to compute it again.
        2) We take point-wise evaluations of this convolution field.

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        flightpath = flight.flightpath
        grid_t = flight.grid_t
        
        # compute convolution with gaussian
        convolution = self.compute_convolution(state=state)
        convolution = convolution.state

        # initialization of measurement data array
        n_steps = flightpath.shape[0]
        data = np.NaN * np.ones((n_steps,))

        for k in range(n_steps):
            # evaluate convolution-state at position p(t_k)
            data[k] = convolution(flightpath[k, :])

        return data

    def d_measurement_d_position(self, flight, state, navigation):
        """
        derivative of the measurement function for a given flightpath in position.
        The derivative is the gradient of the (convoluted) state along the flightpath. After the convolution field is computed (and saved),
        we can use FEniCS to get the derivative.

        @param alpha:
        @param flightpath:
        @param grid_t:
        @param state:
        @return: np.ndarray of shape (grid_t.shape[0], self.n_parameters)
        """
        flightpath, grid_t = flight.flightpath, flight.grid_t
        
        # compute convolution with gaussian
        convolution = self.compute_convolution(state=state)
        Du = convolution.get_derivative()
        
        # initialization
        D_data_d_position = np.empty((grid_t.shape[0], 2))  # (time, (dx,dy))

        for i in range(grid_t.shape[0]):
            # the FEniCS evaluation of the Du at a position unfortunately doesn't work with multiple positions
            # that's why we can't get rid of this loop

            # apply chain rule
            if state.bool_is_transient:
                # todo: extend to transient measurements
                raise NotImplementedError(
                    "In MyDronePointEval.d_measurement_d_position: still need to bring over code for transient measurements")
            else:
                # state is time-independent
                D_data_d_position[i, :] = Du(flightpath[i, :])

        return D_data_d_position

    def d_measurement_d_control(self, flight, state, navigation):
        """
        derivative of the measurement function for a given flightpath in control direction alpha.
        The derivative can easily be computed with the chain rule. After the convolution field is computed (and saved),
        we can use FEniCS to get the derivative.

        @param alpha:
        @param flightpath:
        @param grid_t:
        @param state:
        @return: np.ndarray of shape (grid_t.shape[0], self.n_parameters)
        """
        alpha, flightpath, grid_t = flight.alpha, flight.flightpath, flight.grid_t
        
        # compute convolution with gaussian
        convolution = self.compute_convolution(state=state)
        Du = convolution.get_derivative()

        # parts of the chain rule (only compute once)
        grad_p = flight.d_position_d_control()   # derivative of position
        # todo: optimize this computation such that we don't repeat it as often

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

    # def measure_without_truncation(self, flightpath, grid_t, state) -> np.ndarray:
    #     """! Get measurements along the flight path at the drone location

    #     This function is for testing purposes to check the implementation of the boundary condition is self.measure.
    #     Because of the integration in FE dimension, it takes a very long time to run, it should not be used in the final
    #     product.

    #     @param flightpath  The trajectory of the drone
    #     @param grid_t  the time discretization on which the flightpath lives
    #     @param state  The state which the drone shall measure, State object
    #     """
    #     # initialization
    #     n_steps = flightpath.shape[0]
    #     data = np.NaN * np.ones((n_steps,))

    #     for k in range(n_steps):

    #         # define Gaussian with center around the current flight path position and std deviation self.sigma
    #         pos_x, pos_y = flightpath[k, :]  # current position of the drone
    #         weight = f'exp(-0.5 * ((x[0]-{pos_x})*((x[0]-{pos_x})) +' \
    #                  + f' (x[1]-{pos_y})*(x[1]-{pos_y})) / {self.sigma ** 2})'
    #         weight_fct = dl.Expression(weight, degree=0)

    #         # Re-weight such that the integral is = 1
    #         val_integral = dl.assemble(
    #             weight_fct * dl.dx(domain=self.fom.mesh))  # comment in for non-truncated gaussian
    #         # We would just divide by (np.pi*radius_uniform**2) here, but if the
    #         # mesh is not fine enough this will cause issues.
    #         # (We won't converge towards point evaluation even though that's our
    #         # theoretical limit since our FE solution is continuous)

    #         if state.bool_is_transient:
    #             # todo: this assumes state and drone operate on the same time discretization -> generalize
    #             val = dl.assemble(weight_fct * state.state[k] * dl.dx()) / val_integral
    #         else:
    #             val = dl.assemble(
    #                 weight_fct * state.state * dl.dx()) / val_integral  # comment in for non-truncated gaussian

    #         data[k] = val

    #     return data

