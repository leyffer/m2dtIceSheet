import numpy as np
import fenics as dl
import scipy.linalg as la

from MyDrone import MyDrone

class MyDroneGaussianEval(MyDrone):

    center = np.array([0.75/2, 0.55/2])

    def __init__(self, fom, grid_t=None, sigma=0.1, **kwargs):
        """! Initializer for the drone class with point-wise measurements
        @param fom  Full-order-model (FOM) object. The drone takes
        measurements from this
        @param grid_t the time grid the drone should fly in
        @param **kwargs  Keyword arguments including `sigma_gaussian`
        (Gaussian radius) and `radius_uniform`
        """
        super().__init__(fom=fom, eval_mode="gaussian", grid_t=grid_t, **kwargs)
        self.sigma = sigma

    def measure(self, flightpath, grid_t, state) -> np.ndarray:
        """! Get measurements along the flight path at the drone location

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        if state.bool_is_transient:
            raise NotImplementedError("In MyDroneGaussianEval: time-dependent convolution with Gaussian not implemented yet")

        # set diffusion coefficient to be the variance we want
        diffusion = self.sigma ** 2

        # choose the final time to bring the coefficient 1/(4*pi*D*t) into the form 1/(2*pi*sigma**2)
        t_final = 0.5

        # specify time discretization for heat equation
        dt = grid_t[1] - grid_t[0]
        n_steps = int(t_final / dt)
        # todo: evaluate how fine the time discretization actually needs to be

        # initialize variables
        u_old = state.state  # last time step
        # note: I've double-checked that this initialization will not cause state.state to be changed by the code below
        # (Nicole, Nov 21, 2023)

        for k in range(n_steps):
            u = dl.Function(self.fom.V)
            v = dl.TestFunction(self.fom.V)

            # Define variational form for the heat equation, implicit Euler
            F = dl.inner(u, v) * dl.dx \
                - dl.inner(u_old, v) * dl.dx \
                + dl.Constant(dt * diffusion) * dl.inner(dl.grad(u), dl.grad(v)) * dl.dx

            # todo: boundary conditions!

            # solve for next time step
            dl.solve(F == 0, u)

            # we don't actually need to remember the whole trajectory, only the final state
            u_old = u

        # initialization of measurement data array
        n_steps = flightpath.shape[0]
        data = np.NaN * np.ones((n_steps,))

        for k in range(n_steps):
            # evaluate convolution-state at postion p(t_k)
            data[k] = u_old(flightpath[k, :])

        return data

    def measure_without_truncation(self, flightpath, grid_t, state) -> np.ndarray:
        """! Get measurements along the flight path at the drone location

        @param flightpath  The trajectory of the drone
        @param grid_t  the time discretization on which the flightpath lives
        @param state  The state which the drone shall measure, State object
        """
        # initialization
        n_steps = flightpath.shape[0]
        data = np.NaN * np.ones((n_steps,))

        for k in range(n_steps):

            # define Gaussian with center around the current flight path position and std deviation self.sigma
            pos_x, pos_y = flightpath[k, :]  # current position of the drone
            weight = f'exp(-0.5 * ((x[0]-{pos_x})*((x[0]-{pos_x})) +' \
                     + f' (x[1]-{pos_y})*(x[1]-{pos_y})) / {self.sigma ** 2})'
            weight_fct = dl.Expression(weight, degree=0)

            # Re-weight such that the integral is = 1
            val_integral = dl.assemble(
                weight_fct * dl.dx(domain=self.fom.mesh))  # comment in for non-truncated gaussian
            # We would just divide by (np.pi*radius_uniform**2) here, but if the
            # mesh is not fine enough this will cause issues.
            # (We won't converge towards point evaluation even though that's our
            # theoretical limit since our FE solution is continuous)

            if state.bool_is_transient:
                # todo: this assumes state and drone operate on the same time discretization -> generalize
                val = dl.assemble(weight_fct * state.state[k] * dl.dx()) / val_integral
            else:
                val = dl.assemble(
                    weight_fct * state.state * dl.dx()) / val_integral  # comment in for non-truncated gaussian

            data[k] = val

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