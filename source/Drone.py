"""! Drone that can take measurements along its flightpath"""
try:
    import fenics as dl
    dl.set_log_level(30)
    using_firedrake = False
except ImportError:
    import firedrake as dl
    using_firedrake = True

import numpy as np
import scipy.linalg as la
if not using_firedrake:
    from Parameter import Parameter

from FOM_advectiondiffusion import FOM_advectiondiffusion


class Drone():
    """! Drone class that can take measurements along its flightpath"""
    def __init__(
        self,
        fom:FOM_advectiondiffusion,
        eval_mode:str = "gaussian",
        fly_mode:str = "circle",
        grid_t:np.ndarray[float] = None,
        flying_parameters:dict = None,
        **kwargs):
        """! Initializer for the drone class
            @param fom  Full-order-model (FOM) object. The drone takes
            measurements from this
            @param eval_mode  Evaluation mode of the drone's measurements:

                - `"gaussian"`: The drone takes a measurement that is dispersed
                  over a 2D gaussian
                - `"uniform"`: The drone takes a measurement that is dispersed
                  uniformly over a circle
                - `"point-eval"`: The drone takes a measurement at its exact
                  location

            @param fly_mode  The flightpath type the drone will fly in.
            Currently only `"circle"` is available
            @param grid_t  The time grid that the fom lives on (if transient).
            Used to determine the time step that the drone measures
            @param flying_parameters  The parameters of the flight path and
            measurement
            @param **kwargs  Keyword arguments including `sigma_gaussian`
            (Gaussian radius) and `radius_uniform`
        """

        self.fom = fom

        self.eval_mode = eval_mode
        self.fly_mode = fly_mode
        self.grid_t = grid_t if grid_t is not None else np.arange(0, 1, 1e-2)

        self.sigma_gaussian = kwargs.get("sigma_gaussian", 0.1)
        self.radius_uniform = kwargs.get("radius_uniform", 0.1)

        self.flying_parameters = flying_parameters if flying_parameters is not None else self.get_default_flying_parameters()

    def get_default_flying_parameters(self) -> dict:
        """! Default flying parameters of radius, center and speed for flight mode `"circle"`"""
        if self.fly_mode == "circle":
            flying_parameters = {
                "radius": 0.25,
                "center": np.array([0.75 / 2, 0.55 / 2]),
                "speed": 0.3
            }
            return flying_parameters

        raise RuntimeError(f"Invalid fly_mode={self.fly_mode} provided")

    def set_default_flying_parameters(self, flying_parameters:dict):
        """! Change flying parameters to the supplied parameters"""
        self.flying_parameters = flying_parameters

    def get_trajectory(
        self,
        flying_parameters:dict = None,
        grid_t:np.ndarray =None,
        ) -> tuple[np.ndarray, np.ndarray]:
        """! Get the trajectory of the drone given the flight parameters
            @param flying_parameters  The specified flight parameters
            @param grid_t  The time grid for the transient solution
            @return  Tuple of (position over flight path, corresponding time for
              each position)
        """
        if flying_parameters is None:
            flying_parameters = self.flying_parameters

        if self.fly_mode == "circle":
            center = flying_parameters["center"]
            radius = flying_parameters["radius"]
            speed = flying_parameters["speed"]

            if grid_t is None:
                grid_t = self.grid_t
            round_trip_time = 2 * np.pi * radius / speed
            angles = (grid_t * 2 * np.pi) / round_trip_time
            pos = radius * np.vstack([np.cos(angles), np.sin(angles)]).T
            pos = pos + center
        # TODO: piecewise constant (per time interval) controls for the
        # drone (finite number of controls)
        # TODO: continuous controls for the drone (infinite number of controls)
        # (constraints: fixed amount of flight time or fuel; limits on control total variation; return to base)

        return pos, grid_t

    def measure(
        self,
        flightpath:np.ndarray[float],
        state:np.ndarray[dl.Function],
        mode:str=None,
        **kwargs
        ) -> np.ndarray[float]:
        """! Method to take a measurement

            @param flightpath  The trajectory of the drone
            @param state  The states of the transient solution or the single
            state of the steady state solution
            @param mode  The method of measurement:

                - `"point-eval"`
                - `"gaussian"`
                - `"uniform"`
        """
        mode = mode if mode is not None else self.eval_mode

        if mode == "point-eval":
            return self.measure_pointwise(flightpath, state)

        if mode == "gaussian":
            return self.measure_gaussian(flightpath, state)

        if mode == "uniform":
            return self.measure_uniform(flightpath, state)

        raise RuntimeError(f"invalid eval_mode={mode} encountered in Drone.measure")

    def measure_pointwise(
        self,
        flightpath:np.ndarray[float],
        state:np.ndarray[dl.Function],
        ) -> np.ndarray[float]:
        """! Get measurements along the flight path at the drone location

            @param flightpath  The trajectory of the drone
            @param state  The states of the transient solution or the single
            state of the steady state solution
        """
        # TODO: this implementation assumes currently that a time-dependent state is on the same time-grid as the drone

        if isinstance(state, np.ndarray):
            # TODO: For some reason, the flightpath needs to be recreated as an array with firedrake
            return [state[k](np.array(flightpath[k, :])) for k in range(flightpath.shape[0])]

        return np.array([state(np.array(flightpath[k, :])) for k in range(flightpath.shape[0])])

    def measure_gaussian(
        self,
        flightpath:np.ndarray[float],
        state=np.ndarray[dl.Function]
        ) -> np.ndarray[float]:
        """! Get measurements along the flight path from a Gaussian centered at
        the drone location

            @param flightpath  The trajectory of the drone

            @param state  The states of the transient solution or the single
            state of the steady state solution
        """
        bool_transient = isinstance(state, np.ndarray)
        n_steps = flightpath.shape[0]
        data = np.NaN * np.ones((n_steps,))

        for k in range(n_steps):

            pos_x, pos_y = flightpath[k, :]

            # Cut off after twice the standard deviation (truncated normal),
            # gaussian simplifies (ignoring constant factor) to:
            # exp(-0.5 * (2 sigma / sigma)**2) = exp(-2)
            if not using_firedrake:
                # (unscaled) density function for normal distribution
                weight = f'exp(-0.5 * ((x[0]-{pos_x})*((x[0]-{pos_x})) +'\
                    + f' (x[1]-{pos_y})*(x[1]-{pos_y})) / {self.sigma_gaussian ** 2}) - exp(-2)'

                # ideally should use higher degree too
                weight_fct = dl.Expression(f'max({0}, {weight})', degree=1)
            else:
                x, y = dl.SpatialCoordinate(self.fom.mesh)
                weight = dl.Function(self.fom.V)
                # Use a conditional instead of a max function
                weight_fct = weight.interpolate(
                    dl.conditional(
                        dl.gt(
                            dl.exp(-0.5 * (pow(x - pos_x, 2) + pow(y - pos_y, 2)) \
                                / (self.sigma_gaussian ** 2)) - dl.exp(-2),
                            0
                        ),
                        dl.exp(-0.5 * (pow(x - pos_x, 2) + pow(y - pos_y, 2)) \
                            / (self.sigma_gaussian ** 2)) - dl.exp(-2),
                        0
                    )
                )

            # re-weight such that the integral is = 1
            val_integral = dl.assemble(weight_fct * dl.Measure('dx', self.fom.mesh))

            if bool_transient:
                val = dl.assemble(dl.inner(weight_fct, state[k]) * dl.dx) / val_integral
            else:
                val = dl.assemble(dl.inner(weight_fct, state) * dl.dx) / val_integral
            data[k] = val

        return data

    def measure_uniform(
        self,
        flightpath:np.ndarray[float],
        state=np.ndarray[dl.Function]
        ) -> np.ndarray[float]:
        """! Get measurements along the flight path from a uniform circle
        centered at the drone location

            @param flightpath  The trajectory of the drone

            @param state  The states of the transient solution or the single
            state of the steady state solution
        """

        bool_transient = isinstance(state, np.ndarray)
        n_steps = flightpath.shape[0]
        data = np.NaN * np.ones((n_steps,))

        for k in range(n_steps):
            if not using_firedrake:
                # define subdomain
                class Omega_circle(dl.SubDomain):

                    def __init__(self, pos, radius):
                        super().__init__()
                        self.center = pos
                        self.radius = radius

                    def inside(self, x, on_boundary):
                        if la.norm(x - self.center) <= self.radius:
                            return True
                        return False

                # Define indicator function over circular subdomain with center
                # [pos_x, pos_y] and radius self.radius_uniform
                subdomain = Omega_circle(pos=np.array(flightpath[k, :]), radius=self.radius_uniform)
                material = dl.MeshFunction("size_t", self.fom.mesh, self.fom.mesh.topology().dim(), 0)
                subdomain.mark(material, 1)
                weight_fct = Parameter(material, np.array([0, 1]), degree=0)
                # TODO: I'm not too versed in FEniCS, there might be much
                # smarter ways to implement this. It works for now though

            else:
                pos_x, pos_y = flightpath[k, :]
                x, y = dl.SpatialCoordinate(self.fom.mesh)
                weight = dl.Function(self.fom.V)
                # Use an ugly version of x/abs(x) to get a step function
                weight_fct = weight.interpolate(
                    0.5 + 0.5 * (self.radius_uniform ** 2 - (pow(x - pos_x, 2) + pow(y - pos_y, 2))) \
                        / abs(self.radius_uniform ** 2 - (pow(x - pos_x, 2) + pow(y - pos_y, 2)))
                )

            # Re-weight such that the integral is = 1
            val_integral = dl.assemble(weight_fct * dl.Measure('dx', self.fom.mesh))

            # We would just divide by (np.pi*radius_uniform**2) here, but if the
            # mesh is not fine enough this will cause issues.
            # (We won't converge towards point evaluation even though that's our
            # theoretical limit since our FE solution is continuous)

            if bool_transient:
                val = dl.assemble(dl.inner(weight_fct, state[k]) * dl.dx) / val_integral
            else:
                val = dl.assemble(dl.inner(weight_fct, state) * dl.dx) / val_integral
            data[k] = val

        return data