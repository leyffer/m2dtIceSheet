import numpy as np
import fenics as dl
dl.set_log_level(30)
from Parameter import Parameter
import scipy.linalg as la

class Drone():

    def __init__(self, fom, eval_mode="gaussian", fly_mode="circle", grid_t=None, flying_parameters=None, **kwargs):

        self.fom = fom

        self.eval_mode = eval_mode
        self.fly_mode = fly_mode
        self.grid_t = grid_t if grid_t is not None else np.arange(0, 1+1e-2, 1e-2)

        self.sigma_gaussian = kwargs.get("sigma_gaussian", 0.1)
        self.radius_uniform = kwargs.get("radius_uniform", 0.1)

        self.flying_parameters = flying_parameters if flying_parameters is not None else self.get_default_flying_parameters()

    def get_default_flying_parameters(self):

        if self.fly_mode == "circle":
            flying_parameters = {
                "radius": 0.25,
                "center": np.array([0.75 / 2, 0.55 / 2]),
                "speed": 0.3
            }
            return flying_parameters

        raise RuntimeError("Invalid fly_mode={} provided".format(self.fly_mode))

    def set_default_flying_parameters(self, flying_parameters):
        self.flying_parameters = flying_parameters

    def get_trajectory(self, flying_parameters = None, grid_t=None):

        if flying_parameters is None:
            flying_parameters = self.flying_parameters

        center = flying_parameters["center"]
        radius = flying_parameters["radius"]
        speed = flying_parameters["speed"]

        if grid_t is None:
            grid_t = self.grid_t

        round_trip_time = 2 * np.pi * radius / speed
        angles = (grid_t * 2 * np.pi) / round_trip_time
        pos = radius * np.vstack([np.cos(angles), np.sin(angles)]).T
        pos = pos + center

        return pos, grid_t

    def get_position(self, t, flying_parameters=None):

        pos, __ = self.get_trajectory(flying_parameters=flying_parameters, grid_t=t*np.ones((1,)))
        return pos[0, :]

    def measure(self, flightpath, state, mode=None, **kwargs):

        mode = mode if mode is not None else self.eval_mode

        if mode == "point-eval":
            return self.measure_pointwise(flightpath, state)

        if mode == "gaussian":
            return self.measure_gaussian(flightpath, state)

        if mode == "uniform":
            return self.measure_uniform(flightpath, state)

        raise RuntimeError("invalid eval_mode={} encountered in Drone.measure".format(mode))

    def measure_pointwise(self, flightpath, state):
        # todo: this implementation assumes currently that a time-dependent state is on the same time-grid as the drone

        if isinstance(state, np.ndarray):
            return [state[k](flightpath[k, :]) for k in range(flightpath.shape[0])]

        return np.array([state(flightpath[k, :]) for k in range(flightpath.shape[0])])

    def measure_gaussian(self, flightpath, state):

        bool_transient = isinstance(state, np.ndarray)
        n_steps = flightpath.shape[0]
        data = np.NaN * np.ones((n_steps,))

        for k in range(n_steps):

            pos_x, pos_y = flightpath[k, :]

            # (unscaled) density function for normal distribution
            weight = 'exp(-0.5 * ((x[0]-{})*((x[0]-{})) + (x[1]-{})*(x[1]-{})) / {})'.format(pos_x, pos_x, pos_y, pos_y,
                                                                                             self.sigma_gaussian ** 2)

            # cut off after twice the standard deviation (truncated normal)
            weight_fct = dl.Expression('max({}, {})'.format(-np.exp(-2), weight),
                                       degree=1)  # ideally should use higher degree too

            # todo: I don't think the function is actually cut off :(

            # re-weight such that the integral is = 1
            val_integral = dl.assemble(weight_fct * dl.Measure('dx', self.fom.mesh))
            weight_fct = weight_fct / val_integral

            if bool_transient:
                val = dl.assemble(dl.inner(weight_fct, state[k]) * dl.dx)
            else:
                val = dl.assemble(dl.inner(weight_fct, state) * dl.dx)
            data[k] = val

        return data

    def measure_uniform(self, flightpath, state):

        bool_transient = isinstance(state, np.ndarray)
        n_steps = flightpath.shape[0]
        data = np.NaN * np.ones((n_steps,))

        for k in range(n_steps):

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

            # define indicator function over circular subdomain with center [pos_x, pos_y] and radius self.radius_uniform
            subdomain = Omega_circle(pos=np.array(flightpath[k, :]), radius=self.radius_uniform)
            material = dl.MeshFunction("size_t", self.fom.mesh, self.fom.mesh.topology().dim(), 0)
            subdomain.mark(material, 1)
            weight_fct = Parameter(material, np.array([0, 1]), degree=0)
            # todo: I'm not too versed in FEniCS, there might be much smarter ways to implement this. It works for now though

            # re-weight such that the integral is = 1
            val_integral = dl.assemble(weight_fct * dl.Measure('dx', self.fom.mesh))
            weight_fct = weight_fct / val_integral
            # we would just divide by (np.pi*radius_uniform**2) here, but if the mesh is not fine enough this will cause issues
            # (we won't converge towards point evaluation even though that's our theoretical limit since our FE solution is continuous)

            if bool_transient:
                val = dl.assemble(dl.inner(weight_fct, state[k]) * dl.dx)
            else:
                val = dl.assemble(dl.inner(weight_fct, state) * dl.dx)
            data[k] = val

        return data