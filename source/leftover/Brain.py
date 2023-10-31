import numpy as np
import scipy.linalg as la

from Posterior import Posterior

class Brain():

    def __init__(self, fom, drone, **kwargs):

        self.fom = fom
        self.drone = drone

        self.prior = kwargs.get("prior", None)
        self.noise_model = kwargs.get("noise_model", None)

    def set_prior(self, prior):
        self.prior = prior

    def set_noise_model(self, noise_model):
        self.noise_model = noise_model

    def apply_para2obs(self, para, grid_t=None, state=None, flightpath=None, **kwargs):

        # should we catch the case where grid_t is None here and set a default?
        # I think it makes more sense to default back to the grid_t that self.fom uses, and, if the FOM is steady-state,
        # default to that of the drone. Eventually we don't want to use the same time grid for drone and transient state
        # anyway.

        # TODO: make compatible with multiple parameters to be passed at once in a matrix

        # solve for state
        if state is None:
            state, grid_t_fom = self.fom.solve(parameter=para, grid_t=grid_t)
        else:
            grid_t_fom = kwargs.get("grid_t_fom")

        grid_t = grid_t_fom
        # TODO: once we allow different time grids for drone and FOM, we need to remove grid_t=grid_t_fom here

        # determine flight path
        if flightpath is None:
            flightpath, grid_t_drone = self.drone.get_trajectory(grid_t=grid_t)
        else:
            grid_t_drone = kwargs.get("grid_t_drone")

        # fly out and measure
        observation = self.drone.measure(flightpath, state, mode=kwargs.get("mode", None), grid_t_fom=grid_t_fom, grid_t_drone=grid_t_drone)
        # TODO: eventually we'll make the statistical analysis increasingly dependent on how the measurements are taken
        #  in which case allowing here to pass a measurement mode to the drone is prone to introducing errors. For
        #  testing now I think it's helpful (avoid juggling too many classes), but eventually we'll just want to use the
        #  drone's default: the drone gets equipped with its measurement divices before takeoff.

        if kwargs.get("bool_return_flightpath", False):
            return observation, flightpath

        return observation

    def compute_posterior(self, data=None):
        posterior = Posterior(brain=self)
        posterior.compute_covariance()
        if data is not None:
            posterior.compute_mean(data)
        return posterior

    def eval_utility(self, posterior, oed_mode="D"):

        if oed_mode == "A":
            return self.eval_utility_A(posterior)

        if oed_mode == "D":
            return self.eval_utility_D(posterior)

        if oed_mode == "E":
            return self.eval_utility_E(posterior)

        raise RuntimeError("Invalid oed_mode encountered: {}".format(oed_mode))

    def eval_utility_A(self, posterior):
        return np.trace(posterior.covar)

    def eval_utility_D(self, posterior):
        return la.det(posterior.covar)

    def eval_utility_E(self, posterior):
        covar = posterior.covar
        eigvals = la.eigh(covar, eigvals_only=True)
        return np.max(eigvals)