from typing import Optional, List, Any, assert_type

import scipy.sparse as sparse
import numpy as np

from FullOrderModel import FullOrderModel as FOM
from Drone import Drone
from State import State

# FOM converts parameters to states
# Inverse Problem has a basis and keeps the states for that basis


class InverseProblem:
    """! InverseProblem class
    In this class we provide all functions needed for handling the inverse problem, starting from its setup to its
    solution. In particular, for the OED problem, we provide:

    - a call that applies the inverse posterior covariance matrix for given flight path parameters
    - a call to compute the posterior mean
    - the option to apply a reduction in parameter space (e.g., with active subspaces)

    Note: the details on the last part are not clear yet
    """

    c_scaling = 1e3
    c_diffusion = 0.01

    states: Optional[np.ndarray[State, Any]] = None
    Basis = None
    parameters = None

    def __init__(self, fom: FOM, drone: Drone) -> None:
        """! Initialization for InverseProblem class instance

        @param fom: Forward model, also specifies the prior
        @param drone: specifies how the measurements are taken (for given flight parameters)
        """
        self.fom = fom
        self.drone = drone
        self.n_parameters = self.fom.n_parameters

        # set noise model:
        self.grid_t = drone.grid_t
        # self.diffusion_matrix = self.compute_diffusion_matrix()
        self.diffusion_matrix = None
        # self.mass_matrix = self.compute_mass_matrix()
        self.mass_matrix = None
        # TODO: right now it's somewhat unclear what the noise model actually is - we are just using the mass matrix.
        #  double check in the literature what we should be using exactly

    # TODO: write other functions required for this class
    # TODO: from the old source files, copy over all computations
    # TODO: set up connection to hIppylib
    
    @property
    def K(self):
        if isinstance(self.grid_t, np.ndarray):
            return self.grid_t.shape[0]
        raise ValueError("Time grid not present.")

    def set_noise_model(self, c_scaling, c_diffusion, grid_t):
        """! Noise model initialization (only needed if varying from default c_scaling=1, c_diffusion=1, grid_t=drone.grid_t)

        @param grid_t  Time grid on which the measurements are taken
        @param c_scaling  Noise scaling parameter
        @param c_diffusion  Noise diffusion parameter
        """
        # parameterization for the noise covariance operator
        self.c_scaling = c_scaling
        self.c_diffusion = c_diffusion
        self.grid_t = grid_t

        # matrices involved in the noise model
        self.diffusion_matrix = self.compute_diffusion_matrix()
        self.mass_matrix = self.compute_mass_matrix()

    def compute_diffusion_matrix(self) -> sparse.csr_matrix:
        """! Diffusion matrix

        @return  diffusion matrix (piece-wise linear finite elements)
        """
        delta_t = self.grid_t[1] - self.grid_t[0]
        # TODO: don't assume uniform timestepping
        # delta_ts = np.diff(grid_t)

        A = sparse.diags([1, -1], offsets=[0, 1], shape=(self.K, self.K))
        A = sparse.csr_matrix(A + A.T)
        A[0, 0] = 1
        A[-1, -1] = 1
        A /= delta_t

        return A

    def compute_mass_matrix(self) -> sparse.csr_matrix:
        """! Mass matrix

        @return  mass matrix (piecewise linear finite elements)
        """
        assert_type(self.grid_t, np.ndarray)
        delta_t = self.grid_t[1] - self.grid_t[0]
        # TODO: don't assume uniform timestepping
        # dts = np.diff(grid_t)

        M = sparse.diags(np.array([2, 1]) / 6, offsets=[0, 1], shape=(self.K, self.K))
        M = sparse.csr_matrix(M + M.T)
        M[0, 0] /= 2
        M[-1, -1] /= 2
        M *= delta_t

        return M

    def sample_noise(self, n_samples: int = 1) -> np.ndarray:
        """! Method for sampling

        @param n_samples  number of samples to draw
        @return  The samples
        """
        # TODO: sampling the noise model is just for show, it's not necessarily needed. However, it's a very nice show
        #  and helps visualizing the data a lot, so ... implement it!
        raise NotImplementedError(
            "InverseProblem.sample: still need to check how exactly we are setting up the noise model"
        )

    def compute_noisenorm2(self, d):
        """! Computes the noise norm squared

        @param d  measured values
        @return  noise norm squared
        """
        b = self.apply_noise_covar_inv(d)
        return d.T @ b

    def apply_noise_covar_inv(self, d):
        """! Apply the inverse noise covariance matrix to the observations d

        @param d  measured values
        @return  the inverse noise covariance matrix applied to the observations d
        """
        if self.diffusion_matrix is None:
            self.diffusion_matrix = self.compute_diffusion_matrix()
        if self.mass_matrix is None:
            self.mass_matrix = self.compute_mass_matrix()
        LHS = self.c_scaling * (
            self.c_diffusion * self.diffusion_matrix + self.mass_matrix
        )
        Kd = LHS @ d
        # TODO: still need to bring this parameterization together with the interpretation of the noise model
        return Kd

    def apply_para2obs(
        self, parameter, state: Optional[State] = None, flightpath: Optional[np.ndarray] = None, **kwargs
    ):
        """!
        Applies the parameter-to-observable map:
        1. computes the forward state for the given parameter
        2. computes the flightpath if none is specified
        3. takes the measurements along the flight path

        @param parameter:
        @param state:
        @param flightpath:
        @param kwargs:
        @return:
        """
        # solve for state
        if state is None:
            state = self.fom.solve(parameter=parameter)

        # determine flight path
        if flightpath is None:
            flightpath, grid_t_drone = self.drone.get_trajectory(
                grid_t=kwargs.get("alpha")
            )
        else:
            grid_t_drone = kwargs.get("grid_t_drone")

        # fly out and measure
        observation = self.drone.measure(
            flightpath=flightpath, grid_t=grid_t_drone, state=state
        )

        return observation, flightpath, grid_t_drone, state

    # TODO - cache posterior for optimization
    def compute_posterior(
        self,
        alpha: np.ndarray,
        data: Optional[np.ndarray] = None,
        flightpath: Optional[np.ndarray] = None,
        grid_t: Optional[np.ndarray] = None,
    ):
        """
        Computes the posterior distribution for given flight path parameters and measurements obtained along the flight.
        If no data is provided, the returned posterior will only contain the posterior covariance matrix, but not
        the posterior mean, provided the forward model is linear.

        @param alpha: flight path parameters
        @param data: measurement data (optional)
        @return: posterior
        """
        from Posterior import Posterior

        posterior = Posterior(
            inversion=self, alpha=alpha, data=data, flightpath=flightpath, grid_t=grid_t
        )
        return posterior

    def compute_states(
        self, parameters: np.ndarray, Basis: Optional[np.ndarray] = None
    ):
        """
        Computes the forward solutions for a given parameters (or reduced parameters for a given basis) and saves them
        in self.states

        @param parameters: np.ndarray such that the columns (or Basis applied to
            the columns) form the parameters at which to evaluate
        @param Basis: np.ndarray, parameter basis after parameter space reduction
        @return: None
        """
        self.parameters = parameters
        self.Basis = Basis
        self.states = self.precompute_states(parameters, Basis)

    def precompute_states(
        self, parameters: np.ndarray, Basis: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Computes the forward solutions for a given parameters (or reduced parameters for a given basis).
        Only returns the states, does not save anything.

        @param parameters: np.ndarray such that the columns (or Basis applied to
            the columns) form the parameters at which to evaluate
        @param Basis: np.ndarray, parameter basis after parameter space reduction
        @return: np.ndarray containing the states
        """
        # assemble parameters in full parameter space dimension if necessary
        if Basis is not None:
            parameters = Basis @ parameters

        # initialize
        states = np.zeros(self.n_parameters, dtype=object)

        # compute each state
        for i in range(self.n_parameters):
            # TODO: parallelize
            states[i] = self.fom.solve(parameter=parameters[:, i])

        return states

    def get_states(self):
        """
        returns self.states
        If self.states has not yet been set, they get computed assuming the unit basis
        @return:
        """

        if self.states is None:
            print(
                "WARNING:InverseProblem.get_states was called. This means the user didn't actively precompute the states."
            )
            print("this is ok, but it is error prone.")

            # precompute states assuming unit basis
            self.compute_states(parameters=np.eye(self.n_parameters))

        return self.states
