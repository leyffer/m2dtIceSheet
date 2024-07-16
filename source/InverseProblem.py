from typing import Optional, List, Any
# from typing import assert_type  # compatibility issues with Nicole's laptop (April 1, 2024)

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

    By default, for consistency with the old code, we are using the **deterministic** inverse problem here where the
    noise is weighted by an arbitrary inner product matrix. The full Bayesian version where the noise model is
    consistent with time continuous measurements is slightly more involved. The computations for it are therefore
    outsourced into the subclass InverseProblemBayes.
    """

    c_scaling = 1e3
    c_diffusion = 0.01

    states = None  # Optional[np.ndarray[State, Any]]
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
    # TODO: set up connection to hIppylib
    
    @property
    def n_time_steps(self):
        if isinstance(self.grid_t, np.ndarray):
            return self.grid_t.shape[0]
        raise ValueError("Time grid not present.")

    def set_noise_model(self, c_scaling, c_diffusion, c_boundary=0, *args):
        """! Noise model initialization (only needed if varying from defaults)

        @param c_scaling  Noise scaling parameter
        @param c_diffusion  Noise diffusion parameter
        @param c_boundary  boundary scaling (ignored in determistic setting)
        """
        # parameterization for the noise covariance operator
        self.c_scaling = c_scaling
        self.c_diffusion = c_diffusion

        # matrices involved in the noise model
        self.diffusion_matrix = self.compute_diffusion_matrix()
        self.mass_matrix = self.compute_mass_matrix()

    def compute_diffusion_matrix(self) -> sparse.csr_matrix:
        """! Diffusion matrix

        @return  diffusion matrix (piece-wise linear finite elements)
        """
        delta_t = self.grid_t[1] - self.grid_t[0]
        # TODO: don't assume uniform timestepping

        A = sparse.diags([1, -1], offsets=[0, 1], shape=(self.n_time_steps, self.n_time_steps))
        A = sparse.csr_matrix(A + A.T)
        A[0, 0] = 1
        A[-1, -1] = 1
        A /= delta_t

        return A

    def compute_mass_matrix(self) -> sparse.csr_matrix:
        """! Mass matrix

        @return  mass matrix (piecewise linear finite elements)
        """
        # assert_type(self.grid_t, np.ndarray)  # compatibility issues with Nicole's laptop (April 1, 2024)
        delta_t = self.grid_t[1] - self.grid_t[0]
        # TODO: don't assume uniform timestepping

        M = sparse.diags(np.array([2, 1]) / 6, offsets=[0, 1], shape=(self.n_time_steps, self.n_time_steps))
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

    def compute_noisenorm2(self, measurement_data):
        """! Computes the noise norm squared of `measurement data`, i.e., compute
        $$
        measurement_data^T \Sigma_{noise}^{-1} measurement_data
        $$

        @param measurement_data  measured values
        @return  noise norm squared, i.e., $\| <measurement_data> \|_{\Sigma_{noise}^{-1}}^2
        """
        yolo = self.apply_noise_covar_inv(measurement_data)
        return measurement_data.T @ yolo

    def compute_L2norm2(self, measurement_data):
        if self.mass_matrix is None:
            self.mass_matrix = self.compute_mass_matrix()

        return measurement_data.T @ (self.mass_matrix @ measurement_data)

    def apply_noise_covar_inv(self, measurement_data):
        """! Apply the inverse noise covariance matrix to the observations ` measurement_data`, i.e., compute
        $$
        \Sigma_{noise}^{-1} measurement_data
        $$

        @param measurement_data  measured values
        @return  the inverse noise covariance matrix applied to the observations d
        """
        if self.diffusion_matrix is None:
            self.diffusion_matrix = self.compute_diffusion_matrix()
        if self.mass_matrix is None:
            self.mass_matrix = self.compute_mass_matrix()
        LHS = self.c_scaling * (
            self.c_diffusion * self.diffusion_matrix + self.mass_matrix
        )
        Kd = LHS @ measurement_data
        # TODO: still need to bring this parameterization together with the interpretation of the noise model
        return Kd

    def apply_para2obs(
        self, parameter : np.ndarray, state: Optional[State] = None, flight: Optional["Flight"] = None, **kwargs
    ):
        """!
        Applies the parameter-to-observable map:
        1. computes the forward state for the given parameter
        2. computes the flightpath if none is specified
        3. takes the measurements along the flight path

        @param parameter:  The parameter for which we want to compute the observable
        @param state:  The (unique) state of the FOM for this parameter (to avoid re-computations if necessary) -- optional
        @param flight:  The flight along which to measure. If not yet available, pass the control parameter `alpha` as part of **kwargs
        @param kwargs:  should contain flight control parameter alpha if flight is not provided
        @return:
        - observation of state $u$ for parameter parameter along the trajectory of the flight $p$
        - the flight $p$ (for future use to avoid re-computations)
        - the state $u$ for the given parameter (for future use to avoid re-computations)
        """
        # solve for state
        if state is None:
            state = self.fom.solve(parameter=parameter)
        else:
            if (state.parameter != parameter).any():
                raise RuntimeError("In InverseProblem.apply_para2obs: state and parameter do not match")

        # determine flight path
        if flight is None:
            flight = self.drone.navigation.create_flight(alpha=kwargs.get("alpha"))

        # fly out and measure
        observation = self.drone.measure(
            flight=flight, state=state
        )

        return observation, flight, state

    # TODO - cache posterior for optimization
    def compute_posterior(
        self,
        flight : Optional["Flight"] = None,
        alpha : Optional[np.ndarray] = None,
        grid_t: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None
    ):
        """
        Computes the posterior distribution for a given flight and measurement data obtained along this flight.
        If no data is provided, the returned posterior will only contain the posterior covariance matrix, but not
        the posterior mean, as it assumes the forward model is linear.

        If no flight is provided, it gets computed from the control parameter alpha instead. Make sure that at least
        one of them is called, otherwise we throw a Runtime Error.

        @param alpha: flight path control. Will NOT be used if flight is provided
        @param flight: Flight for which we want to compute the posterior. Will be computed from alpha if not provided.
        @param data: measurement data (optional)
        @return: posterior
        """
        # import class for the posterior
        from Posterior import Posterior
        # TODO: the import is here instead of outside the class definition to avoid a circular import. It's not the best
        #  solution, we should revisit this for efficiency

        if flight is None:
            # if no flight is provided, we compute the one corresponding to the control parameters alpha
            if alpha is None:
                # if no control parameters are provided, something went wrong.
                raise RuntimeError("neither a flight nor valid control parameters were provided")
            flight = self.drone.plan_flight(alpha=alpha, grid_t=grid_t)

        # initialize posterior
        posterior = Posterior(
            inversion=self, flight=flight, data=data
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
            print("WARNING:InverseProblem.get_states was called. This means the user didn't actively precompute the "
                  "states. This is ok, but it is error prone. Pleae just call InverseProblem.precompute_states() "
                  "first in the future.")

            # precompute states assuming unit basis
            self.compute_states(parameters=np.eye(self.n_parameters))

        return self.states
