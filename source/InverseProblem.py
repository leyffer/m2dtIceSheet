"""
Class for performing the Bayesian inverse problem

This class is used to get posterior mean and covariance given a prior and data.
"""

import warnings
from typing import Any, Optional

import numpy as np
import scipy.sparse as sparse
from Drone import Drone
from FullOrderModel import FullOrderModel as FOM
from State import State
from Noise import Noise

# FOM converts parameters to states
# Inverse Problem has a basis and keeps the states for that basis


class InverseProblem:
    """! InverseProblem class
    In this class we provide all functions needed for handling the inverse
    problem, starting from its setup to its solution. In particular, for the OED
    problem, we provide:

    - a call that applies the inverse posterior covariance matrix for given
      flight path parameters
    - a call to compute the posterior mean
    - the option to apply a reduction in parameter space (e.g., with active
      subspaces)

    Note: the details on the parameter space reduction are not clear yet

    By default, for consistency with the old code, we are using the
    **deterministic** inverse problem here where the noise is weighted by an
    arbitrary inner product matrix. The full Bayesian version where the noise
    model is consistent with time continuous measurements is slightly more
    involved. The computations for it are therefore outsourced into the subclass
    InverseProblemBayes.

    self.states are the computed responses to particular parameter bases, e.g.,
    a standard basis (each row of np.eye(self.n_parameters))
    """
    states = None
    Basis = None
    parameters = None

    def __init__(self, fom: FOM, drone: Drone, noise: Noise = None) -> None:
        """! Initialization for InverseProblem class instance

        @param fom: Forward model, also specifies the prior
        @param drone: specifies how the measurements are taken (for given flight parameters)
        """
        self.fom = fom
        self.drone = drone
        self.n_parameters = self.fom.n_parameters

        # set noise model:
        self.grid_t = drone.grid_t
        self.set_noise_model(noise=noise)

    def set_noise_model(
            self, c_scaling: float = 1.0, c_diffusion: float = 0.0, *args, c_boundary: float = 0.0, noise=None
    ):
        """! Noise model initialization (only needed if varying from defaults)

        Initialize the noise model. This is a convenience function to set the
        InverseProblem attributes `c_scaling`, `c_diffusion` (and `c_boundary`)
        and compute the diffusion and mass matrices

        @param c_scaling  Noise scaling parameter
        @param c_diffusion  Noise diffusion parameter
        @param c_boundary  boundary scaling (ignored in deterministic setting)
        """
        if noise is None:
            noise = Noise(grid_t=self.grid_t)
            noise.parameterize_noise_model(c_scaling=c_scaling, c_diffusion=c_diffusion, c_boundary=c_boundary)

        self.noise = noise

    def compute_noisenorm2(self, measurement_data):
        r"""! Computes the noise norm squared of `measurement data`, i.e., compute
        $$
        measurement_data^T \Sigma_{noise}^{-1} measurement_data
        $$

        @param measurement_data  measured values
        @return  noise norm squared, i.e., $\| <measurement_data> \|_{\Sigma_{noise}^{-1}}^2
        """
        return self.noise.compute_noisenorm2(measurement_data=measurement_data)

    def compute_L2norm2(self, measurement_data):
        return self.noise.compute_L2norm2(measurement_data=measurement_data)

    def apply_noise_covar_inv(self, measurement_data):
        """! Apply the inverse noise covariance matrix to the observations ` measurement_data`, i.e., compute
        $$
        \Sigma_{noise}^{-1} measurement_data
        $$

        @param measurement_data  measured values
        @return  the inverse noise covariance matrix applied to the observations d
        """
        return self.noise.apply_noise_covar_inv(measurement_data=measurement_data)

    def apply_para2obs(
        self,
        parameter: np.ndarray,
        state: Optional[State] = None,
        flight: Optional["Flight"] = None,
        **kwargs
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
                raise RuntimeError(
                    "In InverseProblem.apply_para2obs: state and parameter do not match"
                )

        # determine flight path
        if flight is None:
            flight = self.drone.navigation.create_flight(alpha=kwargs.get("alpha"))

        # fly out and measure
        observation = self.drone.measure(flight=flight, state=state)

        return observation, flight, state

    # TODO - cache posterior for optimization
    def compute_posterior(
        self,
        flight: Optional["Flight"] = None,
        alpha: Optional[np.ndarray] = None,
        grid_t: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
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
                raise RuntimeError(
                    "neither a flight nor valid control parameters were provided"
                )
            flight = self.drone.plan_flight(alpha=alpha, grid_t=grid_t)

        # initialize posterior
        posterior = Posterior(inversion=self, flight=flight, data=data)
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
        Computes the forward solutions for a given parameters (or reduced
        parameters for a given basis). Only returns the states, does not save
        anything.

        # TODO - Not clear why `precompute_states` and `compute_states` are separate.

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

    def get_states(self) -> np.ndarray:
        """
        returns self.states
        If self.states has not yet been set, they get computed assuming the standard basis
        @return:
        """

        if self.states is None:
            warnings.warn(
                "InverseProblem.get_states: No saved states. "
                "Computing and saving with standard basis."
            )

            # precompute states assuming standard basis
            self.compute_states(parameters=np.eye(self.n_parameters))

        return self.states
