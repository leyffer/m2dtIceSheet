import warnings
from typing import Any, List, Optional

import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from Drone import Drone
from FullOrderModel import FullOrderModel as FOM
from InverseProblem import InverseProblem
from State import State

# from typing import assert_type  # compatibility issues with Nicole's laptop (April 1, 2024)




# FOM converts parameters to states
# Inverse Problem has a basis and keeps the states for that basis


class InverseProblemBayesDirichlet(InverseProblem):
    """! InverseProblem class
    In this class we provide all functions needed for handling the inverse problem, starting from its setup to its
    solution. In particular, for the OED problem, we provide:

    - a call that applies the inverse posterior covariance matrix for given flight path parameters
    - a call to compute the posterior mean
    - the option to apply a reduction in parameter space (e.g., with active subspaces)

    Note: the details on the last part are not clear yet

    In this notebook we specifically consider a noise model that is consistent with time-continuous measurements. The
    inverse problem is then consistent with the Bayesian setting even in the time-continuous limit.
    """

    c_scaling = 1e3
    c_diffusion = 0.01

    states = None
    Basis = None
    parameters = None
    laplacian_matrix = None
    mass_matrix_LU = None
    mass_matrix_Chol = None

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

    def set_noise_model(self, c_scaling, c_diffusion, c_boundary=1, *args):
        """! Noise model initialization (only needed if varying from defaults)

        @param c_scaling  Noise scaling parameter
        @param c_diffusion  Noise diffusion parameter
        """
        # parameterization for the noise covariance operator
        self.c_scaling = c_scaling
        self.c_diffusion = c_diffusion

        # matrices involved in the noise model
        diffusion_matrix = self.compute_diffusion_matrix()
        mass_matrix = self.compute_mass_matrix()

        K = c_scaling * (c_diffusion * diffusion_matrix + mass_matrix)
        K[0, 0] = 1
        K[0, 1] = 0
        K[-1, -1] = 1
        K[-1, -2] = 0

        M = mass_matrix
        M[0, :] = 0
        M[:, 0] = 0
        M[:, -1] = 0
        M[-1, :] = 0
        M[0, 0] += c_boundary ** 2
        M[-1, -1] += c_boundary ** 2

        self.laplacian_matrix = K
        self.mass_matrix = sparse.csc_matrix(M)

    def sample_noise(self, n_samples: int = 1) -> np.ndarray:
        """! Method for sampling

        *note:* code for Cholesky decomposition is adapted from
        https://gist.github.com/omitakahiro/c49e5168d04438c5b20c921b928f1f5d

        @param n_samples  number of samples to draw
        @return  The samples
        """
        n_steps = self.mass_matrix.shape[0]

        if self.mass_matrix_Chol is None:
            warnings.warn("InverseProblemBayesDirichlet.sample_noise: Computing dense square root")
            self.mass_matrix_Chol = la.sqrtm(self.mass_matrix.toarray())
            # todo: replace with sparse cholesky decomposition

        samples = np.random.normal(size=(n_steps, n_samples))
        rhs = self.mass_matrix_Chol @ samples
        samples_with_bc = sla.spsolve(self.laplacian_matrix, rhs)
        return samples_with_bc

    def apply_noise_covar_inv(self, measurement_data):
        """! Apply the inverse noise covariance matrix to the observations ` measurement_data`, i.e., compute
        $$
        \Sigma_{noise}^{-1} measurement_data
        $$

        @param measurement_data  measured values
        @return  the inverse noise covariance matrix applied to the observations d
        """
        Kd = self.laplacian_matrix @ measurement_data
        # weighted_Kd = self.mass_matrix_LU.solve(Kd)
        weighted_Kd = sla.spsolve(self.mass_matrix, Kd)

        return self.laplacian_matrix.T @ weighted_Kd
