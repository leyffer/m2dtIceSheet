"""
Bayesian inverse problem with Neumann boundary conditions on the noise model
"""
import warnings
from typing import Any, List

import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from Drone import Drone
from FullOrderModel import FullOrderModel as FOM
from InverseProblem import InverseProblem

# FOM converts parameters to states
# Inverse Problem has a basis and keeps the states for that basis


class InverseProblemBayesNeumann(InverseProblem):
    """! InverseProblem class
    In this class we provide all functions needed for handling the inverse
    problem, starting from its setup to its solution. In particular, for the OED
    problem, we provide:

    - a call that applies the inverse posterior covariance matrix for given
      flight path parameters
    - a call to compute the posterior mean
    - the option to apply a reduction in parameter space (e.g., with active
      subspaces)

    Note: the details on the last part are not clear yet

    In this notebook we specifically consider a noise model that is consistent
    with time-continuous measurements. The inverse problem is then consistent
    with the Bayesian setting even in the time-continuous limit.
    """

    c_scaling = 1e3
    c_diffusion = 0.01

    states = None
    Basis = None
    parameters = None
    reformat_matrix = None
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

    def set_noise_model(
        self,
        c_scaling: float = 1.0,
        c_diffusion: float = 1.0,
        c_boundary: float = 1.0,
        **kwargs
    ):
        """! Noise model initialization (only needed if varying from defaults) # Thomas: What are the defaults?

        @param grid_t  Time grid on which the measurements are taken
        @param c_scaling  Noise scaling parameter
        @param c_diffusion  Noise diffusion parameter
        """
        # parameterization for the noise covariance operator
        self.c_scaling = c_scaling
        self.c_diffusion = c_diffusion

        # matrices involved in the noise model
        diffusion_matrix = self.compute_diffusion_matrix()
        mass_matrix = self.compute_mass_matrix()

        n_steps = self.grid_t.shape[0]
        K = sparse.lil_matrix((n_steps + 2, n_steps + 2))
        K[1:-1, 1:-1] = c_scaling * (c_diffusion * diffusion_matrix + mass_matrix)
        K[0, 0] = 1
        K[1, 0] = c_scaling * c_diffusion
        K[-1, -1] = 1
        K[-2, -1] = -c_scaling * c_diffusion

        M = sparse.lil_matrix((n_steps + 2, n_steps + 2))
        M[1:-1, 1:-1] = mass_matrix
        M[0, 0] = c_boundary**2
        M[-1, -1] = c_boundary**2

        # dt_target = 1e-3
        # k_target = np.argmin(np.abs(self.grid_t-dt_target))
        # k_target = np.maximum(k_target, 1)
        k_target = 1
        dt = self.grid_t[k_target]
        reformat = sparse.lil_matrix((n_steps + 2, n_steps))
        reformat[1:-1, :] = sparse.eye(n_steps)
        reformat[0, 0] = -1 / dt
        reformat[0, k_target] = 1 / dt
        reformat[-1, -1 - k_target] = -1 / dt
        reformat[-1, -1] = 1 / dt

        self.laplacian_matrix = K
        self.mass_matrix = sparse.csc_matrix(M)
        self.reformat_matrix = reformat
        # self.mass_matrix_LU = sla.splu(self.mass_matrix, diag_pivot_thresh=0)

    def sample_noise(self, n_samples: int = 1) -> np.ndarray:
        """! Method for sampling

        *note:* code for Cholesky decomposition is adapted from
        https://gist.github.com/omitakahiro/c49e5168d04438c5b20c921b928f1f5d

        @param n_samples  number of samples to draw
        @return  The samples
        """
        if self.mass_matrix is None:
            warnings.warn(
                "InverseProblemBayesNeumann.sample_noise: Mass matrix was not found."
                + " Initializing the noise model with default parameters"
            )
            self.set_noise_model()
        n_steps = self.mass_matrix.shape[0]

        if self.mass_matrix_Chol is None:
            warnings.warn(
                "InverseProblemBayesNeumann.sample_noise: Computing dense square root"
            )
            self.mass_matrix_Chol = la.sqrtm(self.mass_matrix.toarray())
            # todo: replace with sparse cholesky decomposition

        samples = np.random.normal(size=(n_steps, n_samples))
        rhs = self.mass_matrix_Chol @ samples
        samples_with_bc = sla.spsolve(self.laplacian_matrix, rhs)
        if n_samples == 1:
            return samples_with_bc[1:-1]
        return samples_with_bc[1:-1, :]

    def apply_noise_covar_inv(self, measurement_data):
        r"""! Apply the inverse noise covariance matrix to the observations `
        measurement_data`, i.e., compute
        $$
        \Sigma_{noise}^{-1} measurement_data
        $$

        @param measurement_data  measured values
        @return  the inverse noise covariance matrix applied to the observations d
        """
        data_with_BC = self.reformat_matrix @ measurement_data
        Kd = self.laplacian_matrix @ data_with_BC
        # weighted_Kd = self.mass_matrix_LU.solve(Kd)
        weighted_Kd = sla.spsolve(self.mass_matrix, Kd)

        return self.reformat_matrix.T @ (self.laplacian_matrix.T @ weighted_Kd)
