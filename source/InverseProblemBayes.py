from typing import Optional, List, Any
# from typing import assert_type  # compatibility issues with Nicole's laptop (April 1, 2024)

import scipy.sparse as sparse
import numpy as np

from FullOrderModel import FullOrderModel as FOM
from Drone import Drone
from State import State
from InverseProblem import InverseProblem


# FOM converts parameters to states
# Inverse Problem has a basis and keeps the states for that basis


class InverseProblemBayes(InverseProblem):
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

    def set_noise_model(self, c_scaling, c_diffusion):
        """! Noise model initialization (only needed if varying from defaults)

        @param grid_t  Time grid on which the measurements are taken
        @param c_scaling  Noise scaling parameter
        @param c_diffusion  Noise diffusion parameter
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
