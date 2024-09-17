import sys

sys.path.insert(0, "..")

from Noise import Noise

import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import warnings


class NoiseBayesDirichlet(Noise):
    laplacian_matrix = None
    mass_matrix_Chol = None

    def parameterize_noise_model(
            self,
            c_scaling: float = 1.0,
            c_diffusion: float = 1.0,
            c_boundary: float = 1.0,
            **kwargs
    ):
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

        # TODO Thomas: I don't see why the noise model and sampling cannot exist
        separately from the InverseProblem class. The FOM and drone are only
        used to get delta_t and the number of timesteps. Consider moving noise
        functions to a separate class or function that handles this exclusively.

        @param n_samples  number of samples to draw
        @return  The samples
        """
        if self.mass_matrix is None:
            warnings.warn(
                "InverseProblemBayesDirichlet.sample_noise: Mass matrix was not found."
                + " Initializing the noise model with default parameters"
            )
            self.set_noise_model()
        n_steps = self.mass_matrix.shape[0]

        if self.mass_matrix_Chol is None:
            warnings.warn(
                "InverseProblemBayesDirichlet.sample_noise: Computing dense square root"
            )
            self.mass_matrix_Chol = la.sqrtm(self.mass_matrix.toarray())
            # todo: replace with sparse cholesky decomposition

        samples = np.random.normal(size=(n_steps, n_samples))
        rhs = self.mass_matrix_Chol @ samples
        samples_with_bc = sla.spsolve(self.laplacian_matrix, rhs)
        return samples_with_bc

    def apply_noise_covar_inv(self, measurement_data):
        r"""! Apply the inverse noise covariance matrix to the observations `
        measurement_data`, i.e., compute
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
