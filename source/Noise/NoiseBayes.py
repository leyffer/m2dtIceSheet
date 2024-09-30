import sys

sys.path.insert(0, "..")

from Noise import Noise

import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import warnings


class NoiseBayes(Noise):
    laplacian_matrix = None
    mass_matrix_Chol = None
    laplacian_matrix_Chol = None

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

        self.laplacian_matrix = K
        self.mass_matrix = sparse.csc_matrix(mass_matrix)

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
        if self.mass_matrix_Chol is None:
            warnings.warn(
                "InverseProblemBayesDirichlet.sample_noise: Computing dense square root"
            )
            self.mass_matrix_Chol = la.sqrtm(self.mass_matrix.toarray())
            # todo: replace with sparse cholesky decomposition

        if self.laplacian_matrix_Chol is None:
            warnings.warn(
                "InverseProblemBayesDirichlet.sample_noise: Computing dense square root"
            )
            self.laplacian_matrix_Chol = la.sqrtm(self.laplacian_matrix.toarray())
            # todo: replace with sparse cholesky decomposition

        samples = np.random.normal(size=(self.n_time_steps, n_samples))
        rhs = self.mass_matrix_Chol @ samples
        samples = sla.spsolve(self.laplacian_matrix_Chol, rhs)
        return samples

    def apply_noise_covar_inv(self, measurement_data):
        r"""! Apply the inverse noise covariance matrix to the observations `
        measurement_data`, i.e., compute
        $$
        \Sigma_{noise}^{-1} measurement_data
        $$

        @param measurement_data  measured values
        @return  the inverse noise covariance matrix applied to the observations d
        """
        if len(measurement_data.shape) == 1:
            valid_positions = ~np.isnan(measurement_data)
            measurement_data = np.array([measurement_data]).T
            if measurement_data.shape[1] != 1:
                print("wring direction!")
        else:
            valid_positions = ~np.isnan(measurement_data[:, 0])
            # note: since each column of the measurement data is taken for the same flight, the nan entries
            #  are at the same position in each column

        if valid_positions.all():
            return self.laplacian_matrix @ measurement_data

        # remove entries
        covar_inv = self.restrict_matrix(large_matrix=self.laplacian_matrix.copy(), valid_positions=valid_positions)

        action = covar_inv @ measurement_data[valid_positions, :]
        expanded = np.nan * np.ones(measurement_data.shape)
        expanded[valid_positions, :] = action

        return expanded
