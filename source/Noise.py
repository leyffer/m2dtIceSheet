import numpy as np
import scipy.sparse as sparse


class Noise():
    c_scaling: float = 1e3
    c_diffusion: float = 0.0

    def __init__(self, grid_t):
        self.grid_t = grid_t

        self.diffusion_matrix = self.compute_diffusion_matrix()
        self.mass_matrix = self.compute_mass_matrix()

    @property
    def n_time_steps(self):
        if isinstance(self.grid_t, np.ndarray):
            return self.grid_t.shape[0]
        raise ValueError("Time grid not present.")

    def parameterize_noise_model(
            self, c_scaling: float, c_diffusion: float, *args, c_boundary: float = 0.0
    ):
        """! Noise model initialization (only needed if varying from defaults)

        Initialize the noise model. This is a convenience function to set the
        InverseProblem attributes `c_scaling`, `c_diffusion` (and `c_boundary`)
        and compute the diffusion and mass matrices

        @param c_scaling  Noise scaling parameter
        @param c_diffusion  Noise diffusion parameter
        @param c_boundary  boundary scaling (ignored in deterministic setting)
        """
        # parameterization for the noise covariance operator
        self.c_scaling = c_scaling
        self.c_diffusion = c_diffusion

    def compute_diffusion_matrix(self) -> sparse.csr_matrix:
        """! Diffusion matrix

        Here, the diffusion matrix is similar to:
        [[ 1 -1  0  0
          -1  2 -1  0
           0 -1  2 -1
           0  0 -1  1]] * (1/delta_t)
        The middle rows are finite difference forms of -delta_t * (d^2)/(dt^2).
        The first and last rows are finite difference forms of -(d)/(dt) and
        (d)/(dt) respectively.

        @return  diffusion matrix (piece-wise linear finite elements)
        """
        delta_t = self.grid_t[1] - self.grid_t[0]
        # TODO: don't assume uniform timestepping

        A = sparse.diags(
            [-1, 2, -1],
            offsets=[-1, 0, 1],
            shape=(self.n_time_steps, self.n_time_steps),
        )
        # Convert from a diagonal matrix to s CSR matrix (to allow assignment of [0,0] and [-1,-1])
        A = sparse.csr_matrix(A)
        A[0, 0] = 1
        A[-1, -1] = 1
        A /= delta_t

        return A

    def compute_mass_matrix(self) -> sparse.csr_matrix:
        """! Mass matrix

        Here, the mass matrix looks like:
        [[ 4  1  0  0
           1  8  1  0
           0  1  8  1
           0  0  1  4]] * (delta_t/6)

        @return  mass matrix (piecewise linear finite elements)
        """
        delta_t = self.grid_t[1] - self.grid_t[0]
        # TODO: don't assume uniform timestepping

        M = sparse.diags(
            np.array([1, 4, 1]) / 6,
            offsets=[-1, 0, 1],
            shape=(self.n_time_steps, self.n_time_steps),
        )
        M = sparse.csr_matrix(M)
        M[0, 0] /= 2
        M[-1, -1] /= 2
        M.todense()
        M[0, 0] /= 2
        M[-1, -1] /= 2
        M *= delta_t

        return M

    def sample_noise(self, n_samples: int = 1) -> np.ndarray:
        """! Method for sampling

        @param n_samples  number of samples to draw
        @return  The samples
        """
        # TODO: sampling the noise model is just for show, it's not necessarily
        #  needed. However, it's a very nice show and helps visualizing the data
        #  a lot, so ... implement it!
        raise NotImplementedError(
            "InverseProblem.sample: still need to check how exactly we are setting up the noise model"
        )

    def compute_noisenorm2(self, measurement_data):
        r"""! Computes the noise norm squared of `measurement data`, i.e., compute
        $$
        measurement_data^T \Sigma_{noise}^{-1} measurement_data
        $$

        @param measurement_data  measured values
        @return  noise norm squared, i.e., $\| <measurement_data> \|_{\Sigma_{noise}^{-1}}^2
        """
        # todo: catch missing data (nan)
        yolo = self.apply_noise_covar_inv(measurement_data)
        return measurement_data.T @ yolo

    def compute_L2norm2(self, measurement_data):
        # todo: catch missing data (nan)
        return measurement_data.T @ (self.mass_matrix @ measurement_data)

    def apply_noise_covar_inv(self, measurement_data):
        """! Apply the inverse noise covariance matrix to the observations ` measurement_data`, i.e., compute
        $$
        \Sigma_{noise}^{-1} measurement_data
        $$

        @param measurement_data  measured values
        @return  the inverse noise covariance matrix applied to the observations d
        """
        # todo: catch missing data (nan)
        LHS = self.c_scaling * (
                self.c_diffusion * self.diffusion_matrix + self.mass_matrix
        )
        Kd = LHS @ measurement_data
        # TODO: still need to bring this parameterization together with the interpretation of the noise model
        return Kd
