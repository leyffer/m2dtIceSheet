import numpy as np
import scipy.sparse as sparse
import scipy.linalg as la
import scipy.sparse.linalg as sla

class NoiseModel():

    # note:
    # ideally we use FEniCS or Firedrake or whatever here to compute the mass and the diffusion matrix
    # However, I'm running into issues with FEniCS when I'm trying to create a mesh for an interval, so for now I've
    # just set the matrices up by hand. That's not ideal of course, especially since I didn't go through the trouble
    # to account for different time step sizes, but ideally we replace this code anyway.

    def __init__(self, grid_t, c_scaling=1, c_diffusion=1):

        self.c_scaling = c_scaling
        self.c_diffusion = c_diffusion
        self.grid_t = grid_t
        self.K = grid_t.shape[0]

        self.diffusion_matrix = self.compute_diffusion_matrix()
        self.mass_matrix = self.compute_mass_matrix()
        #self.mass_matrix_sqrt = self.compute_mass_sqrt()

    def compute_diffusion_matrix(self):
        dt = self.grid_t[1] - self.grid_t[0]
        # todo: don't assume uniform timestepping

        # diffusion matrix
        A1 = sparse.diags(-np.ones(self.K - 1), offsets=-1)
        diagonal = 2 * np.ones(self.K)
        diagonal[0] = 1
        diagonal[-1] = 1
        A2 = sparse.diags(diagonal, offsets=0)
        A = (A1 + A2 + A1.T) / dt
        A = sparse.csr_matrix(A)

        return A

    def compute_mass_matrix(self):
        dt = self.grid_t[1] - self.grid_t[0]
        # todo: don't assume uniform timestepping

        # mass matrix
        M1 = sparse.diags(np.ones(self.K - 1), offsets=-1)
        diagonal = 4 * np.ones(self.K)
        diagonal[0] = 2
        diagonal[-1] = 2
        M2 = sparse.diags(diagonal, offsets=0)
        M = dt * (M1 + M2 + M1.T) / 6
        M = sparse.csr_matrix(M)

        return M

    def compute_mass_sqrt(self):
        # todo: implement matrix square root for sparse matrices instead (this is a workaround for small matrices only)

        M = self.mass_matrix.todense()
        return la.sqrtm(M)

    def sample(self, n_samples=1):

        LHS = self.c_scaling * (self.c_diffusion * self.diffusion_matrix + self.mass_matrix)

        if n_samples != 1:
            Samples = np.empty((self.K, n_samples))

        for i in range(n_samples):
            # unfortunately we need the loop because the sparse solvers usually don't accept matrices as rhs

            rhs = self.mass_matrix_sqrt @ np.random.normal(size=(self.K,))
            sample = sla.spsolve(LHS, rhs)

            if n_samples == 1:
                return sample

            Samples[:, i] = sample

        return Samples

    def compute_noisenorm2(self, d):
        return d.T @ self.apply_covar_inv(d)

    def apply_covar_inv(self, d, d_left=None):

        LHS = self.c_scaling * (self.c_diffusion * self.diffusion_matrix + self.mass_matrix)
        Kd = LHS @ d

        # if len(Kd.shape) == 1:
        #     Minv_Kd = sla.spsolve(self.mass_matrix, Kd)
        # else:
        #     Minv_Kd = np.empty(Kd.shape)
        #     for i in range(Kd.shape[1]):
        #         Minv_Kd[:, i] = sla.spsolve(self.mass_matrix, Kd[:, i])
        #
        # yolo = LHS.T @ Minv_Kd
        # return yolo

        return Kd



