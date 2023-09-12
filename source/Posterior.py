import numpy as np
import scipy.linalg as la

class Posterior():

    def __init__(self, brain):
        self.brain = brain
        self.prior = brain.prior
        self.noise_model = brain.noise_model
        self.n_para = self.prior.n_para

        self.flightpath = None
        self.covar = None
        self.covar_inv = None
        self.para2obs = None
        self.data = None
        self.mean = None

    def set_flightpath(self, flightpath):
        self.flightpath = flightpath

    def get_para2obs(self):
        if self.para2obs is not None:
            return self.para2obs

        parameter = np.eye(self.prior.n_para)
        G = np.empty((self.noise_model.K, self.prior.n_para))

        for i in range(self.prior.n_para):
            G[:, i], flightpath = self.brain.apply_para2obs(para=parameter[i, :], flightpath=self.flightpath,
                                                            bool_return_flightpath=True)

            if self.flightpath is None:
                self.flightpath = flightpath
            else:
                if not np.isclose(self.flightpath, flightpath).all():
                    raise RuntimeError("measurements were taken over a different flightpath than specified")

        self.para2obs = G
        return self.para2obs

    def compute_covariance(self):

        G = self.get_para2obs()
        yolo = self.noise_model.compute_noisenorm2(G)
        self.covar_inv = yolo + self.prior.covar_inv
        self.covar = la.inv(self.covar_inv)

    def compute_mean(self, data):

        G = self.get_para2obs()
        yolo = self.noise_model.apply_covar_inv(data)
        mean = G.T @ yolo + la.solve(self.prior.covar, self.prior.mean)
        mean = la.solve(self.covar_inv, mean)

        self.data = data
        self.mean = mean

        return mean





