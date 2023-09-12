import numpy as np
import scipy.linalg as la

class Prior():

    def __init__(self, covar, mean):
        self.covar = covar
        self.mean = mean
        self.n_para = self.covar.shape[1]
        self.covar_sqrt = la.sqrtm(self.covar)

        self.covar_inv = la.inv(covar)
        # TODO: we should never ever compute this inverse. It's ok for now because it's low-dimensional and this is for
        #  testing, but really, we need to replace this

    def sample(self, n_samples=1):

        if n_samples == 1:
            size = (self.n_para,)
        else:
            size = (self.n_para, n_samples)

        return self.covar_sqrt @ np.random.normal(size=size)

