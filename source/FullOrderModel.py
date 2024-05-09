import numpy as np
import scipy.linalg as la

from State import State


class FullOrderModel:
    """! Full-order-model (FOM)
    This is the general parent class for full-order models.
    For any particular model the user should create a subclass and specify the functions below.
    """

    n_parameters = -1
    prior_covar = None
    prior_mean = None
    covar_sqrt = None
    n_spatial = None  # spatial dimension (2D, 3D)

    def __init__(self):
        """!
        the main part of the initialization is entirely up to the user. They should overwrite the __init__ call with
        what they need. In case this call will at some point include some reference setup, the user can still call it
        with super().__init__()
        """
        # TODO: set the important variables
        pass

    # TODO: create those functions that we know we'll need, add explanations for their correct setup

    def solve(self, parameter: np.ndarray, *kwargs) -> State:
        """! Solve the transient problem
        @param parameter of interest
        @param kwargs should contain:
            "grid_t" for transient problems
        """
        raise NotImplementedError(
            "FullOrderModel.solve: Needs to be implemented in subclass"
        )

    def set_prior(self, prior_mean: np.ndarray, prior_covar: np.ndarray) -> None:
        """
        sets the prior for the parameter of interest. The size of the prior mean communicates the parameter dimension.
        The prior is assumed to be a Gaussian and thereby uniquely specified by its prior mean and covariance matrix.

        @param prior_mean: np.ndarray of shape (n_parameters,)
        @param prior_covar: np.ndarray of shape (n_parameters, n_parameters)
        @return: None
        """
        # the prior is a Gaussian, which is uniquely specified by its mean and covariance matrix
        self.prior_mean = prior_mean
        self.prior_covar = prior_covar
        self.n_parameters = prior_mean.shape[0]

        # sanity check:
        if prior_covar.shape != (self.n_parameters, self.n_parameters):
            raise RuntimeError(
                "In FullOrderModel.set_prior: invalid shape {} for prior covariance matrix (expected: {})".format(
                    prior_covar.shape, (self.n_parameters, self.n_parameters)
                )
            )

    def get_covar_sqrt(self):
        """
        computes the square root of the prior covariance matrix. For larger parameter spaces, this procedure is
        expensive and we don't need the square root anyway until we get to parameter reduction techniques, that's why we
        are not computing it by default
        """
        if self.covar_sqrt is None:
            self.covar_sqrt = la.sqrtm(self.prior_covar)
            # TODO: find a better (randomized?) way to compute this matrix for large-dimensional parameter spaces (or
            #  find a way to sample without it and get rid of it altogether)
        return self.covar_sqrt

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        draws a sample from the prior of size (n_parameters,) if only one sample to be drawn, and (n_parameters,n_samples) otherwise

        @param n_samples: the number of samples that shall be drawn. Determines the shape of the return
        @return: np.ndarray of size (n_parameters,) or (n_parameters,n_samples)
        """
        if self.n_parameters < 0:
            raise RuntimeError("In FullOrderModel.sample: Prior has not yet been set.")

        if n_samples == 1:
            size = (self.n_parameters,)
        else:
            size = (self.n_parameters, n_samples)

        covar_sqrt = self.get_covar_sqrt()
        samples = covar_sqrt @ np.random.normal(size=size)  # centered around 0

        return (samples.T + self.prior_mean).T
