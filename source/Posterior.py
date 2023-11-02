import numpy as np
import scipy.linalg as la


class Posterior():
    """ ! Posterior distribution (Gaussian)
    This class provides provides the posterior distribution for a given flight path parameterization and measurements.
    It is assumed that the forward model is linear, we have a Gaussian prior (specified in the FOM), and noise is
    additive and Gaussian. Therefore, the posterior distribution is again a Gaussian, uniquely specified by its
    mean and covariance matrix. Based on these assumptions, the covariance matrix is also independent from the
    measurement data. Therefore, when only the flight path is provided but not the measurement data, this class will
    still provide the posterior covariance matrix and a way to compute the action of its inverse.

    see Stuart 2010 (2.16, 2.17)
    """
    # todo: I don't think it makes sense to use the posterior for saving the state solutions for the basis vectors.
    #  I think that should go into the inverse problem class

    covar_inv = None  # inverse posterior covariance matrix
    para2obs = None
    eigvals = None

    def __init__(self, inversion, alpha, data, **kwargs):
        """
        initialization of the posterior distribution.

        @param inversion: the inverse problem setup
        @param alpha: flight path parameters
        @param data: measurement data along flight path (optional)
        @param kwargs: additional arguments as needed / known:
        grid_t: the time discretization for which we are measuring (defaults to drone.grid_t if not provided)
        flightpath: the flightpath for the provided alpha and time discretization (gets computed if not provided)
        states: the forward states computed for the parameter basis vectors (will get computed if not provided)
        """

        # general information about the modelling setup
        self.inversion = inversion  # inverse problem
        self.drone = inversion.drone  # how the measurements are taken
        self.prior = inversion.fom

        # information about the flight:
        self.grid_t = kwargs.get("grid_t", None) # check if the time discretization is known
        if self.grid_t is None:

            # flightpath has to be computed based on drone's default settings
            self.flightpath, self.grid_t = self.drone.get_trajectory(alpha=alpha)

        else:

            # check if flightpath is known already
            self.flightpath = kwargs.get("flightpath", None)

            if self.flightpath is None:

                # compute flightpath for the provided time discretization
                self.flightpath, yolo = self.drone.get_trajectory(alpha=alpha, grid_t=self.grid_t)

                # sanity check:
                if (self.grid_t != yolo).any():
                    raise RuntimeError("In Posterior.__init__: time discretization was changed")

        self.K = self.grid_t.shape[0]  # number of time steps

        # information about the posterior covariance
        self.para2obs = self.get_para2obs()

        # information about the posterior mean
        self.mean, self.data = self.compute_mean(data)  # compute the posterior mean

    def compute_mean(self, data):
        """
        computes the mean of the posterior distribution for given measurement data. self.mean and self.data do not get
        overwritten.

        Formulas for mean: Stuart 2010 (2.16, 2.17)

        @param data: measurements data for flight path self.alpha
        @return: posterior mean, provided data
        """

        if data is not None:

            # follow [Stuart, 2010], 2.17a
            G = self.get_para2obs()
            yolo = self.inversion.apply_noise_covar_inv(data)
            mean = G.T @ yolo + la.solve(self.prior.prior_covar, self.prior.prior_mean)

            covar_inv = self.compute_inverse_covariance()
            mean = la.solve(covar_inv, mean)
            # todo: replace with solve that uses the action of the inverse posterior covariance instead

        else:
            mean = None

        return mean, data

    def get_para2obs(self):
        """
        Computes the parameter-to-observable map, assuming a unit basis for the parameters. The parameter-to-observable
        map gets saved for future use, and won't be computed again if this function is called twice.

        @return:
        """
        # TODO: generalize to non-unit-vector basis for compatibility with parameter reduction setting

        # don't go through the effort of computing this twice
        if self.para2obs is not None:
            return self.para2obs

        # initialization
        parameter = np.eye(self.prior.n_parameters)  # basis for parameters
        G = np.empty((self.K, self.prior.n_parameters))
        states = self.inversion.get_states()

        # iterate over parameter basis
        for i in range(self.prior.n_parameters):

            # compute the measurements for the given parameter
            G[:, i], flightpath, __, __ = self.inversion.apply_para2obs(parameter=parameter[i, :],
                                                                flightpath=self.flightpath,
                                                                grid_t_drone=self.grid_t,
                                                                state=states[i])

            # todo: sanity check for flightpath

        # save for later use
        self.para2obs = G

        return self.para2obs

    def compute_inverse_covariance(self):
        """
        computes the inverse of the posterior covariance matrix. Since it needs the inverse prior covariance matrix,
        we really never want to compute it explicitely, only use its action. This function should - ideally - not be used ever.

        @return: inverse posterior covariance matrix
        """
        if self.covar_inv is None:

            G = self.get_para2obs()
            yolo = self.inversion.compute_noisenorm2(G)
            self.covar_inv = yolo + la.inv(self.prior.prior_covar)
            # todo: get rid of the call to la.inv !!!

        return self.covar_inv

    def apply_inverse_covariance(self, parameter):
        """
        applies the action of the inverse posterior covariance matrix to a parameter without explicitly computing the
        posterior covariance matrix or its inverse. This function should be prefered over any call to
        compute_inverse_covariance.

        See: Stuart 2010 (2.16, 2.17)

        @param parameter:
        @return:
        """
        G = self.get_para2obs()  # map parameter to its observations
        yolo = self.inversion.apply_noise_covar_inv(G @ parameter)  # apply inverse noise covariance matrix

        return G.T @ yolo + la.solve(self.prior.prior_covar, parameter)

    def get_eigenvalues(self):
        """
        computes and returns the eigenvalues of the posterior covariance matrix
        Currently the inverse posterior covariance matrix gets computed explicitely. We need to change this to use its
        action only

        @return: eigenvalues
        """
        if self.eigvals is None:

            covar_inv = self.compute_inverse_covariance()
            eigvals, eigvecs = la.eigh(covar_inv)
            # todo: switch to the action of the inverse covariance matrix instead
            self.eigvals = 1/eigvals

        return self.eigvals

