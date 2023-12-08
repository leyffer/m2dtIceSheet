import numpy as np
import scipy.linalg as la


class Posterior:
    """! Posterior distribution (Gaussian)
    This class provides provides the posterior distribution for a given flight path parameterization and measurements.
    It is assumed that the forward model is linear, we have a Gaussian prior (specified in the FOM), and noise is
    additive and Gaussian. Therefore, the posterior distribution is again a Gaussian, uniquely specified by its
    mean and covariance matrix. Based on these assumptions, the covariance matrix is also independent from the
    measurement data. Therefore, when only the flight path is provided but not the measurement data, this class will
    still provide the posterior covariance matrix and a way to compute the action of its inverse.

    see Stuart 2010 (2.16, 2.17)
    """

    # TODO: I don't think it makes sense to use the posterior for saving the state solutions for the basis vectors.
    #  I think that should go into the inverse problem class

    covar = None  # posterior covariance matrix
    covar_inv = None  # inverse posterior covariance matrix
    invNoiseCovG = None  # inverse noise covariance matrix applied to parameter-to-observable map G=para2obs
    covar_inv_derivative = (
        None  # derivative of the inverse posterior covariance matrix w.r.t. the control
    )
    para2obs = None
    eigvals = None

    # TODO: these variable names are HORRIBLE!

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
        self.n_parameters = self.prior.n_parameters

        # information about the flight:
        self.alpha = alpha
        self.n_controls = alpha.shape[0]
        self.grid_t = kwargs.get(
            "grid_t", None
        )  # check if the time discretization is known

        if self.grid_t is None:
            # flightpath has to be computed based on drone's default settings
            self.flightpath, self.grid_t = self.drone.get_trajectory(alpha=alpha)

        else:
            # check if flightpath is known already
            self.flightpath = kwargs.get("flightpath", None)

            if self.flightpath is None:
                # compute flightpath for the provided time discretization
                self.flightpath, yolo = self.drone.get_trajectory(
                    alpha=alpha, grid_t=self.grid_t
                )

                # sanity check:
                if (self.grid_t != yolo).any():
                    raise RuntimeError(
                        "In Posterior.__init__: time discretization was changed"
                    )

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
            # TODO: replace with solve that uses the action of the inverse posterior covariance instead

        else:
            # can't compute the posterior mean without data
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
            G[:, i], flightpath, __, __ = self.inversion.apply_para2obs(
                parameter=parameter[i, :],
                flightpath=self.flightpath,
                grid_t_drone=self.grid_t,
                state=states[i],
            )

            # TODO: sanity check for flightpath

        # save for later use
        self.para2obs = G

        return self.para2obs

    def compute_inverse_covariance(self):
        """
        computes the inverse of the posterior covariance matrix. Since it needs the inverse prior covariance matrix,
        we really never want to compute it explicitely, only use its action. This function should - ideally - not be used ever.

        @return: inverse posterior covariance matrix
        """
        # TODO: I'm not entirely sure, but I think we don't actually every have to compute the inverse posterior
        #  covariance matrix, its action should be sufficient. Right now it should be fine (make sure everything is
        #  correct) but in the long run we want to replace this call with the action of the inverse posterior
        #  covariance matrix.

        if self.covar_inv is None:
            # only compute once

            G = self.get_para2obs()  # parameter-to-observable map
            self.invNoiseCovG = self.inversion.apply_noise_covar_inv(
                G
            )  # save for use in derivative computation
            yolo = G.T @ self.invNoiseCovG  # squared noise norm
            self.covar_inv = yolo + la.inv(self.prior.prior_covar)
            # TODO: get rid of the call to la.inv !!!

        return self.covar_inv

    def compute_covariance(self):
        """
        computes the posterior covariance matrix by inverting the inverse posterior covariance matrix from
        self.compute_inverse_covariance. This function is for testing purposes only, we should never compute explicitely
        compute these matrix inverses

        @return:
        """
        # TODO: optimize the code so that we can get rid of this function (if necessary, replace with action of the
        #  posterior covariance

        # TODO: the name is misleading, it should problaby be a get command

        if self.covar is None:
            # only compute once
            self.covar = la.inv(self.compute_inverse_covariance())

        return self.covar

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
        yolo = self.inversion.apply_noise_covar_inv(
            G @ parameter
        )  # apply inverse noise covariance matrix

        return G.T @ yolo + la.solve(self.prior.prior_covar, parameter)

    def get_eigenvalues(self):
        """
        computes and returns the eigenvalues of the posterior covariance matrix
        Currently the inverse posterior covariance matrix gets computed explicitely. We need to change this to use its
        action only

        @return: eigenvalues
        """
        if self.eigvals is None:
            # only compute once

            # get the inverse posterior covariance matrix
            covar_inv = self.compute_inverse_covariance()

            # solve eigenvalue probblem
            eigvals, eigvecs = la.eigh(covar_inv)
            # TODO: switch to the action of the inverse covariance matrix instead

            # convert to eigenvalues of the posterior covariance matrix (instead of its inverse)
            self.eigvals = 1 / eigvals

        return self.eigvals

    def d_invPostCov_d_control(self):
        """
        computes the matrix derivative of the inverse posterior covariance matrix with respect to each control parameter
        (returned as a list). The matrix derivative is computed explicitely, which is likely inefficient and can be
        optmized out. In the long term, this function is therefore for testing purposes only, especially in case of
        high-dimensional parameter spaces.

        @return: list containing the matrix derivative of self.covar_inv w.r.t. each control parameter
        """

        if self.covar_inv_derivative is not None:
            # avoid re-computation
            return self.covar_inv_derivative

        # derivative of the parameter-to-observable map
        dG = np.empty((self.K, self.n_parameters, self.n_controls))  # initialization

        for i in range(self.n_parameters):
            # TODO: if we only need the action of the matrix derivative, we should be able to optimize out this for-loop
            dG[:, i, :] = self.drone.d_measurement_d_control(
                alpha=self.alpha,
                flightpath=self.flightpath,
                grid_t=self.grid_t,
                state=self.inversion.states[i],
            )

        self.dG = dG  # save for future use, e.g., testing

        # apply chain rule
        self.covar_inv_derivative = [
            dG[:, :, i].T @ self.invNoiseCovG + self.invNoiseCovG.T @ dG[:, :, i]
            for i in range(self.n_controls)
        ]

        # TODO: this list is very inefficient. There's probably a smarter way using tensor multiplications

        # sanity check:
        if len(self.covar_inv_derivative[0].shape) == 0:
            # d_invPostCov_d_speed = np.array(np.array([d_invPostCov_d_speed]))

            # instead of casting into the correct format we raise an error, because at this point I expect the code
            # to be optimized enough that everything gets the correct shape just from being initialized correctly
            raise RuntimeError(
                "invalid shape = {} for d_invPostCov_d_speed".format(
                    self.covar_inv_derivative[0].shape
                )
            )

        return self.covar_inv_derivative

    def d_PostCov_d_control(self):
        """
        computes the matrix derivative of the posterior covariance matrix with respect to each control parameter
        (returned as a list). The matrix derivative is computed explicitely, which is likely inefficient and can be
        optmized out. In the long term, this function is therefore for testing purposes only, especially in case of
        high-dimensional parameter spaces.

        @return: list containing the matrix derivative of self.covar w.r.t. each control parameter
        """
        # get the posterior covariance matrix and its derivative
        PostCov = self.compute_covariance()
        covar_inv_derivative = self.d_invPostCov_d_control()

        # apply chain rule (matrix form) to get the derivative (be careful about the order!)
        return [
            -PostCov @ covar_inv_derivative[i] @ PostCov for i in range(self.n_controls)
        ]
