import numpy as np
import scipy.linalg as la
from functools import cached_property
from typing import Optional
from functools import lru_cache

from InverseProblem import InverseProblem


class Posterior:
    """! Posterior distribution (Gaussian)

    This class provides provides the posterior distribution for a given flight
    path parameterization and measurements. It is assumed that the forward model
    is linear, we have a Gaussian prior (specified in the FOM), and noise is
    additive and Gaussian. Therefore, the posterior distribution is again a
    Gaussian, uniquely specified by its mean and covariance matrix. Based on
    these assumptions, the covariance matrix is also independent from the
    measurement data. Therefore, when only the flight path is provided but not
    the measurement data, this class will still provide the posterior covariance
    matrix and a way to compute the action of its inverse.

    see Stuart 2010 (2.16, 2.17)
    """

    # TODO: I don't think it makes sense to use the posterior for saving the state solutions for the basis vectors.
    #  I think that should go into the inverse problem class

    covar: Optional[np.ndarray] = None  # posterior covariance matrix
    covar_inv: Optional[np.ndarray] = None  # inverse posterior covariance matrix

    # inverse noise covariance matrix applied to parameter-to-observable map G=para2obs
    invNoiseCovG: Optional[np.ndarray] = None

    # derivative of the inverse posterior covariance matrix w.r.t. the control
    d_invCov_d_control: Optional[np.ndarray] = None
    d_invCov_d_position: Optional[np.ndarray] = None
    # TODO: these variable names are HORRIBLE!

    d_G_d_control = None
    d_G_d_position = None

    def __init__(
        self, inversion: InverseProblem, flight: "Flight", data: np.ndarray, **kwargs
    ):
        """
        initialization of the posterior distribution.

        @param inversion: the inverse problem setup
        @param alpha: flight path parameters
        @param data: measurement data along flight path (optional)
        @param kwargs: additional arguments as needed / known:
            @param grid_t: the time discretization for which we are measuring
                (defaults to drone.grid_t if not provided)
            @param flightpath: the flightpath for the provided alpha and time
                discretization (gets computed if not provided)
            @param states: the forward states computed for the parameter basis
                vectors (will get computed if not provided)
        """

        # general information about the modelling setup
        self.inversion = inversion  # inverse problem
        self.drone = inversion.drone  # how the measurements are taken
        self.prior = inversion.fom
        self.n_parameters = self.prior.n_parameters
        self.n_spatial = self.inversion.fom.n_spatial

        # information about the flight:
        self.flight = flight
        self.alpha = flight.alpha
        self.n_controls = self.alpha.shape[0]
        self.grid_t = flight.grid_t
        self.n_timesteps = self.grid_t.shape[0]  # number of time steps

        # information about the posterior covariance
        # self.para2obs = self.para2obs

        # information about the posterior mean
        self.data = data
        self.mean = self.compute_mean(self.data)  # compute the posterior mean

    def compute_mean(self, data):
        """
        computes the mean of the posterior distribution for given measurement
        data. self.mean and self.data do not get overwritten.

        Formulas for mean: Stuart 2010 (2.16, 2.17)

        @param data: measurements data for flight path self.alpha
        @return: posterior mean
        """

        if data is not None:
            # follow [Stuart, 2010], 2.17a
            G = self.para2obs
            invNoiseCovarData = self.inversion.apply_noise_covar_inv(data)
            mean = G.T @ invNoiseCovarData + la.solve(
                self.prior.prior_covar, self.prior.prior_mean
            )

            covar_inv = self.compute_inverse_covariance()
            mean = la.solve(covar_inv, mean)
            # TODO: replace with solve that uses the action of the inverse posterior covariance instead

        else:
            # can't compute the posterior mean without data
            mean = None

        return mean

    @cached_property
    def para2obs(self):
        """
        Computes the parameter-to-observable map, assuming a unit basis for the
        parameters. The parameter-to-observable map gets saved for future use,
        and won't be computed again if this function is called twice.

        G is M_L without noise covariance

        @return:
        """
        # TODO: generalize to non-unit-vector basis for compatibility with parameter reduction setting

        # initialization
        parameter = np.eye(self.prior.n_parameters)  # basis for parameters

        # initialization for parameters to observable map
        # ith column stores measurements for ith basis function
        G = np.empty((self.n_timesteps, self.prior.n_parameters))

        states = self.inversion.get_states()  # basis states

        # iterate over parameter basis
        if states is not None:
            for i in range(self.prior.n_parameters):
                # compute the measurements for the given parameter
                (
                    G[:, i],  # ith column = ith basis
                    __,
                    __,
                ) = self.inversion.apply_para2obs(
                    parameter=parameter[i, :],
                    flight=self.flight,
                    state=states[i],
                )

            # TODO: sanity check for flightpath

        # save for later use
        return G

    def compute_inverse_covariance(self, inv_prior_factor: float = 1.0):
        """
        computes the inverse of the posterior covariance matrix. Since it needs
        the inverse prior covariance matrix, we really never want to compute it
        explicitly, only use its action. This function should - ideally - not be
        used ever.

        $(G^\\top \\Sigma_{\\mathrm{noise}}^{-1} G)$

        @return: inverse posterior covariance matrix
        """
        # TODO: I'm not entirely sure, but I think we don't actually ever have to compute the inverse posterior
        #  covariance matrix, its action should be sufficient. Right now it should be fine (make sure everything is
        #  correct) but in the long run we want to replace this call with the action of the inverse posterior
        #  covariance matrix.

        if self.covar_inv is None:
            # only compute once

            G = self.para2obs  # parameter-to-observable map

            # save for use in derivative computation
            self.invNoiseCovG = self.inversion.apply_noise_covar_inv(G)
            noise_observations = G.T @ self.invNoiseCovG  # squared noise norm
            # G^T Sigma_noise^{-1} G
            self.covar_inv = noise_observations + inv_prior_factor * la.inv(
                self.prior.prior_covar
            )
            # TODO: get rid of the call to la.inv!!!
            # Do this by substituting the prior covariance with the inverse
            # prior covariance where needed (start with an inverse prior instead
            # of a prior?)

        return self.covar_inv

    def compute_covariance(self, prior_factor: float = 1.0):
        """
        computes the posterior covariance matrix by inverting the inverse
        posterior covariance matrix from self.compute_inverse_covariance. This
        function is for testing purposes only, we should never compute
        explicitly compute these matrix inverses

        @return:
        """
        # TODO: optimize the code so that we can get rid of this function (if necessary, replace with action of the
        #  posterior covariance

        # TODO: the name is misleading, it should probably be a get command

        if self.covar is None:
            # only compute once
            self.covar = la.inv(
                self.compute_inverse_covariance(inv_prior_factor=1.0 / prior_factor)
            )

        return self.covar

    def apply_inverse_covariance(self, parameter):
        """
        applies the action of the inverse posterior covariance matrix to a
        parameter without explicitly computing the posterior covariance matrix
        or its inverse. This function should be preferred over any call to
        compute_inverse_covariance.

        See: Stuart 2010 (2.16, 2.17)

        G is the same as M_L
        \\Sigma_{post}^{-1} = G + \\Sigma_{pr}^{-1}
        Given b, find \\Sigma_{post}^{-1} b = x, i.e. solve \\Sigma_{post} x = b for x
        This is the same as solving for x:
        \\Sigma_{pr} x = (\\Sigma_{pr} G + I) b

        @param parameter:
        @return:
        """
        G = self.para2obs  # map parameter to its observations
        # apply inverse noise covariance matrix
        invNoiseCovG_parameter = self.inversion.apply_noise_covar_inv(G @ parameter)

        return G.T @ invNoiseCovG_parameter + la.solve(
            self.prior.prior_covar, parameter
        )

    @cached_property
    def _eigh(self):
        """
        computes and returns the eigenvalues of the posterior covariance matrix
        Currently the inverse posterior covariance matrix gets computed
        explicitly. We need to change this to use its action only

        @return: eigenvalues (ascending order), eigenvectors
        """
        # only compute once

        # get the inverse posterior covariance matrix
        covar_inv = self.compute_inverse_covariance()

        # solve eigenvalue problem
        # TODO: switch to the action of the inverse covariance matrix instead
        return la.eigh(covar_inv)

    @property
    def eigvals(self):
        """
        computes and returns the eigenvalues of the posterior covariance matrix
        Currently the inverse posterior covariance matrix gets computed
        explicitly. We need to change this to use its action only

        @return: eigenvalues (descending order)
        """
        # solve eigenvalue problem
        eigvals, _ = self._eigh
        # TODO: switch to the action of the inverse covariance matrix instead

        # convert to eigenvalues of the posterior covariance matrix (instead of its inverse)
        eigvals = 1 / eigvals

        return eigvals

    @property
    def eigvectors(self):
        """
        computes and returns the eigenvectors of the posterior covariance matrix
        Currently the inverse posterior covariance matrix gets computed
        explicitly. We need to change this to use its action only

        @return: eigenvectors (descending order for covariance; ascending for inverse covariance)
        """
        # solve eigenvalue problem
        _, eigvectors = self._eigh
        # TODO: switch to the action of the inverse covariance matrix instead

        return eigvectors

    def d_invPostCov_d_control(self):
        """
        computes the matrix derivative of the inverse posterior covariance
        matrix with respect to each control parameter (returned as a list). The
        matrix derivative is computed explicitly, which is likely inefficient
        and can be optimized out. In the long term, this function is therefore
        for testing purposes only, especially in case of high-dimensional
        parameter spaces.

        @return: list containing the matrix derivative of self.covar_inv w.r.t. each control parameter
        """
        # TODO: check overlap with d_invPostdoc_d_position

        if self.d_invCov_d_control is not None:
            # avoid re-computation
            return self.d_invCov_d_control

        # derivative of the parameter-to-observable map
        dG = np.empty(
            (self.n_timesteps, self.n_parameters, self.n_controls)
        )  # initialization

        for i in range(self.n_parameters):
            # TODO: if we only need the action of the matrix derivative, we should be able to optimize out this for-loop
            dG[:, i, :] = self.drone.d_measurement_d_control(
                flight=self.flight,
                state=self.inversion.states[i],
            )

        self.d_G_d_control = dG  # save for future use, e.g., testing

        # apply chain rule
        self.d_invCov_d_control = np.array(
            [
                dG[:, :, i].T @ self.invNoiseCovG + self.invNoiseCovG.T @ dG[:, :, i]
                for i in range(self.n_controls)
            ]
        )

        # TODO: this list is very inefficient. There's probably a smarter way using tensor multiplications

        # sanity check:
        if len(self.d_invCov_d_control[0].shape) == 0:
            # d_invPostCov_d_speed = np.array(np.array([d_invPostCov_d_speed]))

            # instead of casting into the correct format we raise an error, because at this point I expect the code
            # to be optimized enough that everything gets the correct shape just from being initialized correctly
            raise RuntimeError(
                "invalid shape = {} for d_invPostCov_d_speed".format(
                    self.d_invCov_d_control[0].shape
                )
            )

        return self.d_invCov_d_control

    def d_invPostCov_d_position(self):
        """
        computes the matrix derivative of the inverse posterior covariance
        matrix with respect to each control parameter (returned as a list). The
        matrix derivative is computed explicitly, which is likely inefficient
        and can be optimized out. In the long term, this function is therefore
        for testing purposes only, especially in case of high-dimensional
        parameter spaces.

        @return: list containing the matrix derivative of self.covar_inv w.r.t. each control parameter
        """
        # TODO: the dimensions don't work out here yet!

        if self.d_invCov_d_position is not None:
            # avoid re-computation
            return self.d_invCov_d_position

        dG = self.d_G_d_position
        if dG is None:
            # derivative of the parameter-to-observable map
            dG = np.empty(
                (self.n_timesteps, self.n_parameters, self.n_spatial * self.n_timesteps)
            )  # initialization

            for i in range(self.n_parameters):
                # TODO: if we only need the action of the matrix derivative, we should be able to optimize out this for-loop
                dG[:, i, :] = self.drone.d_measurement_d_position(
                    flight=self.flight,
                    state=self.inversion.states[i],
                )
            # save for future use, e.g., testing
            self.d_G_d_position = dG

        # apply chain rule
        self.d_invCov_d_position = np.array(
            [
                dG[:, :, i].T @ self.invNoiseCovG + self.invNoiseCovG.T @ dG[:, :, i]
                for i in range(self.n_spatial * self.n_timesteps)
            ]
        )

        # TODO: this list is very inefficient. There's probably a smarter way using tensor multiplications

        # sanity check:
        if len(self.d_invCov_d_position[0].shape) == 0:
            # d_invPostCov_d_speed = np.array(np.array([d_invPostCov_d_speed]))

            # instead of casting into the correct format we raise an error, because at this point I expect the code
            # to be optimized enough that everything gets the correct shape just from being initialized correctly
            raise RuntimeError(
                "invalid shape = {} for d_invPostCov_d_position".format(
                    self.d_invCov_d_position[0].shape
                )
            )

        return self.d_invCov_d_position

    def d_PostCov_d_control(self):
        """
        computes the matrix derivative of the posterior covariance matrix with
        respect to each control parameter (returned as a list). The matrix
        derivative is computed explicitly, which is likely inefficient and can
        be optimized out. In the long term, this function is therefore for
        testing purposes only, especially in case of high-dimensional parameter
        spaces.

        @return: list containing the matrix derivative of self.covar w.r.t. each
            control parameter
        """
        # get the posterior covariance matrix and its derivative
        PostCov = self.compute_covariance()
        covar_inv_derivative = self.d_invPostCov_d_control()

        # apply chain rule (matrix form) to get the derivative (be careful about the order!)
        return [
            -PostCov @ covar_inv_derivative[i] @ PostCov for i in range(self.n_controls)
        ]

    def d_PostCov_d_position(self):
        """
        computes the matrix derivative of the posterior covariance matrix with
        respect to the position at each time step (returned as a list). The matrix
        derivative is computed explicitly, which is likely inefficient and can
        be optimized out. In the long term, this function is therefore for
        testing purposes only, especially in case of high-dimensional parameter
        spaces.

        Note: Since the list has <number of time steps> many entries, computing all
        of these derivatives can be expensive. In the future we should look at
        optimizing these calls, or maybe some sort of approximation.

        @return: list containing the matrix derivative of self.covar w.r.t. each
            control parameter
        """
        # get the posterior covariance matrix and its derivative
        PostCov = self.compute_covariance()
        covar_inv_derivative = self.d_invPostCov_d_position()

        # apply chain rule (matrix form) to get the derivative (be careful about the order!)
        return [
            -PostCov @ covar_inv_derivative[i] @ PostCov
            for i in range(self.n_spatial * self.n_timesteps)
        ]
