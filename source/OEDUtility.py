import numpy as np
import scipy.linalg as la
from typing import Optional

from Posterior import Posterior


class OEDUtility:
    """! OEDUtility class
    In this class we provide all information about the OED utility function (for A-, D-, and E-optimal experimental
    design). In particular, we provide:

    - evaluate the utility of a given flight path design (A-, D-, and E)
    - compute the eigenvalues of the posterior covariance matrix for a given flight path design
    - evaluate the gradient of the utility function
    """

    # TODO: decide if we should outsource the different OED modes into subclasses

    def __init__(self, default_mode: Optional[str] = "D") -> None:
        """! initialization for OEDUtility class

        @param inversion: inverse problem for which the OED utility shall be computed
        """

        self.default_mode = default_mode

    # TODO: specifiy all function call names that this class needs to have to interact with the other classes
    # TODO: from the old source files, copy over all computations

    def eval_utility(self, posterior: Posterior, mode: Optional[str] = None):
        """
        computes the OED-utility of the posterior covariance matrix for a provided posterior (assuming a linear model).
        Available OED-utility functions are: A, D, E, and "D-inverse" to compute the determinant of the inverse
        posterior covariance matrix.

        @param posterior: posterior distribution for which the utility shall be computed
        @param mode: string: "A", "D", "D-inverse", "E"
        @return:
        """
        # use default if no mode was provided
        if mode is None:
            mode = self.default_mode

        # switch between cases
        if mode == "A":
            return self.eval_utility_A(posterior)

        if mode == "D":
            return self.eval_utility_D(posterior)

        if mode == "D-inverse":
            return self.eval_utility_Dinv(posterior)

        if mode == "E":
            return self.eval_utility_E(posterior)

        raise RuntimeError("Invalid oed_mode encountered: {}".format(mode))

    def eval_utility_A(self, posterior: Posterior):
        """
        A-OED utility criterion: trace of the posterior covariance matrix.

        @param posterior:
        @return:
        """
        return sum(posterior.get_eigenvalues())

    def eval_utility_D(self, posterior: Posterior):
        """
        A-OED utility criterion: determinant of the posterior covariance matrix.

        @param posterior:
        @return:
        """
        # TODO: should we compute the inverse or the log instead? The values get very big
        return np.prod(posterior.get_eigenvalues())

    def eval_utility_Dinv(self, posterior: Posterior):
        """
        determinant of the inverse posterior covariance matrix.

        @param posterior:
        @return:
        """
        # TODO: I think we have 1/ twice now in all the computations, optimize it out
        return np.prod(1 / posterior.get_eigenvalues())

    def eval_utility_E(self, posterior: Posterior):
        """
        E-OED criterion: maximum eigenvalue of posterior covariance matrix
        @param posterior:
        @return:
        """
        return np.max(posterior.get_eigenvalues())

    def d_utility_d_control(self, posterior: Posterior, mode=None):
        """
        computes the derivative of the OED-utility function for the given posterior w.r.t. the control parameters.
        @param posterior:
        @param mode:
        @return: string: "A", "D", "D-inverse", "E"
        """

        # use default mode if none is provided
        mode = mode if mode is not None else self.default_mode

        # switch between cases
        if mode == "A":
            return self.d_utilA_d_control(posterior)

        if mode == "D":
            return self.d_utilD_d_control(posterior)

        if mode == "D-inverse":
            return self.d_utilDinv_d_control(posterior)

        if mode == "E":
            return self.d_utilE_d_control(posterior)

        raise RuntimeError("Invalid oed_mode encountered: {}".format(mode))

    def d_utilA_d_control(self, posterior: Posterior):
        """
        computes the derivative of the A-OED utility function
        Psi(X) = trace(X) where X is the posterior covariance matrix
        with respect to the control parameters of the flight path by applying the chain rule.

        The trace is a linear operator, so its derivative is itself:
        d Psi(X) / d alpha = Psi( d X / d alpha )

        see also: https://en.wikipedia.org/wiki/Matrix_calculus

        @param posterior:
        @return:
        """

        # get derivative information of the posterior covariance matrix
        der = posterior.d_PostCov_d_control()

        # compute gradient
        gradient = np.hstack([np.trace(der[i]) for i in range(posterior.n_controls)])

        return gradient

    def d_utilD_d_control(self, posterior: Posterior):
        """
        computes the derivative of the D-OED utility function
        Psi(X) = det(X) where X is the posterior covariance matrix
        with respect to the control parameters of the flight path by applying the chain rule.

        The derivative is:
        d Psi(X) / d alpha
        = trace( cofactor(X) (d X / d alpha) )
        [ = det(X) trace( X^{-1} (d X / d alpha ) ]

        The second equality only holds only if X is invertible. In our setting that's the case though, so we use the
        second formula in the computations below.

        see also: https://en.wikipedia.org/wiki/Matrix_calculus
        https://en.wikipedia.org/wiki/Minor_(linear_algebra)

        @param posterior:
        @return:
        """

        # get derivative information of the posterior covariance function
        der = posterior.d_PostCov_d_control()

        # only compute the determinant once
        det = la.det(posterior.compute_covariance())
        # TODO: only compute once, save as property within the posterior

        gradient = np.zeros((posterior.n_controls,))
        for i in range(posterior.n_controls):
            # sanity check
            if len(der[i].shape) == 1:
                raise RuntimeError(
                    "invalid shape = {} for derivative matrix".format(der[i].shape)
                )

            # apply transposed cofactor matrix
            yolo = det * la.solve(posterior.covar.T, der[i])
            # TODO: just apply inverse posterior covariance matrix (or its action)
            # TODO: the decomposition cofactor(M) = det(M)*inv(M) only holds for invertible matrices M
            # I don't think the posterior covariance matrix can become singular unless the prior or the noise covariance
            # matrices are degenerate. We might want to catch that case though.

            # finish computing the gradient
            gradient[i] = np.trace(yolo)

        return gradient

    def d_utilDinv_d_control(self, posterior: Posterior):
        """
        computes the derivative of the inverse D-OED utility function
        Psi(X) = det(X^{-1}) where X is the posterior covariance matrix
        with respect to the control parameters of the flight path by applying the chain rule.

        @param posterior:
        @return:
        """

        # get derivative information of the posterior covariance function
        der = posterior.d_invPostCov_d_control()

        # only compute the determinant once
        covar_inv = posterior.compute_inverse_covariance()
        det = la.det(covar_inv)
        # TODO: only compute once, save as property within the posterior

        gradient = np.zeros((posterior.n_controls,))
        for i in range(posterior.n_controls):
            # sanity check
            if len(der[i].shape) == 1:
                raise RuntimeError(
                    "invalid shape = {} for derivative matrix".format(der[i].shape)
                )

            # apply transposed cofactor matrix
            yolo = det * la.solve(covar_inv.T, der[i])
            # TODO: just apply inverse posterior covariance matrix (or its action)
            # TODO: the decomposition cofactor(M) = det(M)*inv(M) only holds for invertible matrices M
            # I don't think the posterior covariance matrix can become singular unless the prior or the noise covariance
            # matrices are degenerate. We might want to catch that case though.

            # finish computing the gradient
            gradient[i] = np.trace(yolo)

        return gradient

    def d_utilE_d_control(self, posterior: Posterior):
        raise NotImplementedError(
            "OEDUtility.d_utilE_d_control: still need to understand how to get the eigenvalue derivative"
        )
