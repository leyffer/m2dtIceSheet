from InverseProblem import InverseProblem
import numpy as np
import scipy.linalg as la

class OEDUtility():
    """! OEDUtility class
    In this class we provide all information about the OED utility function (for A-, D-, and E-optimal experimental
    design). In particular, we provide:

    - evaluate the utility of a given flight path design (A-, D-, and E)
    - compute the eigenvalues of the posterior covariance matrix for a given flight path design
    - evaluate the gradient of the utility function
    """

    # todo: decide if we should outsource the different OED modes into subclasses

    def __init__(self, inversion : InverseProblem, default_mode : str = None) -> None:
        """! initialization for OEDUtility class

        @param inversion: inverse problem for which the OED utility shall be computed
        """

        self.fom = inversion.fom
        self.drone = inversion.drone
        self.inversion = inversion

        self.default_mode = default_mode if default_mode is not None else "D"

    # TODO: specifiy all function call names that this class needs to have to interact with the other classes
    # TODO: from the old source files, copy over all computations

    def eval_utility(self, posterior, mode=None):
        mode = mode if mode is not None else self.default_mode

        if mode == "A":
            return self.eval_utility_A(posterior)

        if mode == "D":
            return self.eval_utility_D(posterior)

        if mode == "E":
            return self.eval_utility_E(posterior)

        raise RuntimeError("Invalid oed_mode encountered: {}".format(mode))

    def eval_utility_A(self, posterior):
        return sum(posterior.get_eigenvalues())

    def eval_utility_D(self, posterior):
        # todo: should we compute the inverse or the log instead? The values get very big
        return np.prod(posterior.get_eigenvalues())

    def eval_utility_E(self, posterior):
        return np.max(posterior.get_eigenvalues())

    def d_utility_d_control(self, posterior, mode=None):
        mode = mode if mode is not None else self.default_mode

        if mode == "A":
            return self.d_utilA_d_control(posterior)

        if mode == "D":
            return self.d_utilD_d_control(posterior)

        if mode == "E":
            return self.d_utilE_d_control(posterior)

        raise RuntimeError("Invalid oed_mode encountered: {}".format(mode))

    def d_utilA_d_control(self, posterior):
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

    def d_utilD_d_control(self, posterior):
        # get derivative information of the posterior covariance function
        der = posterior.d_PostCov_d_control()

        # only compute the determinant once
        det = la.det(posterior.compute_covariance())
        # todo: only compute once, save as property within the posterior

        gradient = np.zeros((posterior.n_controls,))
        for i in range(posterior.n_controls):

            # sanity check
            if len(der[i]) == 1:
                raise RuntimeError("invalid shape = {} for derivative matrix".format(der[i].shape))

            # apply transposed cofactor matrix
            yolo = det * la.solve(posterior.covar.T, der[i])
            # todo: just apply inverse posterior covariance matrix (or its action)
            # todo: the decomposition cofactor(M) = det(M)*inv(M) only holds for invertible matrices M
            # I don't think the posterior covariance matrix can become singular unless the prior or the noise covariance
            # matrices are degenerate. We might want to catch that case though.

            # finish computing the gradient
            gradient[i] = np.trace(yolo)

        return gradient

    def d_utilE_d_control(self, posterior):
        raise NotImplementedError("OEDUtility.d_utilE_d_control: still need to understand how to get the eigenvalue derivative")
