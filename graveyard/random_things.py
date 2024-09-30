def d_invPostCov_d_control_old(self):
    """
    computes the matrix derivative of the inverse posterior covariance
    matrix with respect to each control parameter (returned as a list). The
    matrix derivative is computed explicitly, which is likely inefficient
    and can be optimized out. In the long term, this function is therefore
    for testing purposes only, especially in case of high-dimensional
    parameter spaces.

    @return: list containing the matrix derivative of self.covar_inv w.r.t.
        each control parameter
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
        # TODO: if we only need the action of the matrix derivative, we
        #   should be able to optimize out this for-loop
        dG[:, i, :] = self.drone.d_measurement_d_control(
            flight=self.flight,
            state=self.inversion.states[i],
        )

    self.d_G_d_control = dG  # save for future use, e.g., testing
    # valid_positions = self.valid_positions

    # apply chain rule
    valid_positions = self.flight.valid_positions
    self.d_invCov_d_control = np.array(
        [
            dG[valid_positions, :, i].T @ self.invNoiseCovG[valid_positions, :] + self.invNoiseCovG[valid_positions,
                                                                                  :].T @ dG[valid_positions, :, i]
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
