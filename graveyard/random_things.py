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

    def d_position_d_position_and_control(self, flight: Flight) -> np.ndarray:
        """
        computes the derivative of the flightpath with respect to the its positions
        and the control parameters in alpha. If all positions are computed independently
        from each other, this is just going to be the zero matrix stacked next to
        the d_position_d_control output. However, in some cases the position at time $t_k$
        may be computed as an adjustment to the position at time $t_{k-1}$ (for example),
        in which case the derivative of position w.r.t. position is not the identity. These
        special cases need to be implemented in the subclass.

        @param flight: Flight object
        @return: gradient vector
        """
        alpha = flight.alpha
        initial_position, velocity, angular_velocity = self.split_controls(alpha)
        sol = flight.flightpath
        dt = self.grid_t[1] - self.grid_t[0]
        deriv = sparse.coo_matrix(([], ([], [])), shape=(3 * self.n_timesteps, 3 + 5 * self.n_timesteps))
        deriv = sparse.lil_matrix(deriv)
        splits_row = [0, self.n_timesteps, 2 * self.n_timesteps]
        splits_col = [0, self.n_timesteps, 2 * self.n_timesteps, 3 * self.n_timesteps,
                      3 + 3 * self.n_timesteps, 3 + 4 * self.n_timesteps]

        # initial position
        deriv[splits_row[0], splits_col[3]] = 1
        deriv[splits_row[1], splits_col[3] + 1] = 1
        deriv[splits_row[2], splits_col[3] + 2] = 1

        for i in range(1, self.n_timesteps):
            deriv[splits_row[0] + i, splits_row[0] + i - 1] = 1
            deriv[splits_row[1] + i, splits_row[1] + i - 1] = 1
            deriv[splits_row[2] + i, splits_row[2] + i - 1] = 1

        for i in range(1, self.n_timesteps):
            deriv[splits_row[2] + i, splits_col[5] + i - 1] = dt

        for i in range(1, self.n_timesteps):
            deriv[splits_row[0] + i, splits_col[4] + i - 1] = dt * np.cos(sol[i - 1, 2])

        for i in range(1, self.n_timesteps):
            deriv[splits_row[1] + i, splits_col[4] + i - 1] = dt * np.sin(sol[i - 1, 2])

        for i in range(1, self.n_timesteps):
            deriv[splits_row[0] + i, splits_col[2] + i - 1] = - dt * velocity[i - 1] * np.sin(sol[i - 1, 2])

        for i in range(1, self.n_timesteps):
            deriv[splits_row[1] + i, splits_col[2] + i - 1] = dt * velocity[i - 1] * np.cos(sol[i - 1, 2])

        return deriv
