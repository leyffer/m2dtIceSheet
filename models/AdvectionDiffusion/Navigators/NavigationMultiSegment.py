from __future__ import annotations
import sys

sys.path.insert(0, "../../../source/")

import numpy as np
from typing import Dict, Any, Optional, Literal
from Navigation import Navigation
from Flight import Flight
from NavigationSegment import NavigationSegment


class NavigationMultiSegment(Navigation):
    fixed_controls_indices = []
    fixed_controls_values = []
    n_fixed_controls = 0

    # todo: something is going wrong when fixing the initial position as fixed control

    def __init__(self, grid_t: np.ndarray, transition_times: np.ndarray, subcontrols: None, *args, **kwargs):

        super().__init__(grid_t, *args, **kwargs)

        self.n_segments = transition_times.shape[0] - 1
        self.segments = np.zeros(self.n_segments, dtype=object)
        self.grid_t_sub = np.zeros(self.n_segments, dtype=object)
        self.substeps = np.zeros(self.n_segments + 1, dtype=int)

        t_init = 0
        for n in range(self.n_segments):
            t_stop = np.argmin(np.abs(self.grid_t - transition_times[n + 1]))
            self.substeps[n + 1] = t_stop
            self.grid_t_sub[n] = self.grid_t[t_init: t_stop + 1]
            self.segments[n] = NavigationSegment(grid_t=self.grid_t_sub[n])
            t_init = t_stop

        if t_init != self.grid_t.shape[-1] - 1:
            raise RuntimeError("In NavigationMultiSegment: did not end at final grid_t position", t_init)

        if subcontrols is None:
            subcontrols = [3, 4]
        self.subcontrols = subcontrols
        self.n_subcontrols = len(subcontrols)
        self.n_controls = 2 + self.n_subcontrols * self.n_segments

    def fix_controls(self, fixed_control_indices, fixed_control_values):
        self.fixed_controls_indices = fixed_control_indices
        self.fixed_controls_values = fixed_control_values
        self.n_fixed_controls = len(fixed_control_indices)

    def get_trajectory(
            self, alpha: np.ndarray, grid_t: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """! Get the trajectory of the drone given the flight parameters alpha

        @param alpha The specified flight parameters
        @param grid_t the time grid on which the drone position shall be computed
        @return  Tuple of (position over flight path, corresponding time for each position), grid_t
        """
        # use default time grid if none is provided
        # if grid_t is None:
        #     grid_t = self.grid_t
        # else:
        #     raise NotImplementedError("In NavigationMultiSegment: variable time grids not yet supported")
        # todo: support variable time grids
        grid_t = self.grid_t
        n_steps = grid_t.shape[0]

        alpha_sub = self.split_controls(alpha)
        current_control = alpha_sub[0]
        positions = np.array([[alpha[0], alpha[1]]])

        for n in range(self.n_segments):

            path_segment, __ = self.segments[n].get_trajectory(alpha=current_control, bool_ignore_validity_check=True)
            positions = np.vstack([positions, path_segment[1:, :]])

            final_heading = self.segments[n].final_heading(current_control)
            current_control = np.array([path_segment[-1, 0],  # final position is new initial position
                                        path_segment[-1, 1],  # final position is new initial position
                                        final_heading,  # final heading is new heading
                                        current_control[3],  # keep the same velocity
                                        current_control[4]])  # keep the same angular velocity

            if n < self.n_segments - 1:
                # overwrite those controls that vary for each segment
                current_control[self.subcontrols] = alpha_sub[n + 1]

        valid_positions = self.drone.fom.identify_valid_positions(positions)

        return positions, grid_t, valid_positions

    def split_controls(self, alpha):
        splitting = [5 - self.n_fixed_controls + self.n_subcontrols * i for i in range(self.n_segments)]
        alpha_sub = np.split(ary=alpha, indices_or_sections=splitting)
        current_control = np.zeros(5)
        current_control[self.fixed_controls_indices] = self.fixed_controls_values
        current_control[[i for i in range(5) if i not in self.fixed_controls_indices]] = alpha_sub[0]

        alpha_sub[0] = current_control
        return alpha_sub

    def d_position_d_control(self, flight: Flight) -> np.ndarray:
        """
        computes the derivative of the flightpath with respect to the control
        parameters in alpha. This class is problem specific and needs to be
        written by the user.

        0: starting position x value
        1: starting position y value
        2: starting heading
        3: velocity
        4: angular velocity

        @param flight: Flight object
        @return: gradient matrix,  Shape $<n_spatial * n_timesteps> \times <n_controls>$
        """
        # initialization
        deriv = np.zeros(
            (2, self.grid_t.shape[0], 5 - self.n_fixed_controls + (self.n_segments - 1) * self.n_subcontrols))
        alpha_sub = self.split_controls(flight.alpha)
        current_control = alpha_sub[0]

        subcontrols = [i for i in range(5) if i not in self.fixed_controls_indices]
        deriv_sub = self.segments[0].d_position_d_subcontrol(alpha=current_control,
                                                             grid_t=self.grid_t_sub[0],
                                                             subcontrols=subcontrols)
        deriv[0, :self.substeps[1] + 1, :5 - self.n_fixed_controls] = deriv_sub[:, ::2]
        deriv[0, self.substeps[1] + 1:, :5 - self.n_fixed_controls] = deriv_sub[-1, ::2]
        deriv[1, :self.substeps[1] + 1, :5 - self.n_fixed_controls] = deriv_sub[:, 1::2]
        deriv[1, self.substeps[1] + 1:, :5 - self.n_fixed_controls] = deriv_sub[-1, 1::2]

        col = 5 - self.n_fixed_controls
        for n in range(1, self.n_segments):
            current_control[self.subcontrols] = alpha_sub[n]
            deriv_sub = self.segments[n].d_position_d_subcontrol(alpha=current_control,
                                                                 grid_t=self.grid_t_sub[n],
                                                                 subcontrols=self.subcontrols)
            rows = [*range(self.substeps[n] + 1, self.substeps[n + 1] + 1)]
            deriv[0, rows, col:col + self.n_subcontrols] = deriv_sub[1:, ::2]
            deriv[0, rows[-1] + 1:, col:col + self.n_subcontrols] = deriv_sub[-1, ::2]
            deriv[1, rows, col:col + self.n_subcontrols] = deriv_sub[1:, 1::2]
            deriv[1, rows[-1] + 1:, col:col + self.n_subcontrols] = deriv_sub[-1, 1::2]

        deriv = np.vstack([deriv[0, :, :], deriv[1, :, :]])

        return deriv
