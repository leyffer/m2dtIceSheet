import numpy as np
from typing import Dict, Any, Optional, Literal, Union

from CircularPath import CircularPath

# TODO: find my parents!

class CombinedCircularPath:
    """
    TODO - inheritance from the Path class would be helpful, but that class
      currently has unnecessarily specific parameters
    """

    def __init__(
        self,
        initial_x: float,
        initial_y: float,
        initial_heading: float,
        parameters: list[tuple] | np.ndarray,
        transition_times: list | np.ndarray,
    ):
        """
        A series of parameters and transition times between switching
        transition_times beginning with an initial time

        TODO - transition_time is a parameter as well and needs to be handled
        properly here (not currently considered at all)
        """
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.initial_heading = initial_heading
        self.parameters = parameters
        self.transition_times = transition_times

        if isinstance(parameters[0], np.ndarray):
            self.paths = [
                CircularPath(
                    np.array(
                        [
                            initial_x,
                            initial_y,
                            initial_heading,
                            parameters[0][0],
                            parameters[0][1],
                        ]
                    ),
                    initial_time=transition_times[0],
                )
            ]
        for (i, params), time in zip(enumerate(parameters[1:]), transition_times[1:]):
            previous_final_position = self.paths[i].position(time)
            previous_final_x, previous_final_y = (
                previous_final_position[0][0],
                previous_final_position[0][1],
            )
            previous_final_heading = self.paths[i].heading(time)
            self.paths.append(
                CircularPath(
                    np.array(
                        [
                            previous_final_x,
                            previous_final_y,
                            previous_final_heading,
                            params[0],
                            params[1],
                        ]
                    ),
                    initial_time=time,
                )
            )
            
    def position(self, t: Union[float, np.ndarray]):
        """
        Unlike the constituent paths, it is simpler to construct the absolute
        positions for this composite path and then get the relative positions
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])
        positions = np.empty((t.shape[0], 2))
        positions[:, 0] = self.initial_x
        positions[:, 1] = self.initial_y
        for (i, start_time), end_time in zip(
            enumerate(self.transition_times[:-1]), self.transition_times[1:]
        ):
            indicator = np.logical_and(t >= start_time, t < end_time)
            positions[indicator] = self.paths[i].position(t[indicator])
        indicator = t >= self.transition_times[-1]
        positions[indicator] = self.paths[-1].position(t[indicator])
        return positions

    def relative_position(self, t: Union[float, np.ndarray]):
        """
        Positions relative to the initial position
        """
        rel_positions = self.position(t)
        rel_positions[:, 0] -= self.initial_x
        rel_positions[:, 1] -= self.initial_y
        return rel_positions

    def d_position_d_initial_x(self, t: Union[float, np.ndarray]):
        """
        Derivative of position with respect to initial x position at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.ones((t.shape[0], 2))
        deriv[:, 1] = 0
        return deriv

    def d_position_d_initial_y(self, t: Union[float, np.ndarray]):
        """
        Derivative of position with respect to initial y position at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.ones((t.shape[0], 2))
        deriv[:, 0] = 0
        return deriv

    def d_position_d_initial_heading(self, t: Union[float, np.ndarray]):
        """
        Derivative of heading with respect to initial heading at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        rel_pos = self.relative_position(t)
        deriv = np.empty((t.shape[0], 2))
        deriv[:, 0] = -rel_pos[:, 1]
        deriv[:, 1] = rel_pos[:, 0]
        return deriv

    def d_position_d_velocities(self, t: Union[float, np.ndarray]):
        """
        Derivative of position with respect the various velocities provided at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.zeros((t.shape[0], len(self.paths), 2))

        for (i, start_time), end_time in zip(
            enumerate(self.transition_times[:-1]), self.transition_times[1:]
        ):
            indicator = np.logical_and(t >= start_time, t < end_time)
            deriv[indicator, i, :] = self.paths[i].d_position_d_velocity(t[indicator])
            deriv[t >= end_time, i, :] = self.paths[i].d_position_d_velocity(end_time)

        indicator = t >= self.transition_times[-1]
        deriv[indicator, -1, :] = self.paths[-1].d_position_d_velocity(t[indicator])

        return deriv

    def d_position_d_angular_velocities(self, t: Union[float, np.ndarray]):
        """
        Derivative of position with respect the various angular velocities provided at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.zeros((t.shape[0], len(self.paths), 2))

        for (i, start_time), end_time in zip(
            enumerate(self.transition_times[:-1]), self.transition_times[1:]
        ):
            indicator = np.logical_and(t >= start_time, t < end_time)
            deriv[indicator, i, :] = self.paths[i].d_position_d_angular_velocity(
                t[indicator]
            )
            deriv[t >= end_time, i, :] = self.paths[i].d_position_d_angular_velocity(
                end_time
            )

        indicator = t >= self.transition_times[-1]
        deriv[indicator, -1, :] = self.paths[-1].d_position_d_angular_velocity(
            t[indicator]
        )

        return deriv

    def d_position_d_parameters(self, t: Union[float, np.ndarray]):
        derivatives = {}
        if not isinstance(t, np.ndarray):
            t = np.array([t])
        # alpha[0] is initial x
        derivatives["initial x"] = self.d_position_d_initial_x(t)

        # alpha[1] is initial y
        derivatives["initial y"] = self.d_position_d_initial_y(t)

        # alpha[2] is initial heading
        derivatives["initial heading"] = self.d_position_d_initial_heading(t)

        derivatives["velocity"] = self.d_position_d_velocities(t)
        derivatives["angular velocity"] = self.d_position_d_angular_velocities(t)
        return derivatives