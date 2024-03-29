import numpy as np
from typing import Dict, Any, Optional, Literal



class CombinedCircularPathSingleSpeed:
    """
    TODO - inheritance from the Path class would be helpful, but that class
    currently has unnecessarily specific parameters
    """
    
    #TODO: find my parents

    def __init__(
        self,
        initial_x: float,
        initial_y: float,
        initial_heading: float,
        speed: float,
        angular_velocities: list[tuple] | np.ndarray,
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
        self.speed = speed
        self.angular_velocities = angular_velocities
        self.transition_times = transition_times

        if isinstance(angular_velocities, np.ndarray):
            self.paths = [
                CircularPath(
                    np.array(
                        [
                            initial_x,
                            initial_y,
                            initial_heading,
                            speed,
                            angular_velocities[0],
                        ]
                    ),
                    initial_time=transition_times[0],
                )
            ]

        self.transition_positions = np.zeros((len(transition_times) - 1, 2))
        self.transition_positions[0, :] = [initial_x, initial_y]

        for (i, angular_velocity), time in zip(
            enumerate(angular_velocities[1:]), transition_times[1:]
        ):
            previous_final_position = self.paths[i].position(time)
            previous_final_x, previous_final_y = (
                previous_final_position[0][0],
                previous_final_position[0][1],
            )
            self.transition_positions[i + 1, :] = [previous_final_x, previous_final_y]
            previous_final_heading = self.paths[i].heading(time)
            self.paths.append(
                CircularPath(
                    np.array(
                        [
                            previous_final_x,
                            previous_final_y,
                            previous_final_heading,
                            speed,
                            angular_velocity,
                        ]
                    ),
                    initial_time=time,
                )
            )

    def position(self, t: float | np.ndarray[float, Any]):
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

    def relative_position(self, t: float | np.ndarray[float, Any]):
        """
        Positions relative to the initial position
        """
        rel_positions = self.position(t)
        rel_positions[:, 0] -= self.initial_x
        rel_positions[:, 1] -= self.initial_y
        return rel_positions

    def d_position_d_initial_x(self, t: float | np.ndarray[float, Any]):
        """
        Derivative of position with respect to initial x position at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.ones((t.shape[0], 2))
        deriv[:, 1] = 0
        return deriv

    def d_position_d_initial_y(self, t: float | np.ndarray[float, Any]):
        """
        Derivative of position with respect to initial y position at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.ones((t.shape[0], 2))
        deriv[:, 0] = 0
        return deriv

    def d_position_d_initial_heading(self, t: float | np.ndarray[float, Any]):
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

    def d_position_d_velocity(self, t: float | np.ndarray[float, Any]) -> np.ndarray:
        """
        Derivative of position with respect to velocity at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = self.relative_position(t) / self.speed
        return deriv

    def d_position_d_angular_velocities(self, t: float | np.ndarray[float, Any]):
        """
        Derivative of position with respect the various angular velocities provided at time(s) t

        Dimensions of output are (time t, segment, (x,y))
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

    def d_position_d_parameters(self, t: float | np.ndarray[float, Any]):
        derivatives = {}
        if not isinstance(t, np.ndarray):
            t = np.array([t])
        # alpha[0] is initial x
        derivatives["initial x"] = self.d_position_d_initial_x(t)

        # alpha[1] is initial y
        derivatives["initial y"] = self.d_position_d_initial_y(t)

        # alpha[2] is initial heading
        derivatives["initial heading"] = self.d_position_d_initial_heading(t)

        derivatives["velocity"] = self.d_position_d_velocity(t)
        derivatives["angular velocity"] = self.d_position_d_angular_velocities(t)
        return derivatives