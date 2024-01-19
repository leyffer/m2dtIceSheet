"""
FlightPath class that transforms parameters alpha into a flight path and can
generate derivatives with respect to control parameters alpha.

Some flight paths are provided.

Currently a 2D path.

TODO - ensure consistent behavior when dealing with single time values (instead
of arrays of time values)
"""

import numpy as np
from typing import Dict


class Path:
    def __init__(self, alpha, initial_time: float = 0.0):
        """
        Path defined by parameters:
        - `"initial x"` : initial position in `x` coordinate
        - `"initial y"` : initial position in `y` coordinate
        - `"initial heading"` : initital heading direction (in radians)
        - `"velocity"` : constant velocity parameter (spatial units per time)
        - `"angular velocity"` : constant angular velocity (radians per time)
        """
        self.alpha = alpha  # Parameters
        if isinstance(alpha, np.ndarray):
            self.alpha = {
                "initial x": alpha[0],  # initial position vector
                "initial y": alpha[1],  # initial position vector
                "initial heading": alpha[2],  # initial heading (radians)
                "velocity": alpha[3],  # velocity
                "angular velocity": alpha[4],  # angular velocity
            }
        self.initial_time = initial_time

    def relative_position(self, t) -> np.ndarray:
        """
        Relative position must be implemented by the specified path
        """
        raise NotImplementedError

    def position(self, t) -> np.ndarray:
        """
        Get the position (x, y) given the parameters and time(s) t

        Uses relative positions and initial positions to get absolute positions
        """
        rel_positions = self.relative_position(t)
        positions = rel_positions + np.array(
            [[self.alpha["initial x"], self.alpha["initial y"]]]
        )

        return positions

    def d_position_d_initial_x(self, t) -> np.ndarray:
        """
        Derivative of positions with repsect to the initial x position at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.ones((t.shape[0], 2))
        deriv[:, 1] = 0
        return deriv

    def d_position_d_initial_y(self, t) -> np.ndarray:
        """
        Derivative of positions with repsect to the initial x position at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.ones((t.shape[0], 2))
        deriv[:, 0] = 0
        return deriv

    def d_position_d_initial_heading(self, t) -> np.ndarray:
        """
        Derivative of positions with respect to the initial heading at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        rel_pos = self.relative_position(t)
        deriv = np.empty((t.shape[0], 2))
        deriv[:, 0] = -rel_pos[:, 1]
        deriv[:, 1] = rel_pos[:, 0]
        return deriv

    def d_position_d_alpha(self, t):
        """
        Derivative of positions with respect to parameters alpha at time(s) t
        """
        raise NotImplementedError

    def heading(self, t):
        """
        Heading in radians at time(s) t
        """
        t = t - self.initial_time
        headings = t * self.alpha["angular velocity"] + self.alpha["initial heading"]
        return headings

    def heading_vector(self, t):
        """
        Unit vector of the heading at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        headings = self.heading(t)
        return np.hstack(
            (np.cos(headings[:, np.newaxis]), np.sin(headings[:, np.newaxis]))
        )


class CircularPath(Path):
    """
    Circular arc flight path (if the angular velocity is zero, becomes a linear path)

    Constant velocity and angular velocity
    """

    def __init__(
        self, alpha, linear_tolerance: float = 1e-14, initial_time: float = 0.0
    ):
        """
        Path defined by parameters:
        - `"initial x"` : initial position in `x` coordinate
        - `"initial y"` : initial position in `y` coordinate
        - `"initial heading"` : initital heading direction (in radians)
        - `"velocity"` : constant velocity parameter (spatial units per time)
        - `"angular velocity"` : constant angular velocity (radians per time)

        Additional parameters can be derived:
        - `"radius"` : the arc radius is equal to the velocity divided by the angular velocity
        """
        super(CircularPath, self).__init__(alpha, initial_time=initial_time)
        self.linear_tolerance = linear_tolerance

        # Induced parameters
        if self.alpha["angular velocity"] != 0:
            self.alpha["radius"] = (
                self.alpha["velocity"] / self.alpha["angular velocity"]
            )
        else:
            self.alpha["radius"] = np.inf

    @property
    def linear(self) -> bool:
        """
        If the angular velocity is smaller than the linear_tolerance value,
        assume that the path is linear (to avoid numerical problems)
        """
        return np.abs(self.alpha["angular velocity"]) <= self.linear_tolerance

    def relative_position(self, t) -> np.ndarray:
        """
        Get the position (x, y) relative to the initial position (at the initial time)
        given the parameters and time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        t = t - self.initial_time

        positions = np.empty((t.shape[0], 2))
        # Currently, numerical problems when angular velocity approaches zero (O(1e-14))
        if not self.linear:
            positions[:, 0] = (
                self.alpha["velocity"]
                / self.alpha["angular velocity"]
                * (
                    np.sin(
                        t * self.alpha["angular velocity"]
                        + self.alpha["initial heading"]
                    )
                    - np.sin(self.alpha["initial heading"])
                )
            )
            positions[:, 1] = (
                self.alpha["velocity"]
                / self.alpha["angular velocity"]
                * (
                    -np.cos(
                        t * self.alpha["angular velocity"]
                        + self.alpha["initial heading"]
                    )
                    + np.cos(self.alpha["initial heading"])
                )
            )
        else:  # Linear path
            positions[:, 0] = (
                t * self.alpha["velocity"] * np.cos(self.alpha["initial heading"])
            )
            positions[:, 1] = (
                t * self.alpha["velocity"] * np.sin(self.alpha["initial heading"])
            )

        return positions

    def d_position_d_velocity(self, t) -> np.ndarray:
        """
        Derivative of position with respect to velocity at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = self.relative_position(t) / self.alpha["velocity"]
        return deriv

    def d_position_d_angular_velocity(self, t) -> np.ndarray:
        """
        Derivative of position with respect to angular velocity at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        t = t - self.initial_time

        deriv = np.empty((t.shape[0], 2))

        if not self.linear:
            deriv[:, 0] = -self.alpha["velocity"] / self.alpha[
                "angular velocity"
            ] ** 2 * (
                np.sin(
                    t * self.alpha["angular velocity"] + self.alpha["initial heading"]
                )
                - np.sin(self.alpha["initial heading"])
            ) + self.alpha[
                "velocity"
            ] / self.alpha[
                "angular velocity"
            ] * (
                np.cos(
                    t * self.alpha["angular velocity"] + self.alpha["initial heading"]
                )
                * t
            )  # this is rel_pos_x/w + t*(-rel_pos_y + w/v cos(theta_0) )

            deriv[:, 1] = -self.alpha["velocity"] / self.alpha[
                "angular velocity"
            ] ** 2 * (
                -np.cos(
                    t * self.alpha["angular velocity"] + self.alpha["initial heading"]
                )
                + np.cos(self.alpha["initial heading"])
            ) + self.alpha[
                "velocity"
            ] / self.alpha[
                "angular velocity"
            ] * (
                np.sin(
                    t * self.alpha["angular velocity"] + self.alpha["initial heading"]
                )
                * t
            )  # this is rel_pos_y/w + t*(rel_pos_x + w/v sin(theta_0) )

        else:  # Limit as angular velocity goes to zero
            deriv[:, 0] = (
                -self.alpha["velocity"]
                / 2
                * np.sin(
                    t * self.alpha["angular velocity"] + self.alpha["initial heading"]
                )
                * t**2
            )
            deriv[:, 1] = (
                self.alpha["velocity"]
                / 2
                * np.cos(
                    t * self.alpha["angular velocity"] + self.alpha["initial heading"]
                )
                * t**2
            )
        return deriv

    def d_position_d_alpha(self, t) -> Dict["str", np.ndarray]:
        """
        Derivative of position with respect to control parameters alpha at time(s) t
        """
        derivs = {}
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        derivs["initial x"] = self.d_position_d_initial_x(t)
        derivs["initial y"] = self.d_position_d_initial_y(t)

        # alpha[2] is initial heading (radians); can be simplified to a function
        # of position relative to initial position only
        derivs["initial heading"] = self.d_position_d_initial_heading(t)

        # alpha[3] is velocity; can be simplified to a function of position again
        derivs["velocity"] = self.d_position_d_velocity(t)

        # alpha[4] is angular veolcity; not as easily simplified
        derivs["angular velocity"] = self.d_position_d_angular_velocity(t)

        return derivs


class CombinedCircularPath:
    """
    TODO - inheritance from the Path class would be helpful, but that class
    currently has unecessarily specific paramters
    """
    def __init__(
        self,
        initial_x: float,
        initial_y: float,
        initial_heading: float,
        parameters: list[dict] | np.ndarray,
        transition_times: list | np.ndarray,
    ):
        """
        A series of parameters and transition times between switching
        transition_times begining with an initial time

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

    def position(self, t):
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

    def relative_position(self, t):
        """
        Positions relative to the initial position
        """
        rel_positions = self.position(t)
        rel_positions[:, 0] -= self.initial_x
        rel_positions[:, 1] -= self.initial_y
        return rel_positions

    def d_position_d_initial_x(self, t):
        """
        Derivative of position with respect to initial x position at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.ones((t.shape[0], 2))
        deriv[:, 1] = 0
        return deriv

    def d_position_d_initial_y(self, t):
        """
        Derivative of position with respect to initial y position at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.ones((t.shape[0], 2))
        deriv[:, 0] = 0
        return deriv

    def d_position_d_initial_heading(self, t):
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

    def d_position_d_velocitys(self, t):
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

    def d_position_d_angular_velocitys(self, t):
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

    def d_position_d_paramters(self, t):
        derivs = {}
        if not isinstance(t, np.ndarray):
            t = np.array([t])
        # alpha[0] is initial x
        derivs["initial x"] = self.d_position_d_initial_x(t)

        # alpha[1] is initial y
        derivs["initial y"] = self.d_position_d_initial_y(t)

        # alpha[2] is initial heading
        derivs["initial heading"] = self.d_position_d_initial_heading(t)

        derivs["velocity"] = self.d_position_d_velocitys(t)
        derivs["angular velocity"] = self.d_position_d_angular_velocitys(t)
        return derivs

# TODO - add radius derivative to the CircularPath class and then add a call to
# that for the derivative calculation for this class
class CirclePath:
    """
    Speed and radius version of the circular path

    Similar to the CircularPath but uses the radius as a parameter instead of
    angular velocity: radius is equal to velocity/angular velocity

    Currently implemented in the code somewhere else
    """
    def __init__():
        raise NotImplementedError