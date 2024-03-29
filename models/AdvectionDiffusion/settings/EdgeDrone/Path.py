"""
Path class that transforms parameters alpha into a flight path and can
generate derivatives with respect to control parameters alpha.

Some flight paths are provided.

Currently a 2D path.

TODO - ensure consistent behavior when dealing with single time values (instead
of arrays of time values)
"""

import numpy as np
from typing import Dict, Any, Optional, Literal


class Path:
    """
    Generic path methods
    """

    def __init__(self, alpha: np.ndarray | Dict, initial_time: float = 0.0):
        """
        Path defined by parameters:
        - `"initial x"` : initial position in `x` coordinate
        - `"initial y"` : initial position in `y` coordinate
        - `"initial heading"` : initial heading direction (in radians)
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

    def relative_position(self, t: float | np.ndarray[float, Any]) -> np.ndarray:
        """
        Get the position relative to the initial position (x, y) given the parameters and time(s) t

        Uses absolute positions and initial positions to get relative positions

        Either absolute or relative positions must be implemented
        """
        # TODO - catch not implemented
        positions = self.position(t)
        rel_positions = positions - np.array(
            [[self.alpha["initial x"], self.alpha["initial y"]]]
        )

        return rel_positions

    def position(self, t: float | np.ndarray[float, Any]) -> np.ndarray:
        """
        Get the position (x, y) given the parameters and time(s) t

        Uses relative positions and initial positions to get absolute positions

        Either absolute or relative positions must be implemented
        """
        # TODO - catch not implemented
        rel_positions = self.relative_position(t)
        positions = rel_positions + np.array(
            [[self.alpha["initial x"], self.alpha["initial y"]]]
        )

        return positions

    def d_position_d_initial_x(self, t: float | np.ndarray[float, Any]) -> np.ndarray:
        """
        Derivative of positions with respect to the initial x position at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.ones((t.shape[0], 2))
        deriv[:, 1] = 0
        return deriv

    def d_position_d_initial_y(self, t: float | np.ndarray[float, Any]) -> np.ndarray:
        """
        Derivative of positions with respect to the initial x position at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.ones((t.shape[0], 2))
        deriv[:, 0] = 0
        return deriv

    def d_position_d_initial_heading(
        self, t: float | np.ndarray[float, Any]
    ) -> np.ndarray:
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

    def d_position_d_alpha(self, t: float | np.ndarray[float, Any]):
        """
        Derivative of positions with respect to parameters alpha at time(s) t
        """
        raise NotImplementedError

    def heading(self, t: float | np.ndarray[float, Any]):
        """
        Heading in radians at time(s) t
        """
        t = t - self.initial_time
        headings = t * self.alpha["angular velocity"] + self.alpha["initial heading"]
        return headings

    def heading_vector(self, t: float | np.ndarray[float, Any]):
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

    Parameters are:
    - initial x
    - initial y
    - initial heading
    - velocity
    - angular velocity

    Derived parameters (as a consequence of other parameters) are:
    - radius
    - center x
    - center y
    """

    def __init__(
        self,
        alpha: np.ndarray | Dict,
        linear_tolerance: float = 1e-14,
        initial_time: float = 0.0,
    ):
        """
        Path defined by parameters:
        - `"initial x"` : initial position in `x` coordinate
        - `"initial y"` : initial position in `y` coordinate
        - `"initial heading"` : initial heading direction (in radians)
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
            self.alpha["center x"] = self.alpha["initial x"] - self.alpha[
                "radius"
            ] * np.sin(self.alpha["initial heading"])
            self.alpha["center y"] = self.alpha["initial y"] + self.alpha[
                "radius"
            ] * np.cos(self.alpha["initial heading"])
        else:
            self.alpha["radius"] = np.inf
            self.alpha["center x"] = np.inf
            self.alpha["center y"] = np.inf

        self.alpha["round trip time"] = (
            2 * np.pi * self.alpha["radius"] / self.alpha["velocity"]
        )

    @property
    def linear(self) -> np.bool_:
        """
        If the angular velocity is smaller than the linear_tolerance value,
        assume that the path is linear (to avoid numerical problems)
        """
        return np.abs(self.alpha["angular velocity"]) <= self.linear_tolerance

    def relative_position(self, t: float | np.ndarray[float, Any]) -> np.ndarray:
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

    def d_position_d_velocity(self, t: float | np.ndarray[float, Any]) -> np.ndarray:
        """
        Derivative of position with respect to velocity at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = self.relative_position(t) / self.alpha["velocity"]
        return deriv

    def d_position_d_angular_velocity(
        self, t: float | np.ndarray[float, Any]
    ) -> np.ndarray:
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

    def d_position_d_alpha(
        self, t: float | np.ndarray[float, Any]
    ) -> Dict["str", np.ndarray]:
        """
        Derivative of position with respect to control parameters alpha at time(s) t
        """
        derivatives = {}
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        derivatives["initial x"] = self.d_position_d_initial_x(t)
        derivatives["initial y"] = self.d_position_d_initial_y(t)

        # alpha[2] is initial heading (radians); can be simplified to a function
        # of position relative to initial position only
        derivatives["initial heading"] = self.d_position_d_initial_heading(t)

        # alpha[3] is velocity; can be simplified to a function of position again
        derivatives["velocity"] = self.d_position_d_velocity(t)

        # alpha[4] is angular velocity; not as easily simplified
        derivatives["angular velocity"] = self.d_position_d_angular_velocity(t)

        return derivatives


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

    def d_position_d_velocities(self, t: float | np.ndarray[float, Any]):
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

    def d_position_d_angular_velocities(self, t: float | np.ndarray[float, Any]):
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

        derivatives["velocity"] = self.d_position_d_velocities(t)
        derivatives["angular velocity"] = self.d_position_d_angular_velocities(t)
        return derivatives


class CombinedCircularPathSingleSpeed:
    """
    TODO - inheritance from the Path class would be helpful, but that class
    currently has unnecessarily specific parameters
    """

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


# TODO - add radius derivative to the CircularPath class and then add a call to
# that for the derivative calculation for this class
class CirclePath(CircularPath):
    """
    Circular path with parameters radius [0] and speed [1]

    Similar to the CircularPath but uses the radius as a parameter instead of
    angular velocity: radius is equal to velocity/angular velocity

    Angular velocity is a derived parameter. The circle center is fixed (for derivatives).
    """

    def __init__(
        self,
        alpha: np.ndarray,
        center: np.ndarray = np.array([0.75 / 2, 0.55 / 2]),
        grid_t: np.ndarray = np.arange(0, 4 + 1e-2, 1e-2),
    ):
        """
        alpha is [0] radius and [1] speed

        Initial location is along x-axis from center (determined by initial `t`
        value)

        The alpha parameters are converted into the parameters used by the
        `CircularPath` class. Angular velocity is now a derived quantity.
        """
        # "initial x" : initial position in x coordinate
        # "initial y" : initial position in y coordinate
        # "initial heading" : initial heading direction (in radians)
        # "velocity" : constant velocity parameter (spatial units per time)
        # "angular velocity" : constant angular velocity (radians per time)

        self.grid_t = grid_t

        radius = alpha[0]
        velocity = alpha[1]
        converted_alpha = {
            "initial x": center[0] + radius * np.cos(self.grid_t[0] * 2 * np.pi),
            "initial y": center[1] + radius * np.sin(self.grid_t[0] * 2 * np.pi),
            "initial heading": np.pi / 2,
            "velocity": velocity,
            "angular velocity": velocity / radius,
        }
        super(CirclePath, self).__init__(
            alpha=converted_alpha, initial_time=self.grid_t[0]
        )

    def d_position_d_velocity(self, t: float | np.ndarray[float, Any]) -> np.ndarray:
        """
        Derivative of position with respect to velocity at time(s) t
        """
        if t is None:
            t = self.grid_t
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.empty((t.shape[0], 2))
        deriv[:, 0] = np.cos(
            self.alpha["initial heading"]
            + self.alpha["angular velocity"] * (t - self.initial_time)
        ) * (t - self.initial_time)
        deriv[:, 1] = np.sin(
            self.alpha["initial heading"]
            + self.alpha["angular velocity"] * (t - self.initial_time)
        ) * (t - self.initial_time)
        return deriv

    def d_position_d_radius(self, t: float | np.ndarray[float, Any]):
        """
        Derivative of position with respect to radius at time(s) t
        """
        if t is None:
            t = self.grid_t
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.empty((t.shape[0], 2))
        theta = self.alpha["angular velocity"] * (t - self.initial_time)
        c = np.cos(self.alpha["initial heading"] + theta)
        s = np.sin(self.alpha["initial heading"] + theta)
        deriv[:, 0] = s - c * theta
        deriv[:, 1] = -c - s * theta
        return deriv

    def d_position_d_alpha(
        self, t: float | np.ndarray[float, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Derivative of position with respect to control parameters alpha at time(s) t
        """
        derivatives = {}
        if t is None:
            t = self.grid_t
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        # alpha[0] is radius
        derivatives["radius"] = self.d_position_d_radius(t)

        # alpha[1] is velocity
        derivatives["velocity"] = self.d_position_d_velocity(t)

        return derivatives


class Edge:
    """
    A single edge path
    """

    def __init__(
        self,
        start_position: np.ndarray,
        end_position: np.ndarray,
        initial_time: float = 0.0,
        final_time: float = 1.0,
    ):
        """Path goes from start position to end position between some initial and final time"""
        self.start_position = start_position
        self.end_position = end_position
        self.initial_time = initial_time
        self.final_time = final_time
        self.dt = self.final_time - self.initial_time

    @property
    def speed(self) -> float:
        """Speed along the edge"""
        return self.length / self.dt

    @property
    def heading(self) -> float:
        """Angular heading along the edge"""
        v = self.end_position - self.start_position
        return np.arctan2(v[1], v[0])

    @property
    def length(self) -> float:
        """Length of the edge"""
        return np.linalg.norm(self.end_position - self.start_position)

    def relative_position(self, t: np.ndarray) -> np.ndarray:
        """Position relative to the start position"""
        if isinstance(t, np.ndarray):
            return (
                (t - self.initial_time)[:, np.newaxis]
                / self.dt
                * (self.end_position - self.start_position)[np.newaxis, :]
            )
        return (
            (t - self.initial_time)
            / self.dt
            * (self.end_position - self.start_position)
        )

    def position(self, t: np.ndarray) -> np.ndarray:
        """Absolute position"""
        return self.relative_position(t) + self.start_position


class GraphEdges:
    """
    Multiple graph edges
    """

    def __init__(
        self,
        nodes: np.ndarray,
        initial_time: float = 0.0,
        final_time: float = 1.0,
        grid_t: Optional[np.ndarray[float, Any]] = None,
        grid_t_mode: Literal["arc_length", "equal_time"] = "arc_length",
    ) -> None:
        """
        Given nodes, construct a path between them of edges
        """
        self.nodes = nodes
        self.lengths = np.linalg.norm(nodes[:-1] - nodes[1:], axis=-1)
        if grid_t is not None:
            self.final_time = grid_t[0]
            self.initial_time = grid_t[-1]
            self.dt = self.final_time - self.initial_time
        else:  # time grid not provided, use arc length (constant speed)
            self.final_time = final_time
            self.initial_time = initial_time
            self.dt = self.final_time - self.initial_time
            if grid_t_mode == "arc_length":
                arc_length = np.zeros((self.nodes.shape[0],))
                arc_length[1:] = np.cumsum(self.lengths)
                arc_length = arc_length / arc_length[-1]
                grid_t = arc_length * self.dt + self.initial_time
            elif grid_t_mode == "equal_time":
                grid_t = np.linspace(
                    self.initial_time, self.final_time, self.nodes.shape[0]
                )
        self.grid_t = grid_t
        self.edges = [
            Edge(n0, n1, t0, t1)
            for n0, n1, t0, t1 in zip(
                self.nodes[:-1], self.nodes[1:], self.grid_t[:-1], self.grid_t[1:]
            )
        ]

    def get_edge_number(self, t: np.ndarray) -> np.ndarray:
        """Get the segment/edge index from the provided time(s)"""
        edge_numbers = np.zeros(t.shape, dtype=int)
        t_head = 0
        for e_head, t_val in enumerate(t):
            while t_val > self.grid_t[t_head]:
                t_head += 1
                if t_head == len(self.grid_t):
                    break
            edge_numbers[e_head] = max(t_head - 1, 0)
        return edge_numbers

    def position(self, t: np.ndarray) -> np.ndarray:
        """Position at time t"""
        edge_numbers = self.get_edge_number(t)
        pos = np.empty((t.shape[0], self.nodes.shape[1]))
        for i, (edge_number, t_val) in enumerate(zip(edge_numbers, t)):
            pos[i] = self.edges[edge_number].position(t_val)
        return pos

    def d_position_d_node_locations(self, t: np.ndarray) -> np.ndarray:
        """
        Derivative of position with respect to node locations at time(s) t

        Dimensions of output are (time t, parameters (node x; node y), (x,y))
        """
        edge_numbers = self.get_edge_number(t)
        derivs = np.zeros((t.shape[0], 2*len(self.nodes), self.nodes[0].shape[0]))

        for i, (edge_ind, time) in enumerate(zip(edge_numbers, t)):
            # node x
            derivs[i, 2*edge_ind, 0] = (time - self.grid_t[edge_ind]) / (
                self.grid_t[edge_ind + 1] - self.grid_t[edge_ind]
            )
            derivs[i, 2*(edge_ind + 1), 0] = (self.grid_t[edge_ind + 1] - time) / (
                self.grid_t[edge_ind + 1] - self.grid_t[edge_ind]
            )
            # node y
            derivs[i, 2*edge_ind + 1, 1] = (time - self.grid_t[edge_ind]) / (
                self.grid_t[edge_ind + 1] - self.grid_t[edge_ind]
            )
            derivs[i, 2*(edge_ind + 1) + 1, 1] = (self.grid_t[edge_ind + 1] - time) / (
                self.grid_t[edge_ind + 1] - self.grid_t[edge_ind]
            )
        return derivs
