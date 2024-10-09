from __future__ import annotations
import sys

sys.path.insert(0, "../../../source/")

import numpy as np
from typing import Dict, Any, Optional, Literal, Union

from CircularPath import CircularPath
from Navigation import Navigation

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

    def d_position_d_velocity(self, t: Union[float, np.ndarray]) -> np.ndarray:
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

    def d_position_d_radius(self, t: Union[float, np.ndarray]):
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
        self, t: Union[float, np.ndarray]
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
# TODO - add radius derivative to the CircularPath class and then add a call to
# that for the derivative calculation for this class
