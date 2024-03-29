import numpy as np
from typing import Dict, Any, Optional, Literal

from Path import Path

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