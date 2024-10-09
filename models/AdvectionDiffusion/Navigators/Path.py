"""
Path class that transforms parameters alpha into a flight path and can
generate derivatives with respect to control parameters alpha.

Some flight paths are provided.

Currently a 2D path.

TODO - ensure consistent behavior when dealing with single time values (instead
of arrays of time values)
"""

from __future__ import annotations
import sys

sys.path.insert(0, "../../../source/")

import numpy as np
from typing import Dict, Any, Optional, Literal, Union
from Navigation import Navigation


class Path(Navigation):
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

    def relative_position(self, t: Union[float, np.ndarray]) -> np.ndarray:
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

    def position(self, t: Union[float, np.ndarray]) -> np.ndarray:
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

    def d_position_d_initial_x(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        Derivative of positions with respect to the initial x position at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.ones((t.shape[0], 2))
        deriv[:, 1] = 0
        return deriv

    def d_position_d_initial_y(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        Derivative of positions with respect to the initial x position at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        deriv = np.ones((t.shape[0], 2))
        deriv[:, 0] = 0
        return deriv

    def d_position_d_initial_heading(
        self, t: Union[float, np.ndarray]
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

    def d_position_d_alpha(self, t: Union[float, np.ndarray]):
        """
        Derivative of positions with respect to parameters alpha at time(s) t
        """
        raise NotImplementedError

    def heading(self, t: Union[float, np.ndarray]):
        """
        Heading in radians at time(s) t
        """
        t = t - self.initial_time
        headings = t * self.alpha["angular velocity"] + self.alpha["initial heading"]
        return headings

    def heading_vector(self, t: Union[float, np.ndarray]):
        """
        Unit vector of the heading at time(s) t
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        headings = self.heading(t)
        return np.hstack(
            (np.cos(headings[:, np.newaxis]), np.sin(headings[:, np.newaxis]))
        )
