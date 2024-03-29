import numpy as np
from typing import Dict, Any, Optional, Literal




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