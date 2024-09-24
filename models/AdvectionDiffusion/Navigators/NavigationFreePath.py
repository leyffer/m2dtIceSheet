"""This is a dummy navigator used to essentially bypass the Navigator paradigm

For this navigator, the control parameters are the path
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np

sys.path.insert(0, "../../../source/")
from Navigation import Navigation

# from scipy.interpolate import make_interp_spline


class NavigationFreePath(Navigation):
    """
    Generic path defined by (x(t), y(t))
    """

    def __init__(
        self,
        alpha: np.ndarray = np.hstack(
            (np.linspace(0.5, 1, 401), np.linspace(0.5, 1, 401))
        ),
        grid_t: np.ndarray = np.arange(0, 4 + 1e-2, 1e-2),
        num_spatial_dimensions: int = 2,
    ):
        """! Path defined by (x(t), y(t)) and t

        Args:
        @param alpha:
        """
        # Parameters, in this case, x y coordinates for some time grid
        self.alpha = alpha
        self.grid_t = grid_t
        self.num_spatial_dimensions = num_spatial_dimensions

    def get_trajectory(
        self, alpha: np.ndarray = None, grid_t: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """! Get the trajectory of the drone given the flight parameters alpha

        @param alpha The specified flight parameters (this is the trajectory already)
        @param grid_t the time grid on which the drone position shall be computed
        @return  Tuple of (position over flight path, corresponding time for each position)
        """
        if alpha is None:
            alpha = self.alpha
        if grid_t is None:
            grid_t = self.grid_t

        position = self._convert_alpha_to_position(alpha)

        if position.shape[0] != grid_t.shape[0]:
            raise ValueError(
                f"Alpha converted to position ({position.shape}) and "
                f"time grid ({grid_t.shape}) don't have the matching shapes"
            )

        # TODO We need to use an interpolator if we want to allow sampling
        #  differently (e.g., arc-length sampling), but we need derivatives to
        #  do this correctly

        # self.interpolator = make_interp_spline(grid_t, position, k=3, axis = 0)

        return position, grid_t, self.drone.fom.identify_valid_positions(position)

    def _convert_alpha_to_position(self, alpha: np.ndarray) -> np.ndarray:
        return alpha.reshape(
            (self.num_spatial_dimensions, alpha.shape[0] // self.num_spatial_dimensions)
        ).T
