from __future__ import annotations
import sys

sys.path.insert(0, "../../../source/")

import numpy as np
from typing import Dict, Any, Optional, Literal
# from scipy.interpolate import make_interp_spline

from Navigation import Navigation

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
        """
        Path defined by (x(t), y(t))
        """
        self.alpha = (
            alpha  # Parameters, in this case, x y coordinates for some time grid
        )
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
                f"Alpha converted to position ({position.shape}) and time grid ({grid_t.shape}) don't have the matching shapes"
            )

        # use an interpolator? why?
        # self.interpolator = make_interp_spline(grid_t, position, k=3, axis = 0)

        return position, grid_t

    def _convert_alpha_to_position(self, alpha: np.ndarray) -> np.ndarray:
        return alpha.reshape((-1, alpha.shape[0]//self.num_spatial_dimensions)).T
