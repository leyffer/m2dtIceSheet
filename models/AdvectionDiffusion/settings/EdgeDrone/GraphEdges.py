import numpy as np
from typing import Dict, Any, Optional, Literal
from Edge import Edge


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