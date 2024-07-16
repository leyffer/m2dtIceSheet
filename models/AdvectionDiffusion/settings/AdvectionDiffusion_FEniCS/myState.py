import sys

sys.path.insert(0, "../source/")
import fenics as dl
import numpy as np
from State import State


class myState(State):
    state: dl.Function
    convolution = None

    memory_tree = None
    memory_cell2vertex = None
    memory_cell_coordinates = None

    def __init__(
        self,
        fom: "FOM",
        state: dl.Function,
        bool_is_transient: bool,
        parameter: np.ndarray,
        other_identifiers: dict,
        **kwargs
    ) -> None:
        super().__init__(
            fom, state, bool_is_transient, parameter, other_identifiers, **kwargs
        )
        # Gradient space is Vector space of one dimension lower than FOM dimension
        self.gradient_space = dl.VectorFunctionSpace(fom.mesh, 'DG', fom.polyDim - 1) # Discontinuous Galerkin

        self.final_time = self.grid_t[-1]
        self.setup_measurement_memory(meshDim=kwargs.get("memory_meshDim", 5 * self.fom.meshDim))

    def get_derivative(self, t: float = None, x=None):

        """
        computes and saves the spatial derivative of the state
        @return:
        """
        # make sure we don't recompute the derivative every time it gets called
        if self.Du is None:

            self.Du = np.zeros(self.n_steps, dtype=object)
            for k in range(self.n_steps):
                du = dl.grad(self.state[k])
                self.Du[k] = dl.project(du, self.gradient_space)

        if t is None:
            raise RuntimeError(
                "transient state myState.get_derivative called without specifying a time at which to evaluate")

        if x is not None:
            return self.apply_interpolation_rule(states=self.Du, t=t, x=x)
        def evaluate_Du(x):
            return self.apply_interpolation_rule(states=self.Du, t=t, x=x)

        # note: returning a function that evaluates the derivative does cause additional overhead and is not the most
        # elegant solution. However, the way the addition of the derivatives works in FEniCS, I couldn't find a simple
        # way to evaluate them at a given point. It works much nicer for the states. (Nicole, May 28, 2024)

        return evaluate_Du

    def get_state(self, t=None, x=None):
        if t is None:
            # todo: doesn't this error also happen right now when the state is stationary?
            raise RuntimeError(
                "transient state myState.get_state called without specifying a time at which to evaluate")

        return self.apply_interpolation_rule(states=self.state, t=t, x=x)

    def apply_interpolation_rule(self, states, t, x=None):

        if t > self.final_time:
            raise RuntimeError("transient state called for time step beyond simulation horizon")

        k_right = np.argmax(self.grid_t >= t)

        if self.grid_t[k_right] == t:
            if x is None:
                return states[k_right]
            else:
                return states[k_right](x)

        if k_right == 0:
            raise RuntimeError(
                "In myState.apply_interpolation_rule: encountered k_right = 0. This happened for time={}".format(t))

        t_left = self.grid_t[k_right - 1]
        t_right = self.grid_t[k_right]
        t_diff = t_right - t_left

        state_left = states[k_right - 1]
        state_right = states[k_right]
        state_diff = state_right - state_left

        if x is not None:
            eval_left = state_left(x)
            return eval_left + (t - t_left) * (state_right(x) - eval_left) / t_diff

        # apply linear interpolation rule
        return state_left + (t - t_left) * state_diff / t_diff

    def setup_measurement_memory(self, meshDim: int):
        """
        initializations for saving measurements on a mesh
        """
        # create mesh in which measurements will be stored
        mesh = self.fom.create_mesh(meshDim=meshDim)
        V = dl.FunctionSpace(mesh, "P", 1)
        self.memory_cell_coordinates = mesh.coordinates()

        # build a tree for locating cells within the mesh
        self.memory_tree = dl.BoundingBoxTree()
        self.memory_tree.build(mesh)
        self.memory_cell2vertex = mesh.topology()(mesh.topology().dim(), 0)
        # memory_cell2nodes returns, for a given cell number the node numbers of the cells edges

        self.memory_vertex2dof = dl.vertex_to_dof_map(V)

        # create a function in which measurements will be stored
        self.measurement_memory = dl.Function(V)

        # set all entries to NaN (since no measurements have been computed yet)
        self.measurement_memory.vector().vec().array = np.nan * np.ones(V.dim())

    def remember_measurement(self, pos, t, detector):
        """
        Check if a measurement has already been taken at position pos and time t (or at least in their vicinity).
        If so, return it. If not, compute the missing entries on the memory mesh.
        """
        # test if we already know a value for this position and time
        stored_data = self.measurement_memory(pos)
        # todo: this part won't work with transient measurements yet

        # if so, return it
        if not np.isnan(stored_data):
            return stored_data

        # if not, find out in which mesh cell pos is located
        p = dl.Point(pos)
        cell_numbers = self.memory_tree.compute_entity_collisions(p)
        # all cells in the tree that contain p (multiple cells are possible if p is on cell boundary)

        for cell_no in cell_numbers:

            # identify the nodes for this cell element
            node_numbers = self.memory_cell2vertex(cell_no)
            node_coordinates = self.memory_cell_coordinates[node_numbers]

            # compute those missing values (probably 6 if transient, 3 if stationary)
            for i, node_no in enumerate(node_numbers):

                # get position in dof map for the node we are looking at
                dof_no = self.memory_vertex2dof[node_no]

                if np.isnan(self.measurement_memory.vector().vec().array[dof_no]):
                    # compute data for this node
                    data = detector.measure_at_position(pos=node_coordinates[i, :], t=t, state=self,
                                                        bool_from_memory=False)
                    # setting bool_from_memory to False ensures that we are actually computing a measurement
                    # and that we don't run into an infinite loop

                    # store this data in memory
                    self.measurement_memory.vector().vec().array[dof_no] = data

        # return the evaluation
        return self.measurement_memory(p)

    def set_convolution(self, convolution, key):
        """
        this function is intended for the MyDroneGaussianEval class to save the convoluted state computed in
        MyDroneGaussianEval.measure such that it does not need to get re-computed for other flight paths or when
        taking the derivative. The "key" parameter is there to distinguish between different drones measuring this
        state.

        Discussion:
        Right now, we only consider one drone at a time, so the key is probably not strictly necessary, but I think
        it's probably good practice to build it in already. In particular, we can probably think of other use cases
        beyond the MyDroneGaussianEval class

        @param convolution:
        @param key: unique identifier (string) of the drone
        @return:
        """
        if self.convolution is None:
            self.convolution = {}
        self.convolution[key] = convolution

    def get_convolution(self, key):
        if isinstance(self.convolution, dict):
            return self.convolution.get(key)
        return None

    def measure_pointwise(
            self, position: np.ndarray, t: float | np.ndarray
    ) -> np.ndarray:
        """
        Given positions of type [x,y], return the value of the state at the positions

        # todo: does this function get called? Intuitively this feels redundant with the code in the pointwise drone
        """
        print(
            "myState_stationary.measure_pointwise got called. Why? Shouldn't the pointwise drone class take care of this?")

        state = self.get_state(t=t)
        return np.array(
            [state(x, y) for x, y in zip(position[:, 0], position[:, 1])]
        )


# todo: the functions below - why are they here? What are they used for? They seems out of place to me (Nicole, May 28, 2024)
def make_circle_kernel(radius: float, dx: float) -> np.ndarray:
    """!
    Make a circular uniform kernel

    @param radius  the radius of the kernel
    @param dx  the grid spacing of the space that the kernel will be applied to
    @return  a 2D kernel centered at zero with 1's everywhere within the radius
        of zero

    TODO - allow for values between 0-1 on the boundary of the circle
        (anti-aliasing based on fraction of a grid square filled)
    """
    w = int(np.ceil(radius / dx))
    x = np.linspace(-w * dx, w * dx, 2 * w + 1)
    y = np.linspace(-w * dx, w * dx, 2 * w + 1)
    X, Y = np.meshgrid(x, y)
    return (X**2 + Y**2 < radius**2).astype(
        float
    )  # no partial cells, but that would be nice


def make_truncated_gaussian_kernel(
    radius: float, dx: float, sigma: float
) -> np.ndarray:
    """!
    Make a truncated Gaussian kernel

    @param radius  the radius of the kernel (truncation)
    @param dx  the grid spacing of the space that the kernel will be applied to
    @param sigma  the sigma parameter of the Gaussian
    """
    w = int(np.ceil(radius / dx))
    x = np.linspace(-w * dx, w * dx, 2 * w + 1)
    y = np.linspace(-w * dx, w * dx, 2 * w + 1)
    X, Y = np.meshgrid(x, y)
    r_squared = X**2 + Y**2
    truncation = (r_squared < radius**2).astype(
        float
    )  # no partial cells, but that would be nice
    return (
        np.exp(-0.5 * r_squared / (sigma**2))
        / sigma
        / np.sqrt(2 * np.pi)
        * truncation
    )
