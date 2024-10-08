import sys

from typing import Union

sys.path.insert(0, "../source/")
import fenics as dl
import numpy as np
from State import State


class myState(State):
    state: dl.Function
    convolution = None

    memory_tree = None
    memory_cell2vertex = None
    memory_vertex2dof = None
    memory_cell_coordinates = None
    measurement_memory = None
    derivative_memory = None
    bools_new_memories = True

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
        # self.gradient_space = dl.VectorFunctionSpace(fom.mesh, 'DG', fom.polyDim - 1)
        self.gradient_space = fom.gradient_space
        # less error-prone to use the gradient_space initialized in the FOM class rather than hard-coding it here

        self.final_time = self.grid_t[-1]
        self.setup_measurement_memory(meshDim=kwargs.get("memory_meshDim", 20 * self.fom.meshDim))

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
                # self.Du[k] = dl.interpolate(du, self.gradient_space)
                # todo: originally we had dl.project here, but that solves the projection PDE. Since we set up the gradient
                #  space specifically for the gradient, we can just interpolate it onto it. However, according to
                #  https://fenicsproject.discourse.group/t/evaluate-ufl-product-what-do-the-variables-mean/9270
                #  the interpolation does not work with legacy FEniCS

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
        """
        applies this state's interpolation rule for t to the passed collection of states. The ordering in the argument
        states needs to correspond to self.grid_t.

        For our transient state here, we interpolate linearly between the entries at positions k_right-1 and k_right,
        where k_right is chosen as the smallest index such that self.grid_t[k_right] >= t. Note that this is self.grid_t,
        which may be different from the drone.grid_t.
        """
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
        V = dl.FunctionSpace(mesh, "CG", 1)  # piecewise linear
        self.memory_cell_coordinates = mesh.coordinates()

        # the space in which we interpolate the derivatives of the momorized measurements
        self.memory_derivative_space = dl.VectorFunctionSpace(mesh, "DG", 0)  # piecewise constants

        # build a tree for locating cells within the mesh
        self.memory_tree = dl.BoundingBoxTree()
        self.memory_tree.build(mesh)
        self.memory_cell2vertex = mesh.topology()(mesh.topology().dim(), 0)
        # memory_cell2nodes returns, for a given cell number the node numbers of the cells edges

        self.memory_vertex2dof = dl.vertex_to_dof_map(V)

        # create a function in which measurements will be stored
        self.measurement_memory = np.zeros(self.n_steps, dtype=object)
        self.derivative_memory = np.zeros(self.n_steps, dtype=object)
        # we need one instance for each time step (for stationary model, n_steps = 1)

        for k in range(self.n_steps):
            # initialize FEniCS function on the mesh
            memory_k = dl.Function(V)

            # set all entries to NaN (since no measurements have been computed yet)
            memory_k.vector().vec().array = np.nan * np.ones(V.dim())

            # include into the memory
            self.measurement_memory[k] = memory_k

        # in order to not recompute the derivatives when no new memories have been added, we toggle these booleans
        # whenever the memory functions are modified
        self.bools_new_memories = [False] * self.n_steps

    def remember_measurement(self, pos, t, detector):
        """
        Check if a measurement has already been taken at position pos and time t (or at least in their vicinity).
        If so, return it. If not, compute the missing entries on the memory mesh.
        """
        # test if we already know a value for this position and time
        stored_data = self.apply_interpolation_rule(states=self.measurement_memory, t=t, x=pos)
        # todo: this part won't work with transient measurements yet

        # if so, return it
        if not np.isnan(stored_data):
            return stored_data

        # if not, find out in which mesh cell pos is located
        p = dl.Point(pos)
        cell_numbers = self.memory_tree.compute_entity_collisions(p)
        # all cells in the tree that contain p (multiple cells are possible if p is on cell boundary)

        # find out at which position we need to look in the memory array
        k_right = np.argmax(self.grid_t >= t)

        # we also need to compute measurements at the previous position
        # (because we are interpolating between measurements)
        # only exception is if we are already at position 0 (this happens in the stationary case, or when t=0)
        time_step_numbers = [k_right]
        if k_right > 0:
            time_step_numbers.append(k_right - 1)

        for cell_no in cell_numbers:

            # identify the nodes for this cell element
            node_numbers = self.memory_cell2vertex(cell_no)
            node_coordinates = self.memory_cell_coordinates[node_numbers]

            # compute those missing values (probably 6 if transient, 3 if stationary)
            for i, node_no in enumerate(node_numbers):

                # get position in dof map for the node we are looking at
                dof_no = self.memory_vertex2dof[node_no]

                for k in time_step_numbers:

                    if np.isnan(self.measurement_memory[k].vector().vec().array[dof_no]):
                        # compute data for this node
                        data = detector.measure_at_position(pos=node_coordinates[i, :], t=self.grid_t[k], state=self,
                                                            bool_from_memory=False)
                        # setting bool_from_memory to False ensures that we are actually computing a measurement
                        # and that we don't run into an infinite loop

                        # todo: passing t=self.grid_t[k] here is a potential source of error because it relies on the
                        #  interpolation method down the road identifying k again as the corresponding index.
                        #  Instead, we should pass k directly

                        # store this data in memory
                        self.measurement_memory[k].vector().vec().array[dof_no] = data

                        # mark that these memories have been modified since the last computation of the derivatives
                        self.bools_new_memories[k] = True

        # return the evaluation
        return self.apply_interpolation_rule(states=self.measurement_memory, t=t, x=pos)

    def remember_derivative(self, pos, t, detector):
        """
        In this function we compute the derivative of the memories stored in memory. We do **not** compute them with
        the chain rule, but use FEniCS functionality. This is for consistency: the memory measurements are build to be
        the field over which we optimize, so the derivatives should be of this field. Of course this means that the
        derivatives are not necessarily the ones we'd get from the chain rule - but that's the cost of using an
        approximation. The best choice is probably to start with the memory on until close enough to the minimum, and
        then toggle it off to remove the approximation error.
        """
        # update the derivatives of the measurements
        for k in range(self.n_steps):
            if self.bools_new_memories[k]:
                d_memory_k = dl.grad(self.measurement_memory[k])
                self.derivative_memory[k] = dl.project(d_memory_k, self.memory_derivative_space)
                # self.derivative_memory[k] = dl.interpolate(d_memory_k, self.memory_derivative_space)
                # dl.project solves the projection PDE. Interpolating is faster and should cause less error because
                # we know which space d_memory_k lives in, and it's memory_derivative_space.
                # However, using dl.interpolate is throwing an error. According to
                # https://fenicsproject.discourse.group/t/evaluate-ufl-product-what-do-the-variables-mean/9270
                # we need to use dl.project

                # track changes for memory_k from this point forward agin
                self.bools_new_memories[k] = False

        return self.apply_interpolation_rule(states=self.derivative_memory, t=t, x=pos)

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
            self, position: np.ndarray, t: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Given positions of type [x,y], return the value of the state at the positions

        # todo: does this function get called? Intuitively this feels redundant with the code in the pointwise drone
        """
        print(
            "myState.measure_pointwise got called. Why? Shouldn't the pointwise drone class take care of this?")

        state = self.get_state(t=t)
        return np.array(
            [state(x, y) for x, y in zip(position[:, 0], position[:, 1])]
        )
