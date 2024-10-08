"""
Full order model:
Has access to:
    mesh : domain
    velocity : velocity field for advection
    parameter_functions : the parameterized basis

Creates:
    solved State
"""

from __future__ import annotations

import sys
from typing import Optional

import fenics as dl
import matplotlib.pyplot as plt  # For plotting
import mshr
import numpy as np  # For standard array manipulation
import scipy.sparse as sparse  # For sparse matrices and linear algebra
from mpi4py import MPI  # For parallelization

sys.path.insert(0, "../source/")
from FullOrderModel import FullOrderModel
from myState import myState as State

dl.set_log_level(30)

# Initialize MPI communications for mesh generation using mshr
comm = MPI.COMM_SELF
rank = comm.Get_rank()
nproc = comm.Get_size()

# dl.parameters['allow_extrapolation'] = True


class FOM(FullOrderModel):
    n_spatial = 2

    def __init__(
        self,
        meshDim: int = 50,
        polyDim: int = 1,
        bool_mpi: bool = False,
        kappa: float = 1e-3,
        dt: float = 0.1,
        final_time: float = 4,
        mesh_shape: str = "houses",
        centers: list = None,
        **kwargs,
    ):
        """! Initializer for the Full-order-model (FOM)
        @param meshDim  Integer number of mesh nodes on each boundary
        @param polyDim  Polynomial dimension of test and trial function space
        @param bool_mpi  Boolean indicating MPI communications for mesh generation
        @param kappa  Advection-diffusion equation parameter
        @param dt  Time step size for transient solves
        @param final_time  Final time for transient solutions
        @param centers list of [x,y] coordinates for parameter centers
        @param **kwargs  Keyword arguments passed to set_defaults
        """

        # call initialization of parent class
        super().__init__()
        # in this particular instance it doesn't matter when we call it since
        # the FullOrderInitialization is currently empty

        # MPI indicator
        self.bool_mpi = bool_mpi

        # Discretization
        self.mesh_shape = mesh_shape
        self.meshDim = meshDim
        self.mesh = self.create_mesh(meshDim)
        self.boundary_marker = self.create_boundary_marker()
        self.memory_meshDim = kwargs.get("memory_meshDim", 20 * self.meshDim)

        # Create the velocity field for the advection term
        self.velocity = self.create_velocity_field(polyDim)

        # Trial and test space for advection-diffusion eq
        # ('P' == Polynomial, 'CG' = Continuous Galerkin, 'DG' = Discontinuous Galerkin)
        # P = CG according to https://fenicsproject.discourse.group/t/difference-between-finite-elements/5975
        self.V = dl.FunctionSpace(self.mesh, "CG", polyDim)
        self.polyDim = polyDim
        self.gradient_space = dl.VectorFunctionSpace(
            self.mesh, "DG", self.polyDim - 1
        )  # Discontinuous Galerkin
        # see ufl.finiteelement.elementlist.show_elements() for finite element
        # family options

        # Finite-element dimension
        self.nFE = self.V.dim()

        # True initial condition
        m_true = "exp(-100 * ((x[0]-0.35)*(x[0]-0.35) + (x[1]-0.7)*(x[1]-0.7)))"
        self.m_true = dl.interpolate(
            dl.Expression(f"min(0.5, {m_true})", degree=3), self.V
        )

        # Parameters
        self.m_parameterized = self.set_parameter_functions(
            centers=centers
        )  # parameter basis functions
        self.n_para = self.m_parameterized.shape[0]  # number of parameters

        # Equation setup
        self.kappa = kappa
        self.dt = dt
        self.final_time = final_time

        # Important matrices
        self.M = self.mass_matrix()
        self.I = self.inner_product_matrix()

        self.set_defaults(**kwargs)

    def set_parameter_functions(self, centers=None):
        """! Initialize the parameterized functions used with the provided
        parameters to make the initial condition
            @return  Numpy array of separate initial condition elements that we
            scale and sum to get the initial condition
        """

        if centers is None:
            centers = [[0.35, 0.7], [0.8, 0.2], [0.7, 0.5], [0.1, 0.9], [0.1, 0.2]]
        m = np.zeros(len(centers), dtype=object)

        for i, _ in enumerate(m):
            m_str = (
                f"exp(-100 * ((x[0]-{centers[i][0]})*(x[0]-{centers[i][0]})"
                + f" + (x[1]-{centers[i][1]})*(x[1]-{centers[i][1]})))"
            )
            m_truncated = dl.Expression(f"min(0.5, {m_str})", degree=1)
            m[i] = dl.interpolate(m_truncated, self.V)

        return m

    def set_defaults(self, **kwargs):
        """! Function for setting default values. Not implemented"""
        return

    def inner_product_matrix(self):  # -> dl.matrix.Matrix:
        """! Return an inner product matrix
        @return  Matrix for inner product of u, v
        """
        # TODO: add valid type hint

        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)
        a = dl.inner(dl.grad(u), dl.grad(v)) * dl.dx

        A = dl.assemble(a)
        A = dl.as_backend_type(A).mat()  # PETSc matrix
        A = sparse.csr_matrix(A.getValuesCSR()[::-1], shape=A.size)

        # TODO: is it correct that we don't need to apply boundary conditions?
        # TODO: use the function apply_inner_product to avoid double-definitions

        return A + self.M

    def apply_inner_product(
        self, u: dl.Function, v: dl.Function
    ):  # -> dl.matrix.Matrix:
        """! Apply the inner product matrix
        @return  Matrix for inner product of u, v
        """
        # TODO: add valid type hint
        return dl.assemble(
            dl.inner(dl.grad(u), dl.grad(v)) * dl.dx
        ) + self.apply_mass_matrix(u, v)

    def mass_matrix(self):  # -> dl.matrix.Matrix:
        """! Return the mass matrix
        @return  Mass matrix
        """
        # TODO: add valid type hint
        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)

        a = dl.inner(u, v) * dl.dx

        A = dl.assemble(a)
        A = dl.as_backend_type(A).mat()  # PETSc matrix
        A = sparse.csr_matrix(A.getValuesCSR()[::-1], shape=A.size)

        # TODO: is it correct that we don't need to apply boundary conditions?
        # TODO: use the function apply_mass_matrix to avoid double-definitions

        return A

    def apply_mass_matrix(self, u: dl.Function, v: dl.Function):  # -> dl.matrix.Matrix:
        """! Return the mass matrix
        @return  Mass matrix
        """
        # TODO: add valid type hint
        return dl.assemble(dl.inner(u, v) * dl.dx)

    def create_mesh(self, meshDim: int) -> dl.MeshGeometry:
        """! Create the mesh for the FOM. Mesh is either:
        - "houses": 2D unit square with two cutouts to simulate houses
        - "square": 2D unit square

        @param meshDim An approximate for how many nodes should be placed on
        the boundary in each direction. Total number of nodes will be in the
        order of meshDim * meshDim
        @return  Mesh geometry
        """
        if self.mesh_shape == "square":
            return dl.UnitSquareMesh(meshDim, meshDim, diagonal="right")

        together = mshr.Rectangle(dl.Point(0.0, 0.0), dl.Point(1.0, 1.0))

        if self.mesh_shape == "houses":
            r2 = mshr.Rectangle(dl.Point(0.25, 0.15), dl.Point(0.5, 0.4))
            r3 = mshr.Rectangle(dl.Point(0.6, 0.6), dl.Point(0.75, 0.85))
            together = together - r2 - r3

        if self.bool_mpi:
            return mshr.generate_mesh(comm, together, meshDim)
        return mshr.generate_mesh(together, meshDim)

    def identify_valid_positions(self, positions: np.ndarray) -> np.ndarray:
        """!
        Identify which positions lie within the domain

        @param positions: np.ndarray() of size (<arbitrary>, <spatial dimension>),
        each row containing a position for which needs to be checked if it lies within the modelled domain


        @return: np.ndarray of dimension (position.shape[0],) containing True at index i if position[i,:]
        is inside the modelled domain
        """
        buffer = 0
        a = (positions >= np.array([0, 0]) + buffer).all(axis=1)
        b = (positions <= np.array([1, 1]) - buffer).all(axis=1)
        valid_positions = a * b

        if self.mesh_shape == "square" or not self.mesh_shape == "houses":
            return valid_positions

        a = (positions > np.array([0.25, 0.15]) - buffer).all(axis=1)
        b = (positions < np.array([0.5, 0.4]) + buffer).all(axis=1)
        valid_positions[a * b] = False

        a = (positions > np.array([0.6, 0.6]) - buffer).all(axis=1)
        b = (positions < np.array([0.75, 0.85]) + buffer).all(axis=1)
        valid_positions[a * b] = False

        return valid_positions

    def create_boundary_marker(self):
        """! Create the markers for the boundary
        @return  Mesh indicators for the boundary elements
        """
        mesh = self.mesh

        # Construct facet markers
        boundary = dl.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        for f in dl.facets(mesh):
            mp = f.midpoint()
            if dl.near(mp[0], 0.0):  # left
                boundary[f] = 1
            elif dl.near(mp[0], 1.0):  # right
                boundary[f] = 2
            elif dl.near(mp[1], 0.0) or dl.near(mp[1], 1):  # walls
                boundary[f] = 3
            elif self.mesh_shape == "square":
                continue
            elif dl.near(mp[0], 0.25) or dl.near(mp[0], 0.5):
                if 0.15 <= mp[1] and mp[1] <= 0.4:
                    boundary[f] = 4
            elif dl.near(mp[1], 0.15) or dl.near(mp[1], 0.4):
                if 0.25 <= mp[0] and mp[0] <= 0.5:
                    boundary[f] = 4
            elif dl.near(mp[0], 0.6) or dl.near(mp[0], 0.75):
                if 0.6 <= mp[1] and mp[1] <= 0.85:
                    boundary[f] = 4
            elif dl.near(mp[1], 0.6) or dl.near(mp[1], 0.85):
                if 0.6 <= mp[0] and mp[0] <= 0.75:
                    boundary[f] = 4

        # TODO: account for the case where we have no houses

        return boundary

    def create_velocity_field(self, polyDim) -> dl.Function:
        """! Creation of velocity field for the advection term in the advection-diffusion equation

        The velocity field is modeled as the solution to a steady state Navier-Stokes equations.

        @return  Velocity field as a FEniCS Function.
        """

        mesh = self.mesh
        boundary = self.boundary_marker

        # Initialize function spaces:
        # H^1_0(Omega)^2, Velocity function space
        V = dl.VectorElement("CG", mesh.ufl_cell(), 2)
        # L^2(Omega)
        Q = dl.FiniteElement("CG", mesh.ufl_cell(), 1)
        TH = dl.MixedElement([V, Q])
        W = dl.FunctionSpace(mesh, TH)

        # Boundary conditions on velocity (V)
        bc_left = dl.DirichletBC(W.sub(0), (0, 1), boundary, 1)
        bc_right = dl.DirichletBC(W.sub(0), (0, -1), boundary, 2)
        bc_top_bottom = dl.DirichletBC(W.sub(0), (0, 0), boundary, 3)
        if self.mesh_shape == "houses":
            bc_houses = dl.DirichletBC(W.sub(0), (0, 0), boundary, 4)
            # Boundary conditions are applied in order, i.e., first boundary
            # conditions are able to be overwritten by later boundary
            # conditions; here, the corners are overwritten by the top/bottom
            bcW = [bc_left, bc_right, bc_top_bottom, bc_houses]
        else:
            bcW = [bc_left, bc_right, bc_top_bottom]

        # initialize trial and test functions
        v, q = dl.TestFunctions(W)
        w = dl.Function(W)
        u, p = dl.split(w)

        # Define variational form for Navier-Stokes
        F = (
            dl.Constant(1 / 50) * dl.inner(dl.grad(u), dl.grad(v)) * dl.dx
            + dl.dot(dl.dot(dl.grad(u), u), v) * dl.dx
            - p * dl.div(v) * dl.dx
            - q * dl.div(u) * dl.dx
        )

        # Solve the problem
        dl.solve(F == 0, w, bcs=bcW)

        return u

    def plot(self, u, mesh: dl.MeshGeometry = None, ax=None, time: float = 0, **kwargs):
        """! Plot the state u"""

        # TODO: distinguish between different types of u
        # TODO: include more plotting parameters, such as vmax

        if mesh is None:
            mesh = self.mesh
        if isinstance(u, str):
            u = dl.Expression(f"{u}", degree=3)
        if isinstance(u, State):
            u = u.get_state(t=time)

        if ax is None:
            plt.figure()
            c = dl.plot(u, mesh=mesh)
            plt.colorbar(c)
        else:
            c = dl.plot(u, mesh=mesh)
            plt.sca(ax)

            return c

    def find_next(
        self, u_old: dl.Function, dt: float, kappa: Optional[float] = None
    ) -> dl.Function:
        """!
        Apply implicit Euler to the initial condition u_old with step size
        dt and diffusion parameter self.kappa to get an updated u
            @param u_old  Initial condition
            @param dt  Time step size
            @param kappa  Diffusion parameter (not used, self.kappa is used instead)
            @return  The updated u dl.Function representing the next state dt in the future
        """
        if kappa is None:
            kappa = self.kappa

        u = dl.Function(self.V)
        v = dl.TestFunction(self.V)

        # Define variational form for the advection-diffusion equation
        F = (
            dl.inner(u, v) * dl.dx
            - dl.inner(u_old, v) * dl.dx
            + dl.Constant(dt * kappa) * dl.inner(dl.grad(u), dl.grad(v)) * dl.dx
            + dl.Constant(dt) * dl.inner(v * self.velocity, dl.grad(u)) * dl.dx
        )

        # Solve the problem
        dl.solve(F == 0, u)

        return u

    def implicit_Euler(
        self,
        m_init: dl.Function,
        dt: Optional[float] = None,
        final_time: Optional[float] = None,
        kappa: Optional[float] = None,
        grid_t: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray[dl.Function], np.ndarray, dict]:
        """! Perform implicit Euler repeatedly to integrate the transient problem over time
        @param m_init  Initial condition
        @param dt  Time step size for implicit Euler
        @param final_time  Final time to integrate to
        @param kappa  Diffusion coefficient
        @param grid_t  Array containing time values for integration
        @return sol  np.ndarray of dl.Function objects representing the solution over time
        @return grid_t  np.ndarray of floats representing the times for the solutions over time
        @ return other_identifiers  Dict with some other descriptive information about the solve
        """
        # TODO: add valid type hint

        # Set defaults
        if grid_t is None:
            if dt is None:
                dt = self.dt
            if final_time is None:
                final_time = self.final_time
            if kappa is None:
                kappa = self.kappa
            grid_t = np.linspace(0, final_time, int(final_time / dt + 1))

        # TODO: instead of writing the time stepping method here, we should make
        #  one file with functions we can call on
        sol = np.empty(grid_t.shape, dtype=object)
        sol[0] = dl.project(m_init, self.V)

        # Integrate in time over the time grid
        for k in range(1, sol.shape[0]):
            sol[k] = self.find_next(
                sol[k - 1], dt=grid_t[k] - grid_t[k - 1], kappa=kappa
            )

        other_identifiers = {"kappa": kappa}

        return sol, grid_t, other_identifiers

    def assemble_initial_condition(self, para: np.ndarray[float]) -> dl.Function:
        """! Assemble the initial condition given parameters para

        The initial condition is composed of functions that are scaled by the provided parameters
            @param para  Initial condition parameters
            @return  The scaled sum of initial conditions
        """
        # TODO: outsource the parameterization into a different class for playing around better

        if para.shape[0] == 1:
            # This is a very simple parameterization where the true initial
            # condition is just scaled by a scalar
            return para[0] * self.m_true

        # Multiply each parametrized initial condition by the provided parameter
        # and sum together to get initial condition
        m = para[0] * self.m_parameterized[0]
        for i in range(1, para.shape[0]):
            m = m + para[i] * self.m_parameterized[i]
        return m

    def solve(self, parameter: np.ndarray, **kwargs) -> State:
        """! Solve the transient problem
        @param parameter of interest
        @param kwargs should contain:
            "grid_t" for transient problems
        """
        m_init = self.assemble_initial_condition(parameter)
        grid_t = kwargs.get("grid_t", None)
        sol, grid_t, other_identifiers = self.implicit_Euler(
            m_init=m_init, grid_t=grid_t
        )

        state = State(
            fom=self,
            state=sol,
            bool_is_transient=True,
            parameter=parameter,
            memory_meshDim=self.memory_meshDim,
            other_identifiers=other_identifiers,
            grid_t=grid_t,
        )
        return state

