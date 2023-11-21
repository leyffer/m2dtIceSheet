"""!
Full order model using firedrake
"""
import sys
import firedrake as dl

import gmsh
from makeMesh import makeMesh

# TODO: make separate subclasses for the FEniCS and the FireDrake case. It
# should be a user decision which FE solver they want to use. Letting them make
# that decision by creating an environment that does specifically not include
# the other solver seems restrictive at best. It also means that when we share
# the code we need to supply two separate environments

import numpy as np  # For standard array manipulation
from mpi4py import MPI  # For parallelization



sys.path.insert(0, "../source/")
from FullOrderModel import FullOrderModel

# Initialize MPI communications for mesh generation using mshr
comm = MPI.COMM_SELF
rank = comm.Get_rank()
nproc = comm.Get_size()


class FOM(FullOrderModel):
    def __init__(
        self,
        meshDim: int = 50,
        polyDim: int = 1,
        bool_mpi: bool = False,
        kappa: float = 1e-3,
        dt: float = 0.1,
        final_time: float = 4,
        mesh_shape: str = "houses",
        **kwargs
    ):
        """! Initializer for the Full-order-model (FOM)
        @param meshDim  Integer number of mesh nodes on each boundary
        @param polyDim  Polynomial dimension of test and trial function space
        @param bool_mpi  Boolean indicating MPI communications for mesh generation
        @param kappa  Advection-diffusion equation parameter
        @param dt  Time step size for transient solves
        @param final_time  Final time for transient solutions
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
        self.mesh = self.create_mesh(meshDim)
        self.boundary_marker = self.create_boundary_marker()

        # Create the velocity field for the advection term
        self.velocity = self.create_velocity_field()

        # Trial and test space for advection-diffusion eq ('P' == Polynomial)
        self.V = dl.FunctionSpace(self.mesh, "P", polyDim)

        # Finite-element dimension
        self.nFE = self.V.dim()

        # True initial condition
        x, y = dl.SpatialCoordinate(self.mesh)
        self.m_true = dl.Function(self.V).interpolate(
            dl.exp(-100 * (pow(x - 0.35, 2) + pow(y - 0.7, 2)))
        )

        # Parameters
        self.m_parameterized = self.set_parameter_functions()

        # Equation setup
        self.kappa = kappa
        self.dt = dt
        self.final_time = final_time

        # Important matrices
        self.M = self.mass_matrix()
        self.I = self.inner_product_matrix()

        self.set_defaults(**kwargs)

    def set_parameter_functions(self) -> np.ndarray[dl.Function]:
        """! Initialize the parameterized functions used with the provided
        parameters to make the initial condition
            @return  Numpy array of separate initial condition elements that we
            scale and sum to get the initial condition
        """

        m = np.zeros(5, dtype=object)
        centers = [[0.35, 0.7], [0.8, 0.2], [0.7, 0.5], [0.1, 0.9], [0.1, 0.2]]

        # We have to use a conditional statement with Firedrake instead of a max function
        x, y = dl.SpatialCoordinate(self.mesh)
        for i, _ in enumerate(m):
            m_i = dl.Function(self.V)
            m[i] = m_i.interpolate(
                dl.conditional(
                    dl.gt(  # If a greater than b
                        dl.exp(
                            -100
                            * (pow(x - centers[i][0], 2) + pow(y - centers[i][1], 2))
                        ),
                        0.5,
                    ),
                    # then
                    0.5,
                    # otherwise
                    dl.exp(
                        -100 * (pow(x - centers[i][0], 2) + pow(y - centers[i][1], 2))
                    ),
                )
            )

        return m

    def set_defaults(self, **kwargs):
        """! Function for setting default values. Not implemented"""
        return

    def inner_product_matrix(self) -> dl.matrix.Matrix:
        """! Return an inner product matrix
        @return  Matrix for inner product of u, v
        """
        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)
        a = dl.inner(dl.grad(u), dl.grad(v)) * dl.dx

        A = dl.assemble(a)
        A.a += self.M.a  # Add the bilinear forms from the two matrices
        # TODO - Should boundary conditions also be combined somehow? The boundary conditions for these matrices are zero here
        # A.bcs = list(A.bcs) + list(self.M.bcs)  # Combine boundary conditions
        return A

    def apply_inner_product(self, u: dl.Function, v: dl.Function) -> dl.matrix.Matrix:
        """! Apply the inner product matrix
        @return  Matrix for inner product of u, v
        """
        return dl.assemble(
            dl.inner(dl.grad(u), dl.grad(v)) * dl.dx
        ) + self.apply_mass_matrix(u, v)

    def mass_matrix(self) -> dl.matrix.Matrix:
        """! Return the mass matrix
        @return  Mass matrix
        """
        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)

        a = dl.inner(u, v) * dl.dx

        A = dl.assemble(a)
        return A

    def apply_mass_matrix(self, u: dl.Function, v: dl.Function) -> dl.matrix.Matrix:
        """! Return the mass matrix
        @return  Mass matrix
        """
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
        if self.mesh_shape == "houses":
            mesh = dl.Mesh(makeMesh(meshDim=meshDim, meshShape=self.mesh_shape))
        elif self.mesh_shape == "square":
            mesh = dl.UnitSquareMesh(meshDim, meshDim)
        return mesh

    def create_boundary_marker(self):
        """! Create the markers for the boundary
        @return  Mesh indicators for the boundary elements
        """
        return

    def create_velocity_field(self) -> dl.Function:
        """! Creation of velocity field for the advection term in the advection-diffusion equation

        The velocity field is modeled as the solution to a steady state Navier
        Stokes equations.

            @return  Velocity field as a FEniCS/Firedrake Function.
        """

        mesh = self.mesh

        V = dl.VectorElement(
            "P", mesh.ufl_cell(), 2
        )  # H^1_0(Omega)^2, Velocity function space
        Q = dl.FiniteElement("P", mesh.ufl_cell(), 1)  # L^2(Omega)
        TH = dl.MixedElement([V, Q])

        W = dl.FunctionSpace(mesh, TH)
        if self.mesh_shape == "square":
            # Boundary conditions on velocity (V)

            ## Square mesh using gmsh (numbering is specified by makeMesh)
            # bc_bottom = dl.DirichletBC(W.sub(0), (0, 0), 1)
            # bc_right  = dl.DirichletBC(W.sub(0), (0,-1), 2)
            # bc_top    = dl.DirichletBC(W.sub(0), (0, 0), 3)
            # bc_left   = dl.DirichletBC(W.sub(0), (0, 1), 4)

            ## Square mesh using UnitSquareMesh (numbering is specified by UnitSquareMesh)
            bc_left = dl.DirichletBC(W.sub(0), (0, 1), 1)
            bc_right = dl.DirichletBC(W.sub(0), (0, -1), 2)
            bc_bottom = dl.DirichletBC(W.sub(0), (0, 0), 3)
            bc_top = dl.DirichletBC(W.sub(0), (0, 0), 4)

            # Since the boundary conditions are singular at the corners, the
            # order of bcs must end with the zero conditions to regularize,
            # i.e., the bottom and top walls must come after the left/right
            bcW = [bc_left, bc_right, bc_top, bc_bottom]
        if self.mesh_shape == "houses":
            # Boundary conditions on velocity (V)
            bc_bottom = dl.DirichletBC(W.sub(0), (0, 0), 1)
            bc_right = dl.DirichletBC(W.sub(0), (0, -1), 2)
            bc_top = dl.DirichletBC(W.sub(0), (0, 0), 3)
            bc_left = dl.DirichletBC(W.sub(0), (0, 1), 4)
            bc_houses = dl.DirichletBC(W.sub(0), (0, 0), 5)

            # Since the boundary conditions are singular at the corners, we
            # need the zero conditions after the left/right
            bcW = [bc_left, bc_right, bc_top, bc_bottom, bc_houses]

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

    def plot(self, u, mesh: dl.MeshGeometry = None):
        """! Plot the state u"""

        # TODO: distinguish between different types of u
        # TODO: include more plotting parameters, such as vmax
        # TODO: make firedrake compatible

        if mesh is None:
            mesh = self.mesh
        if isinstance(u, dl.Function):
            dl.tripcolor(u)
        elif str(type(u)) == "<class 'ufl.tensors.ListTensor'>":
            dl.quiver(u)
        elif isinstance(u, dl.MeshGeometry):
            dl.triplot(u)

    def find_next(
        self, u_old: dl.Function, dt: float, kappa: float = None
    ) -> dl.Function:
        """! Apply implicit Euler to the initial condition u_old with step size
        dt and diffusion parameter self.kappa to get an updated u
            @param u_old  Initial condition
            @param dt  Time step size
            @param kappa  Diffusion parameter (not used, self.kappa is used instead)
            @return  The updated u function
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
        dt: float = None,
        final_time: float = None,
        kappa: float = None,
        grid_t: np.ndarray[float] = None,
    ) -> tuple[np.ndarray[dl.Function], np.ndarray]:
        """! Perform implicit Euler repeatedly to integrate the transient problem over time
        @param m_init  Initial condition
        @param dt  Time step size for implicit Euler
        @param final_time  Final time to integrate to
        @param kappa  Diffusion coefficient
        @param grid_t  Array containing time values for integration
        @return  Tuple containing (numpy array with solutions to time integration; time grid)
        """

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
        # one file with functions we can call on
        sol = np.empty(grid_t.shape, dtype=object)

        # sol[0] = dl.Function(self.V)
        # sol[0].interpolate(m_init)
        sol[0] = m_init

        # Integrate in time over the time grid
        for k in range(1, sol.shape[0]):
            sol[k] = self.find_next(
                sol[k - 1], dt=grid_t[k] - grid_t[k - 1], kappa=kappa
            )

        return sol, grid_t

    def assemble_initial_condition(self, para: list[float]) -> dl.Function:
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

    def solve(self, para, grid_t=None):
        """! Solve the transient problem
        @param para  Parameters for the initial condition
        @param grid_t  Grid for time integration
        """
        m_init = self.assemble_initial_condition(para)
        return self.implicit_Euler(m_init=m_init, grid_t=grid_t)
