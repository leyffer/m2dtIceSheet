"""! Full-order-model (FOM) for advection diffusion equation using finite elements
"""
try:
    import fenics as dl
    dl.set_log_level(30)
    using_firedrake = False
except ImportError:
    import firedrake as dl
    using_firedrake = True
try:
    import mshr
    using_gmsh = False
except ImportError:
    import gmsh
    using_gmsh = True

import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For standard array manipulation
import scipy.sparse as sparse  # For sparse matrices and linear algebra
from makeMesh import makeMesh
from mpi4py import MPI  # For parallelization
if not using_firedrake:
    from Parameter import \
        Parameter  # For the parameterization via the FEniCS Expression class

# Initialize MPI communications for mesh generation using mshr
comm = MPI.COMM_SELF
rank = comm.Get_rank()
nproc = comm.Get_size()

class FOM_advectiondiffusion:
    """! Full-order-model (FOM) for advection-diffusion equation
    """

    def __init__(
        self,
        meshDim:int = 50,
        polyDim:int = 1,
        bool_mpi:bool = False,
        kappa:float = 1e-3,
        dt:float = 0.1,
        final_time:float = 4,
        mesh_shape:str = "houses",
        **kwargs):
        """! Initializer for the Full-order-model (FOM)
            @param meshDim  Integer number of mesh nodes on each boundary
            @param polyDim  Polynomial dimension of test and trial function space
            @param bool_mpi  Boolean indicating MPI communications for mesh generation
            @param kappa  Advection-diffusion equation parameter
            @param dt  Time step size for transient solves
            @param final_time  Final time for transient solutions
            @param **kwargs  Keyword arguments passed to set_defaults
        """

        # MPI indicator
        self.bool_mpi = bool_mpi

        # Discretization
        self.mesh_shape = mesh_shape
        self.mesh = self.create_mesh(meshDim)
        self.boundary_marker = self.create_boundary_marker()

        # Create the velocity field for the advection term
        self.velocity = self.create_velocity_field()

        # Trial and test space for advection-diffusion eq ('P' == Polynomial)
        self.V = dl.fem.FunctionSpace(self.mesh, 'P', polyDim)

        # Finite-element dimension
        self.nFE = self.V.dim()

        # True initial condition
        if using_firedrake:
            x, y = dl.SpatialCoordinate(self.mesh)
            self.m_true = dl.fem.Function(self.V).interpolate(
                dl.exp(-100 * (pow(x - 0.35, 2)  + pow(y - 0.7, 2)))
            )
        else:
            m_true = 'exp(-100 * ((x[0]-0.35)*(x[0]-0.35) + (x[1]-0.7)*(x[1]-0.7)))'
            self.m_true = dl.interpolate(dl.Expression(f'min(0.5, {m_true})', degree=3), self.V)

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

    def set_parameter_functions(self) -> np.ndarray[dl.fem.Function]:
        """! Initialize the parameterized functions used with the provided
        parameters to make the initial condition
            @return  Numpy array of separate initial condition elements that we
            scale and sum to get the initial condition
        """
        m = np.zeros(5, dtype=object)
        centers = [[0.35, 0.7], [0.8, 0.2], [0.7, 0.5], [0.1, 0.9], [0.1, 0.2]]

        if using_firedrake:
            # We have to use a conditional statement with Firedrake instead of a max function
            x, y = dl.SpatialCoordinate(self.mesh)
            for i, _ in enumerate(m):
                m_i = dl.fem.Function(self.V)
                m[i] = m_i.interpolate(
                    dl.conditional(
                        dl.gt(  # If a greater than b
                            dl.exp(-100 * (pow(x - centers[i][0], 2) + pow(y - centers[i][1], 2))),
                            0.5
                        ),
                        # then
                        0.5,
                        # otherwise
                        dl.exp(-100 * (pow(x - centers[i][0], 2) + pow(y - centers[i][1], 2)))
                    )
                )
        else:
            for i, _ in enumerate(m):
                m_str = f'exp(-100 * ((x[0]-{centers[i][0]})*(x[0]-{centers[i][0]})' \
                    + f' + (x[1]-{centers[i][1]})*(x[1]-{centers[i][1]})))'
                m_truncated = dl.Expression(f'min(0.5, {m_str})', degree=1)
                m[i] = dl.interpolate(m_truncated, self.V)

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
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

        A = dl.assemble(a)
        if not using_firedrake:
            A = dl.as_backend_type(A).mat()  # PETSc matrix
            A = sparse.csr_matrix(A.getValuesCSR()[::-1], shape=A.size)
            return A + self.M
        else:
            A.a += self.M.a  # Add the bilinear forms from the two matrices
            # TODO - Should boundary conditions also be combined somehow? The boundary conditions for these matrices are zero here
            # A.bcs = list(A.bcs) + list(self.M.bcs)  # Combine boundary conditions
            return A

    def apply_inner_product(self, u:dl.fem.Function, v:dl.fem.Function) -> dl.matrix.Matrix:
        """! Apply the inner product matrix
            @return  Matrix for inner product of u, v
        """
        return dl.assemble(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx) + self.apply_mass_matrix(u, v)

    def mass_matrix(self) -> dl.matrix.Matrix:
        """! Return the mass matrix
            @return  Mass matrix
        """
        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)

        a = ufl.inner(u, v) * ufl.dx

        A = dl.assemble(a)
        if not using_firedrake:
            A = dl.as_backend_type(A).mat()  # PETSc matrix
            A = sparse.csr_matrix(A.getValuesCSR()[::-1], shape=A.size)
        return A

    def apply_mass_matrix(self, u:dl.fem.Function, v:dl.fem.Function) -> dl.matrix.Matrix:
        """! Return the mass matrix
            @return  Mass matrix
        """
        return dl.assemble(ufl.inner(u, v) * ufl.dx)

    def create_mesh(self, meshDim:int) -> dl.mesh.Mesh:
        """! Create the mesh for the FOM. Mesh is either:
            - "houses": 2D unit square with two cutouts to simulate houses
            - "square": 2D unit square

            @param meshDim An approximate for how many nodes should be placed on
            the boundary in each direction. Total number of nodes will be in the
            order of meshDim * meshDim
            @return  Mesh geometry
        """
        if using_gmsh:
            if self.mesh_shape == "houses":
                mesh = dl.Mesh(makeMesh(meshDim=meshDim, meshShape=self.mesh_shape))
            elif self.mesh_shape == "square":
                mesh = dl.UnitSquareMesh(meshDim, meshDim)
            return mesh
        else:
            r1 = mshr.Rectangle(dl.Point(0.0, 0.0), dl.Point(1.0, 1.0))
            r2 = mshr.Rectangle(dl.Point(0.25, 0.15), dl.Point(0.5, 0.4))
            r3 = mshr.Rectangle(dl.Point(0.6, 0.6), dl.Point(0.75, 0.85))

            together = r1 - r2 - r3
            if self.bool_mpi:
                return mshr.generate_mesh(comm, together, meshDim)
            return mshr.generate_mesh(together, meshDim)

    def create_boundary_marker(self):
        """! Create the markers for the boundary
            @return  Mesh indicators for the boundary elements
        """
        if using_firedrake:  # We do not use boundary markers for Firedrake
            return

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

        return boundary


    def create_velocity_field(self) -> dl.fem.Function:
        """! Creation of velocity field for the advection term in the advection-diffusion equation

        The velocity field is modeled as the solution to a steady state Navier
        Stokes equations.

            @return  Velocity field as a FEniCS/Firedrake Function.
        """

        mesh = self.mesh
        if not using_firedrake:
            boundary = self.boundary_marker

        V = dl.VectorElement("P", mesh.ufl_cell(), 2)  # H^1_0(Omega)^2, Velocity function space
        Q = dl.FiniteElement("P", mesh.ufl_cell(), 1)  # L^2(Omega)
        TH = dl.MixedElement([V, Q])

        W = dl.fem.FunctionSpace(mesh, TH)
        if not using_firedrake:
            # Boundary conditions on velocity (V)
            bc_left       = dl.fem.DirichletBC(W.sub(0), (0, 1), boundary, 1)
            bc_right      = dl.fem.DirichletBC(W.sub(0), (0,-1), boundary, 2)
            bc_top_bottom = dl.fem.DirichletBC(W.sub(0), (0, 0), boundary, 3)
            bc_houses     = dl.fem.DirichletBC(W.sub(0), (0, 0), boundary, 4)
            bcW = [bc_left, bc_right, bc_top_bottom, bc_houses]
        else:
            if self.mesh_shape == "square":
                # Boundary conditions on velocity (V)

                ## Square mesh using gmsh (numbering is specified by makeMesh)
                # bc_bottom = dl.fem.DirichletBC(W.sub(0), (0, 0), 1)
                # bc_right  = dl.fem.DirichletBC(W.sub(0), (0,-1), 2)
                # bc_top    = dl.fem.DirichletBC(W.sub(0), (0, 0), 3)
                # bc_left   = dl.fem.DirichletBC(W.sub(0), (0, 1), 4)

                ## Square mesh using UnitSquareMesh (numbering is specified by UnitSquareMesh)
                bc_left   = dl.fem.DirichletBC(W.sub(0), (0, 1), 1)
                bc_right  = dl.fem.DirichletBC(W.sub(0), (0,-1), 2)
                bc_bottom = dl.fem.DirichletBC(W.sub(0), (0, 0), 3)
                bc_top    = dl.fem.DirichletBC(W.sub(0), (0, 0), 4)

                # Since the boundary conditions are singular at the corners, the
                # order of bcs must end with the zero conditions to regularize,
                # i.e., the bottom and top walls must come after the left/right
                bcW = [bc_left, bc_right, bc_top, bc_bottom]
            if self.mesh_shape == "houses":
                # Boundary conditions on velocity (V)
                bc_bottom = dl.fem.DirichletBC(W.sub(0), (0, 0), 1)
                bc_right  = dl.fem.DirichletBC(W.sub(0), (0,-1), 2)
                bc_top    = dl.fem.DirichletBC(W.sub(0), (0, 0), 3)
                bc_left   = dl.fem.DirichletBC(W.sub(0), (0, 1), 4)
                bc_houses = dl.fem.DirichletBC(W.sub(0), (0, 0), 5)

                # Since the boundary conditions are singular at the corners, we
                # need the zero conditions after the left/right
                bcW = [bc_left, bc_right, bc_top, bc_bottom, bc_houses]

        v, q = dl.TestFunctions(W)
        w = dl.fem.Function(W)
        u, p = dl.split(w)

        # Define variational form for Navier-Stokes
        F = dl.Constant(1 / 50) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx\
            + ufl.dot(ufl.dot(ufl.grad(u), u), v) * ufl.dx \
            - p * dl.div(v) * ufl.dx - q * dl.div(u) * ufl.dx

        # Solve the problem
        dl.solve(F == 0, w, bcs = bcW)
        return u

    def plot(self, u, mesh:dl.mesh.Mesh=None):
        """! Plot the state u"""

        # TODO: distinguish between different types of u
        # TODO: include more plotting parameters, such as vmax
        # TODO: make firedrake compatible

        if mesh is None:
            mesh = self.mesh
        if not using_firedrake:
            if isinstance(u, str):
                u = dl.Expression(f'{u}', degree=3)

            plt.figure()
            c = dl.plot(u, mesh=mesh)
            plt.colorbar(c)
        else:
            if isinstance(u, dl.fem.Function):
                dl.tripcolor(u)
            elif str(type(u)) == "<class 'ufl.tensors.ListTensor'>":
                dl.quiver(u)
            elif isinstance(u, dl.mesh.Mesh):
                dl.triplot(u)

    def find_next(self, u_old:dl.fem.Function, dt:float, kappa:float = None) -> dl.fem.Function:
        """! Apply implicit Euler to the initial condition u_old with step size
        dt and diffusion parameter self.kappa to get an updated u
            @param u_old  Initial condition
            @param dt  Time step size
            @param kappa  Diffusion parameter (not used, self.kappa is used instead)
            @return  The updated u function
        """
        if kappa is None:
            kappa = self.kappa

        u = dl.fem.Function(self.V)
        v = dl.TestFunction(self.V)

        # Define variational form for the advection-diffusion equation
        F = ufl.inner(u, v) * ufl.dx \
            - ufl.inner(u_old, v) * ufl.dx \
            + dl.Constant(dt * kappa) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
            + dl.Constant(dt) * ufl.inner(v * self.velocity, ufl.grad(u)) * ufl.dx

        # Solve the problem
        dl.solve(F == 0, u)

        return u

    def implicit_Euler(
        self,
        m_init:dl.fem.Function,
        dt:float=None,
        final_time:float=None,
        kappa:float=None,
        grid_t:np.ndarray[float]=None,
        ) -> tuple[np.ndarray[dl.fem.Function], np.ndarray]:
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
                dt=self.dt
            if final_time is None:
                final_time = self.final_time
            if kappa is None:
                kappa = self.kappa
            grid_t = np.linspace(0, final_time, int(final_time / dt + 1))

        # TODO: instead of writing the time stepping method here, we should make
        # one file with functions we can call on
        sol = np.empty(grid_t.shape, dtype = object)

        # sol[0] = dl.fem.Function(self.V)
        # sol[0].interpolate(m_init)
        sol[0] = m_init

        # Integrate in time over the time grid
        for k in range(1, sol.shape[0]):
            sol[k] = self.find_next(sol[k-1], dt = grid_t[k] - grid_t[k-1],
                                   kappa = kappa)

        return sol, grid_t

    def assemble_initial_condition(self, para:list[float]) -> dl.fem.Function:
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
        return self.implicit_Euler(m_init = m_init, grid_t = grid_t)
