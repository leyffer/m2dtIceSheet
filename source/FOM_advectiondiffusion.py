# for finite element setup
import fenics as dl
dl.set_log_level(30)
import mshr

# for standard array manipulation
import numpy as np

# for sparse matrices and linear algebra
import scipy.sparse as sla
import scipy.sparse as sparse

# for the parameterization via the FEniCS Expression class
from Parameter import Parameter

import matplotlib.pyplot as plt  # plotting

# for parallelization
from mpi4py import MPI

comm = MPI.COMM_SELF
rank = comm.Get_rank()
nproc = comm.Get_size()


class FOM_advectiondiffusion:

    def __init__(self, meshDim=50, polyDim = 1, bool_mpi=False, kappa = 1e-3, dt = 0.1, final_time = 4, **kwargs):

        # MPI info
        self.bool_mpi = bool_mpi

        # discretization
        self.mesh = self.create_mesh(meshDim)
        self.boundary_marker = self.create_boundary_marker()
        self.velocity = self.create_velocity_field()  # for advection term
        self.V = dl.FunctionSpace(self.mesh, 'P', polyDim)  # trial and test space for advection-diffusion eq
        self.nFE = self.V.dim()  # FE dimension

        # true initial condition
        m_true = 'exp(-100 * ((x[0]-0.35)*(x[0]-0.35) + (x[1]-0.7)*(x[1]-0.7)))'
        self.m_true = dl.interpolate(dl.Expression('min(0.5, {})'.format(m_true), degree=3), self.V)

        # parameters
        self.m_parameterized = self.set_parameter_functions()

        # equation setup
        self.kappa = kappa
        self.dt = dt
        self.final_time = final_time

        # important matrices
        self.M = self.mass_matrix()
        self.I = self.inner_product_matrix()

        self.set_defaults(**kwargs)

    def set_parameter_functions(self):
        m = np.zeros(5, dtype=object)
        centers = [[0.35, 0.7], [0.8, 0.2], [0.7, 0.5], [0.1, 0.9], [0.1, 0.2]]

        for i in range(5):
            m_str = 'exp(-100 * ((x[0]-{})*(x[0]-{}) + (x[1]-{})*(x[1]-{})))'.format(centers[i][0], centers[i][0], centers[i][1], centers[i][1])
            yolo = dl.Expression('min(0.5, {})'.format(m_str), degree=1)
            m[i] = dl.interpolate(yolo, self.V)

        return m

    def set_defaults(self, **kwargs):
        return

    def inner_product_matrix(self):
        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)
        a = dl.inner(dl.grad(u), dl.grad(v)) * dl.dx

        A = dl.assemble(a)
        A = dl.as_backend_type(A).mat()  # PETSc matrix
        A = sparse.csr_matrix(A.getValuesCSR()[::-1], shape=A.size)
        return A + self.M

    def apply_inner_product(self, u, v):
        return self.apply_mass_matrix(u, v) + dl.assemble(dl.inner(dl.grad(u), dl.grad(v)) * dl.dx)

    def mass_matrix(self):
        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)

        a = dl.inner(u, v) * dl.dx

        A = dl.assemble(a)
        A = dl.as_backend_type(A).mat()  # PETSc matrix
        return sparse.csr_matrix(A.getValuesCSR()[::-1], shape=A.size)

    def apply_mass_matrix(self, u, v):
        return dl.assemble(dl.inner(u, v) * dl.dx)

    def create_mesh(self, meshDim):
        """
        this function creates the mesh (2D unit square with two cutouts to simulate houses)
        The input meshDim (int) is an approximate for how many nodes should be placed on the boundary in each direction.
        Total number of nodes will be in the order of meshDim x meshDim
        returns the mesh
        """
        r1 = mshr.Rectangle(dl.Point(0.0, 0.0), dl.Point(1.0, 1.0))
        r2 = mshr.Rectangle(dl.Point(0.25, 0.15), dl.Point(0.5, 0.4))
        r3 = mshr.Rectangle(dl.Point(0.6, 0.6), dl.Point(0.75, 0.85))

        together = r1 - r2 - r3
        if self.bool_mpi:
            return mshr.generate_mesh(comm, together, meshDim)
        return mshr.generate_mesh(together, meshDim)

    def create_boundary_marker(self):

        mesh = self.mesh

        # Construct facet markers
        bndry = dl.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        for f in dl.facets(mesh):
            mp = f.midpoint()
            if dl.near(mp[0], 0.0):  # left
                bndry[f] = 1
            elif dl.near(mp[0], 1.0):  # right
                bndry[f] = 2
            elif dl.near(mp[1], 0.0) or dl.near(mp[1], 1):  # walls
                bndry[f] = 3
            elif dl.near(mp[0], 0.25) or dl.near(mp[0], 0.5):
                if 0.15 <= mp[1] and mp[1] <= 0.4:
                    bndry[f] = 4
            elif dl.near(mp[1], 0.15) or dl.near(mp[1], 0.4):
                if 0.25 <= mp[0] and mp[0] <= 0.5:
                    bndry[f] = 4
            elif dl.near(mp[0], 0.6) or dl.near(mp[0], 0.75):
                if 0.6 <= mp[1] and mp[1] <= 0.85:
                    bndry[f] = 4
            elif dl.near(mp[1], 0.6) or dl.near(mp[1], 0.85):
                if 0.6 <= mp[0] and mp[0] <= 0.75:
                    bndry[f] = 4

        return bndry


    def create_velocity_field(self):
        """
        creates the underlying velocity field for the advetion term in the advection-diffusion equation. The
        velocity field is modeled as the solution to a steady state Navier Stokes equations.

        returns the velocity field as a FEniCS Function.
        """

        mesh = self.mesh
        bndry = self.boundary_marker

        P2 = dl.VectorElement("P", mesh.ufl_cell(), 2)
        P1 = dl.FiniteElement("P", mesh.ufl_cell(), 1)
        TH = dl.MixedElement([P2, P1])

        W = dl.FunctionSpace(mesh, TH)
        V = W.sub(0).collapse()  # H^1_0(Omega)^2
        Q = W.sub(1).collapse()  # L^2(Omega)
        # todo: do we need V and Q?

        bc_left = dl.DirichletBC(W.sub(0), (0, 1), bndry, 1)
        bc_right = dl.DirichletBC(W.sub(0), (0, -1), bndry, 2)
        bc_topbottom = dl.DirichletBC(W.sub(0), (0, 0), bndry, 3)
        bc_houses = dl.DirichletBC(W.sub(0), (0, 0), bndry, 4)
        bcW = [bc_left, bc_right, bc_topbottom, bc_houses]

        v, q = dl.TestFunctions(W)
        w = dl.Function(W)
        u, p = dl.split(w)

        # Define variational forms
        F = dl.Constant(1 / 50) * dl.inner(dl.grad(u), dl.grad(v)) * dl.dx + dl.dot(dl.dot(dl.grad(u), u), v) * dl.dx \
            - p * dl.div(v) * dl.dx - q * dl.div(u) * dl.dx

        # Solve the problem
        dl.solve(F == 0, w, bcW)
        return u

    def plot(self, u):
        """plots the state u"""

        # todo: distinguish between different types of u
        # todo: include more plotting parameters, such as vmax

        if isinstance(u, str):
            u = dl.Expression('{}'.format(u), degree=3)

        plt.figure()
        c = dl.plot(u, mesh=self.mesh)
        plt.colorbar(c)

    def find_next(self, u_old, dt, kappa):
        """
        solves for the next time step when applying implicit Euler to initial condition u_old with step size dt and
        diffusion parameter self.kappa
        """

        u = dl.Function(self.V)
        v = dl.TestFunction(self.V)

        F = dl.inner(u, v) * dl.dx
        F = F - dl.inner(u_old, v) * dl.dx
        F = F + dl.Constant(dt * self.kappa) * dl.inner(dl.grad(u), dl.grad(v)) * dl.dx
        F = F + dl.Constant(dt) * dl.inner(v * self.velocity, dl.grad(u)) * dl.dx

        dl.solve(F == 0, u)

        return u

    def implicit_Euler(self, m_init, dt=None, final_time=None, kappa=None, grid_t=None):

        if grid_t is None:

            if dt is None: dt=self.dt
            if final_time is None: final_time = self.final_time
            if kappa is None: kappa = self.kappa
            grid_t = np.linspace(0, final_time, int(final_time / dt + 1))

        # todo: instead of writing the time stepping method here, we should make one file with functions we can call on
        sol = np.empty(grid_t.shape, dtype=object)

        # sol[0] = dl.Function(self.V)
        # sol[0].interpolate(m_init)
        sol[0] = m_init

        for k in range(1, sol.shape[0]):
            sol[k] = self.find_next(sol[k - 1], dt=grid_t[k]-grid_t[k-1], kappa=kappa)

        return sol, grid_t

    def assemble_initial_condition(self, para):
        # todo: outsource the parameterization into a different class for playing around better

        if para.shape[0] == 1:
            # this is a very simple parameterization where the true initial condition is just scaled by a scalar
            return para[0] * self.m_true

        m = para[0] * self.m_parameterized[0]
        for i in range(1, para.shape[0]):
            m = m + para[i] * self.m_parameterized[i]
        return m

    def solve(self, para, grid_t=None):
        m_init = self.assemble_initial_condition(para)
        return self.implicit_Euler(m_init=m_init, grid_t=grid_t)