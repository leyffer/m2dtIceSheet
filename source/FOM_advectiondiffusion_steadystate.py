from FOM_advectiondiffusion import FOM_advectiondiffusion
from typing import Any

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

class FOM_advectiondiffusion_steadystate(FOM_advectiondiffusion):
    """! Full-order-model (FOM) for the advection-diffusion equation at steady state
    Subclass of FOM_advectiondiffusion
    """

    # Initialize the default forcing and parameters
    default_forcing = None
    default_para = None

    def set_defaults(self, **kwargs):
        """! Specify default forcing and parameters
            @param **kwargs  Not used
        """
        self.default_forcing = self.m_true
        self.default_para = [0.1, 20, 0.01]

    def solve_steady(
        self,
        forcing:dl.Function=None,
        para:list[float]=None
        ) -> dl.Function:
        """! Solve the steady state advection-diffusion equation
            @param forcing  The forcing function
            @param para  The equation parameters
        """
        forcing = forcing if forcing is not None else self.default_forcing
        para = para if para is not None else self.default_para

        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)
        sol = dl.Function(self.V)

        a = para[0] * dl.inner(dl.grad(u), dl.grad(v)) * dl.dx \
            + para[1] * dl.inner(v * self.velocity, dl.grad(u)) * dl.dx \
            + para[2] * dl.inner(u, v) * dl.dx

        f = dl.inner(forcing, v) * dl.dx

        if using_firedrake:
            if self.mesh_shape == "square":
                ## Square mesh using gmsh (numbering is specified by makeMesh)
                # bc_bottom = dl.DirichletBC(self.V, 0, 1)
                # bc_right  = dl.DirichletBC(self.V, 0, 2)
                # bc_top    = dl.DirichletBC(self.V, 0, 3)
                # bc_left   = dl.DirichletBC(self.V, 0, 4)

                ## Square mesh using UnitSquareMesh (numbering is specified by UnitSquareMesh)
                bc_left   = dl.DirichletBC(self.V, 0, 1)
                bc_right  = dl.DirichletBC(self.V, 0, 2)
                bc_bottom = dl.DirichletBC(self.V, 0, 3)
                bc_top    = dl.DirichletBC(self.V, 0, 4)
                # Since all bcs are zero, order does not matter
                bcs = [bc_bottom, bc_right, bc_top, bc_left]

            if self.mesh_shape == "houses":
                ## Houses boundary conditions (numbering is specified by makeMesh)
                bc_bottom = dl.DirichletBC(self.V, 0, 1)
                bc_right  = dl.DirichletBC(self.V, 0, 2)
                bc_top    = dl.DirichletBC(self.V, 0, 3)
                bc_left   = dl.DirichletBC(self.V, 0, 4)
                bc_houses = dl.DirichletBC(self.V, 0, 5)
                bcs = [bc_left, bc_right, bc_top, bc_bottom, bc_houses]
        else:
            boundary = self.boundary_marker
            bc_left       = dl.DirichletBC(self.V, 0, boundary, 1)
            bc_right      = dl.DirichletBC(self.V, 0, boundary, 2)
            bc_top_bottom = dl.DirichletBC(self.V, 0, boundary, 3)
            bc_houses     = dl.DirichletBC(self.V, 0, boundary, 4)
            bcs = [bc_left, bc_right, bc_top_bottom, bc_houses]

        dl.solve(a == f, sol, bcs = bcs)
        return sol

    def assemble_forcing(self, para:list[float]) -> dl.Function:
        """! Assemble the forcing function given parameters para

        The forcing function is composed of functions that are scaled by the provided parameters
            @param para  Forcing function parameters
            @return  The scaled sum of forcing functions
        """
        if para.shape[0] == 1:
            # This is a very simple parameterization where the true forcing is
            # just scaled by a scalar
            return para[0] * self.default_forcing

        m = para[0] * self.m_parameterized[0]
        # Multiply each parametrized forcing by the provided parameter and sum
        # together to get forcing
        for i in range(1, para.shape[0]):
            m = m + para[i] * self.m_parameterized[i]
        return m

    def solve(self, para:list[float], grid_t:Any=None) -> dl.Function:
        """! Solve the steady state equation with the parameterized forcing
            @param para  Forcing parameters
            @param grid_t  Time grid (not used in any computation)
        """
        forcing = self.assemble_forcing(para)
        return self.solve_steady(forcing=forcing), grid_t
