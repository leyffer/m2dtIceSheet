from FOM_advectiondiffusion import FOM_advectiondiffusion

import fenics as dl
dl.set_log_level(30)

class FOM_advectiondiffusion_steadystate(FOM_advectiondiffusion):

    default_forcing = None
    default_para = None

    def set_defaults(self, **kwargs):
        self.default_forcing = self.m_true
        self.default_para = [0.1, 20, 0.01]

    def solve_steady(self, forcing=None, para=None):

        forcing = forcing if forcing is not None else self.default_forcing
        para = para if para is not None else self.default_para

        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)
        sol = dl.Function(self.V)

        a = para[0] * dl.inner(dl.grad(u), dl.grad(v)) * dl.dx
        a = a + para[1] * dl.inner(v * self.velocity, dl.grad(u)) * dl.dx
        a = a + para[2] * dl.inner(u, v) * dl.dx

        f = dl.inner(forcing, v) * dl.dx

        bndry = self.boundary_marker
        bc_left = dl.DirichletBC(self.V, 0, bndry, 1)
        bc_right = dl.DirichletBC(self.V, 0, bndry, 2)
        bc_topbottom = dl.DirichletBC(self.V, 0, bndry, 3)
        bc_houses = dl.DirichletBC(self.V, 0, bndry, 4)
        bcs = [bc_left, bc_right, bc_topbottom, bc_houses]

        dl.solve(a == f, sol, bcs)
        return sol

    def assemble_forcing(self, para):
        return para[0] * self.default_forcing

    def solve(self, para, grid_t=None):
        forcing = self.assemble_forcing(para)
        return self.solve_steady(forcing=forcing), grid_t
