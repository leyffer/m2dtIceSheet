from FOM import FOM

import sys
sys.path.insert(0, "../source/")
from myState import myState as State

import fenics as dl
dl.set_log_level(30)

import numpy as np

class FOM_stationary(FOM):
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
        # TODO: Why these default parameters?

        self.default_forcing = self.m_true
        self.default_para = [0.1, 20, 0.01]
        # todo: rename to kappa

    def solve_steady(self, forcing: dl.Function=None, kappa: list[float]=None) -> dl.Function:
        """! Solve the steady state advection-diffusion equation
            @param forcing  The forcing function
            @param kappa  The equation parameters
        """
        # get parameterization and forcing information
        forcing = forcing if forcing is not None else self.default_forcing
        kappa = kappa if kappa is not None else self.default_para

        # set up trial and test functions
        u = dl.TrialFunction(self.V)
        v = dl.TestFunction(self.V)
        sol = dl.Function(self.V)

        # set up bilinear form
        a = kappa[0] * dl.inner(dl.grad(u), dl.grad(v)) * dl.dx \
            + kappa[1] * dl.inner(v * self.velocity, dl.grad(u)) * dl.dx \
            + kappa[2] * dl.inner(u, v) * dl.dx

        # set up forcing term
        f = dl.inner(forcing, v) * dl.dx

        # set boundary conditions
        boundary = self.boundary_marker
        bc_left       = dl.DirichletBC(self.V, 0, boundary, 1)
        bc_right      = dl.DirichletBC(self.V, 0, boundary, 2)
        bc_top_bottom = dl.DirichletBC(self.V, 0, boundary, 3)
        bc_houses     = dl.DirichletBC(self.V, 0, boundary, 4)
        bcs = [bc_left, bc_right, bc_top_bottom, bc_houses]
        # todo: catch the case without houses

        # solve the steadystate equation
        dl.solve(a == f, sol, bcs = bcs)

        other_identifiers = {
            "kappa" : kappa
        }

        return sol, other_identifiers

    def assemble_forcing(self, parameter:np.ndarray) -> dl.Function:
        """! Assemble the forcing function given parameters para

        The forcing function is composed of functions that are scaled by the provided parameters
            @param parameter  Forcing function parameters
            @return  The scaled sum of forcing functions
        """
        if parameter.shape[0] == 1:
            # This is a very simple parameterization where the true forcing is
            # just scaled by a scalar
            return parameter[0] * self.default_forcing

        m = parameter[0] * self.m_parameterized[0]
        # Multiply each parametrized forcing by the provided parameter and sum
        # together to get forcing
        for i in range(1, parameter.shape[0]):
            m = m + parameter[i] * self.m_parameterized[i]
        return m

    def solve(self, parameter: np.ndarray, *kwargs) -> State:
        """! Solve the transient problem
            @param parameter of interest
            @param kwargs should contain:
                "grid_t" for transient problems
        """
        forcing = self.assemble_forcing(parameter)
        sol, other_identifiers = self.solve_steady(forcing=forcing)

        state = State(fom=self, state=sol, bool_is_transient=False, parameter=parameter, other_identifiers=other_identifiers)
        return state
