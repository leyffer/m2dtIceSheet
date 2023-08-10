

class Brain():

    def __init__(self, fom, drone, **kwargs):

        self.fom = fom
        self.drone = drone

    def apply_para2obs(self, para, grid_t=None, state=None, flightpath=None, **kwargs):

        # should we catch the case where grid_t is None here and set a default?
        # I think it makes more sense to default back to the grid_t that self.fom uses, and, if the FOM is steady-state,
        # default to that of the drone. Eventually we don't want to use the same time grid for drone and transient state
        # anyway.

        # solve for state
        if state is None:
            state, grid_t_fom = self.fom.solve(para=para, grid_t=grid_t)
        else:
            grid_t_fom = kwargs.get("grid_t_fom")

        grid_t = grid_t_fom
        # todo: once we allow different time grids for drone and FOM, we need to remove grid_t=grid_t_fom here

        # determine flight path
        if flightpath is None:
            flightpath, grid_t_drone = self.drone.get_trajectory(grid_t=grid_t)
        else:
            grid_t_drone = kwargs.get("grid_t_drone")

        # fly out and measure
        observation = self.drone.measure(flightpath, state, mode=kwargs.get("mode", None), grid_t_fom=grid_t_fom, grid_t_drone=grid_t_drone)
        # todo: eventually we'll make the statistical analysis increasingly dependent on how the measurements are taken
        #  in which case allowing here to pass a measurement mode to the drone is prone to introducing errors. For
        #  testing now I think it's helpful (avoid juggling too many classes), but eventually we'll just want to use the
        #  drone's default: the drone gets equipped with its measurement divices before takeoff.

        return observation