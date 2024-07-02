import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def film_the_state(state, filename, flight=None, runtime=3, dt_disp=0.1):
    """
    creates a gif of the state at time steps approximately 0, dt_disp, 2*dt_disp, ...
    If a flight is provided, it is plotted too. The runtime determines the speed the
    gif will be played (provide in seconds, approximately)
    """

    if flight is None:
        bool_include_flight = False
        grid_t = state.grid_t
    else:
        bool_include_flight = True
        flightpath = flight.flightpath
        grid_t = flight.grid_t

    # convert settings
    slicer = np.argmin(np.abs(grid_t - dt_disp))
    grid_t_disp = flight.grid_t[::slicer]

    ims = []

    fig, ax = plt.subplots(1, 1)

    for k in range(grid_t_disp.shape[0]):
        background = state.fom.plot(state, time=grid_t[k], ax=ax).collections

        # adjust the title
        title = ax.annotate("t = {:.1f}.".format(grid_t_disp[k]), (0.5, 1.03), xycoords="axes fraction", ha="center")

        if bool_include_flight:

            # plot the drone part
            path, = plt.plot(flightpath[:k * slicer + 1, 0], flightpath[:k * slicer + 1, 1], color="r")
            position, = plt.plot(flightpath[k * slicer, 0], flightpath[k * slicer, 1], color="r", marker="d")

            # remember this frame
            ims.append(background + [path, position, title])
        else:
            # remember this frame
            ims.append(background + [title])

    ani = animation.ArtistAnimation(fig, ims, repeat=False)

    extended_filename = filename + ".gif"
    writervideo = animation.FFMpegWriter(fps=int(len(ims) / runtime))
    ani.save(extended_filename, writer=writervideo)


def film_the_data(data, filename, grid_t, runtime=3, dt_disp=0.1):
    """
    creates a gif of the data at time steps approximately 0, dt_disp, 2*dt_disp, ...
    The runtime determines the speed the gif will be played (provide in seconds, approximately)
    """

    # convert settings
    slicer = np.argmin(np.abs(grid_t - dt_disp))
    grid_t_disp = grid_t[::slicer]

    ims = []

    fig, ax = plt.subplots(1, 1)

    for k in range(grid_t_disp.shape[0]):
        im0, = plt.plot(grid_t[:k * slicer + 1], data[:k * slicer + 1], color="r")

        # adjust the title
        title = ax.annotate("t = {:.1f}.".format(grid_t_disp[k]), (0.5, 1.03), xycoords="axes fraction", ha="center")

        # remember this frame
        ims.append([im0, title])

    ani = animation.ArtistAnimation(fig, ims, repeat=False)

    extended_filename = filename + ".gif"
    writervideo = animation.FFMpegWriter(fps=int(len(ims) / runtime))
    ani.save(extended_filename, writer=writervideo)
