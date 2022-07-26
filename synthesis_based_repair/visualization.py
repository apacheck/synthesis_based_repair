#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
# from nptyping import ndarray
import os
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D
import copy




def plot_trajectories(trajectories, ax, **kwargs):
    """ Plots the trajectories given in a numpy array on an axis that either accepts 2 dim or 3 dim graphics
    :param trajectories:
    :param ax:
    :param kwargs:
    :return:
    """
    for trajectory in trajectories:
        plot_trajectory(trajectory, ax, **kwargs)


def plot_trajectory(trajectory, ax, **kwargs):
    """ Plots the trajectory given in a numpy array on an axis that either accepts 2 dim or 3 dim graphics
    :param trajectories:
    :param ax:
    :param kwargs:
    :return:
    """
    if trajectory.shape[1] == 3:
        ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], **kwargs)
    elif trajectory.shape[1] == 2:
        ax.plot(trajectory[:, 1], trajectory[:, 1], **kwargs)
    else:
        raise Exception("Can't plot with these dimensions: {}".format(trajectories.shape))


def plot_sat_unsat_trajectories(trajectories, sat_array, ax_sat, ax_unsat, **kwargs):
    """ Plots the trajectories that satisfy or don't satisfy a constraint

    Plots the trajectories that satisfy a constraint (as given by a bool array) on one axis and the ones that don't on
    another

    :param trajectories:
    :param sat_array:
    :param ax_sat:
    :param ax_unsat:
    :param kwargs:
    :return:
    """

    kwargs['color'] = 'g'
    plot_trajectories(trajectories[sat_array], ax_sat, **kwargs)
    kwargs['color'] = 'r'
    plot_trajectories(trajectories[np.logical_not(sat_array)], ax_unsat, **kwargs)

    ax_sat.set_title("Satisfy Constraint: {:.2f}%".format(100 * np.mean(sat_array)))
    ax_unsat.set_title("Violate Constraint: {:.2f}%".format(100 * (1 - np.mean(sat_array))))


def apply_plot_limits(ax, plot_limits):
    """ Applies the plot limits to the plots

    :param ax:
    :param plot_limits:
    :return:
    """
    ax.set_xlim(plot_limits[0, :])
    ax.set_ylim(plot_limits[1, :])
    if plot_limits.shape[0] == 3:
        ax.set_zlim(plot_limits[2, :])


def create_ax_array(dim, ncols=3):
    """ Creates an array of axes for visualizing the results of the learning process

    Creates 3 separats axes (one for original trajectories, one for satisfy constraint, and one for violate constraint

    :param dim:
    :return: fig, ax
    """

    if dim == 2:
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    else:
        fig = plt.figure(figsize=(5 * ncols, 5))
        ax = [None] * ncols
        for ii in range(ncols):
            ax[ii] = fig.add_subplot(1, ncols, ii+1, projection='3d')
        ax = np.array(ax)

    return fig, ax
