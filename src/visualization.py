#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
# from nptyping import ndarray
import os
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D
import copy


# def plot_trajectories_and_symbols(skills: List[str], intermediate_states: List[List[str]],
#                                   folder_trajectories: str, symbols: Dict, limits: ndarray,
#                                   folder_save: str, plot_train: bool = False, **kwargs):
def plot_trajectories_and_symbols(skills, intermediate_states,
                                  folder_trajectories, symbols, limits,
                                  folder_save, plot_train=False, **kwargs):
    plt.ion()
    for skill in skills:
        _, _ = plot_one_skill_trajectories_and_symbols(skill, intermediate_states, folder_trajectories + "/" + skill,
                                                       symbols, limits, plot_train=plot_train, **kwargs)
        plt.savefig(folder_save + "/" + skill + ".png")


# def plot_one_skill_trajectories_and_symbols(skill: str, intermediate_states: List[List[str]], folder_trajectories: str,
#                                             symbols: Dict, limits: ndarray, ax: plt.Axes = None,
#                                             plot_train: bool = False,
#                                             **kwargs) -> Tuple[plt.Figure, plt.Axes]:
def plot_one_skill_trajectories_and_symbols(skill, intermediate_states,
                                            folder_trajectories,
                                            symbols, limits, ax=None,
                                            plot_train=False, three_d_one_plot=True,
                                            **kwargs):
    if ax is None:
        if limits.shape[0] == 3:
            fig = plt.figure()
            if not three_d_one_plot:
                ax = [[None, None], [None, None]]
                ax[0][0] = fig.add_subplot(2, 2, 1, projection='3d')
                ax[0][1] = fig.add_subplot(2, 2, 2)
                ax[1][0] = fig.add_subplot(2, 2, 3)
                ax[1][1] = fig.add_subplot(2, 2, 4)
                ax = np.array(ax)
            else:
                ax = fig.add_subplot(1, 1, 1, projection='3d')
        elif limits.shape[0] == 2:
            fig, ax = plt.subplots()

    # unique_states = []
    # for i_state in intermediate_states:
    #     for state in i_state:
    #         if state not in unique_states:
    #             unique_states.append(state)
    #
    # for state in unique_states:
    #     plot_symbol()

    if plot_train:
        folder_trajectories += "/train/"
    files_folder = [f for f in os.listdir(folder_trajectories) if os.path.isfile(os.path.join(folder_trajectories, f))]
    files_traj = [f for f in files_folder if 'rollout' in f]

    colors = ['r', 'b', 'g', 'k', 'c', 'm']
    for ii, f in enumerate(files_traj):
        data = np.loadtxt(folder_trajectories + "/" + f, delimiter=" ", dtype=float)
        ax = plot_one_skill_trajectories_and_symbols_numpy(skill, intermediate_states, data[np.newaxis, :, :], symbols,
                                                           limits, ax=ax, **kwargs)



    return fig, ax


# def plot_one_skill_trajectories_and_symbols_numpy(skill: str, intermediate_states: List[List[str]],
#                                                   data_trajectories: ndarray,
#                                                   symbols: Dict, limits: ndarray, ax: plt.Axes = None,
#                                                   **kwargs) -> plt.Axes:
def plot_one_skill_trajectories_and_symbols_numpy(skill, intermediate_states,
                                                  data_trajectories,
                                                  symbols, limits, ax=None, three_d_one_plot=True,
                                                  **kwargs):
    if ax is None:
        if limits.shape[0] == 3:
            fig = plt.figure()
            if not three_d_one_plot:
                ax = [[None, None], [None, None]]
                ax[0][0] = fig.add_subplot(2, 2, 1, projection='3d')
                ax[0][1] = fig.add_subplot(2, 2, 2)
                ax[1][0] = fig.add_subplot(2, 2, 3)
                ax[1][1] = fig.add_subplot(2, 2, 4)
                ax = np.array(ax)
            else:
                ax = fig.add_subplot(1, 1, 1, projection='3d')
        elif limits.shape[0] == 2:
            fig, ax = plt.subplots()

    if limits.shape[0] == 3:
        # 3d plot
        if three_d_one_plot:
            ax_three = ax
        else:
            ax_three = ax[0, 0]

        ax_three.set_xlim(limits[0, :])
        ax_three.set_ylim(limits[1, :])
        ax_three.set_zlim(limits[2, :])
        ax_three.set_title(skill)
        ax_three.set_xlabel('x')
        ax_three.set_ylabel('y')
        ax_three.set_zlabel('z')

        if not three_d_one_plot:
            # y vs z
            ax[0, 1].set_xlim(limits[1, :])
            ax[0, 1].set_ylim(limits[2, :])
            ax[0, 1].set_xlabel('y')
            ax[0, 1].set_ylabel('z')

            # x vs z
            ax[1, 0].set_xlim(limits[0, :])
            ax[1, 0].set_ylim(limits[2, :])
            ax[1, 0].set_xlabel('x')
            ax[1, 0].set_ylabel('z')

            # x vs y
            ax[1, 1].set_xlim(limits[0, :])
            ax[1, 1].set_ylim(limits[1, :])
            ax[1, 1].set_xlabel('x')
            ax[1, 1].set_ylabel('y')
    elif limits.shape[0] == 2:

        # x vs y
        ax.set_xlim(limits[0, :])
        ax.set_ylim(limits[1, :])
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # Plot symbols
    if limits.shape[0] == 3 and three_d_one_plot and symbols is not None:
        for sym_name, symbol in symbols.items():
            plot_symbol(symbol, limits, ax=ax, color=symbol.get_color(), alpha=0.1)
    if limits.shape[0] == 3 and not three_d_one_plot:
        if symbols is not None:
            for sym_name, symbol in symbols.items():
                if np.all(symbol.get_dims() == 0):
                    x_low = symbol.get_bnds()[0, 0]
                    x_high = symbol.get_bnds()[0, 1]
                    ax[1, 0].plot([x_low, x_low], limits[2, :], 'k')
                    ax[1, 0].plot([x_high, x_high], limits[2, :], 'k')
                    ax[1, 0].text(np.mean(symbol.get_bnds()[0, :]), limits[2, 0], sym_name, ha='center', va='bottom')
                    ax[1, 1].plot([x_low, x_low], limits[1, :], 'k')
                    ax[1, 1].plot([x_high, x_high], limits[1, :], 'k')
                    ax[1, 1].text(np.mean(symbol.get_bnds()[0, :]), limits[1, 0], sym_name, ha='center', va='bottom')
                if np.all(symbol.get_dims() == 1):
                    y_low = symbol.get_bnds()[0, 0]
                    y_high = symbol.get_bnds()[0, 1]
                    ax[0, 1].plot([y_low, y_low], limits[2, :], 'k')
                    ax[0, 1].plot([y_high, y_high], limits[2, :], 'k')
                    ax[0, 1].text(np.mean(symbol.get_bnds()[0, :]), limits[2, 0], sym_name, ha='center', va='bottom')
                    ax[1, 1].plot(limits[0, :], [y_low, y_low], 'k')
                    ax[1, 1].plot(limits[0, :], [y_high, y_high], 'k')
                    ax[1, 1].text(limits[0, 0], np.mean(symbol.get_bnds()[0, :]), sym_name, ha='center', va='bottom')
                if np.all(symbol.get_dims() == 2):
                    z_low = symbol.get_bnds()[0, 0]
                    z_high = symbol.get_bnds()[0, 1]
                    ax[0, 1].plot(limits[1, :], [z_low, z_low], 'k')
                    ax[0, 1].plot(limits[1, :], [z_high, z_high], 'k')
                    ax[0, 1].text(np.mean(symbol.get_bnds()[0, :]), limits[1, 0], sym_name, ha='center', va='bottom')
                    ax[1, 0].plot(limits[0, :], [z_low, z_low], 'k')
                    ax[1, 0].plot(limits[0, :], [z_high, z_high], 'k')
                    ax[1, 0].text(limits[0, 0], np.mean(symbol.get_bnds()[0, :]), sym_name, ha='center', va='bottom')
                if np.array_equal(symbol.get_dims(), np.array([0, 1, 2])):
                    x_low = symbol.get_bnds()[0, 0]
                    x_high = symbol.get_bnds()[0, 1]
                    y_low = symbol.get_bnds()[1, 0]
                    y_high = symbol.get_bnds()[1, 1]
                    z_low = symbol.get_bnds()[2, 0]
                    z_high = symbol.get_bnds()[2, 1]
                    ax[1, 0].add_patch(Rectangle((x_low, z_low), (x_high - x_low), (z_high - z_low),
                                                 edgecolor='k',
                                                 facecolor='none',
                                                 lw=1))
                    ax[1, 0].text(np.mean(symbol.get_bnds()[0, :]), np.mean(symbol.get_bnds()[2, :]), sym_name, ha='center',
                                  va='center')
                    ax[1, 1].add_patch(Rectangle((x_low, y_low), (x_high - x_low), (y_high - y_low),
                                                 edgecolor='k',
                                                 facecolor='none',
                                                 lw=1))
                    ax[1, 1].text(np.mean(symbol.get_bnds()[0, :]), np.mean(symbol.get_bnds()[1, :]), sym_name, ha='center',
                                  va='center')
                    ax[0, 1].add_patch(Rectangle((y_low, z_low), (y_high - y_low), (z_high - z_low),
                                                 edgecolor='k',
                                                 facecolor='none',
                                                 lw=1))
                    ax[0, 1].text(np.mean(symbol.get_bnds()[1, :]), np.mean(symbol.get_bnds()[2, :]), sym_name, ha='center',
                                  va='center')
    elif limits.shape[0] == 2:
        if symbols is not None:
            for sym_name, symbol in symbols.items():
                if np.size(symbol.get_dims()) == 1:
                    if symbol.get_dims() == [0]:
                        x_low = symbol.get_bnds()[0, 0]
                        x_high = symbol.get_bnds()[0, 1]
                        ax.plot([x_low, x_low], limits[1, :], 'k')
                        ax.plot([x_high, x_high], limits[1, :], 'k')
                        ax.text(np.mean(symbol.get_bnds()[0, :]), limits[1, 0], sym_name, ha='center', va='bottom')
                    if symbol.get_dims() == [1]:
                        y_low = symbol.get_bnds()[1, 0]
                        y_high = symbol.get_bnds()[1, 1]
                        ax.plot(limits[0, :], [y_low, y_low], 'k')
                        ax.plot(limits[0, :], [y_high, y_high], 'k')
                        ax.text(limits[0, 0], np.mean(symbol.get_bnds()[1, :]), sym_name, ha='center', va='bottom')
                else:
                    if symbol.get_type() == 'rectangle':
                        sym_limits = copy.deepcopy(limits)
                        sym_limits[symbol.get_dims(), :] = symbol.get_bnds()[symbol.get_dims(), :]

                        x_low = sym_limits[0, 0]
                        x_high = sym_limits[0, 1]
                        y_low = sym_limits[1, 0]
                        y_high = sym_limits[1, 1]
                        ax.add_patch(Rectangle((x_low, y_low), x_high - x_low, y_high - y_low,
                                               edgecolor='black',
                                               facecolor=symbol.get_color(),
                                               fill=False,
                                               lw=1))
                    if symbol.get_type() == 'circle':
                        ax.add_patch(Circle(symbol.get_center(), symbol.get_radius(),
                                            edgecolor='black',
                                            facecolor=symbol.get_color(),
                                            fill=False,
                                            lw=1))

    for data in data_trajectories:
        if data.shape[1] == 6:
            # Plot stretch example
            ax.plot(data[:, 0], data[:, 1], np.zeros(data.shape[0]), **kwargs)
            for d in data:
                # ax.plot([d[0], d[0] + 0.01], [d[1], d[1] + 0.01], **kwargs)
                l_eff = 0.1
                x_robot = d[0]
                y_robot = d[1]
                t_robot = d[2]
                l_wrist = d[3]
                z_wrist = d[4]
                t_wrist = d[5]
                x_ee = l_eff * np.cos(t_robot + t_wrist) + l_wrist * np.sin(t_robot) + x_robot
                y_ee = l_eff * np.sin(t_robot + t_wrist) - l_wrist * np.cos(t_robot) + y_robot
                z_ee = z_wrist
                ax.plot(x_ee, y_ee, z_ee, '.', color='yellow')
        elif limits.shape[0] == 3:
            # if np.any(data[:, 2] < -0.18):
            #     print("here")
            # data = np.loadtxt(folder_trajectories + "/" + f, delimiter=" ", dtype=float)
            if three_d_one_plot:
                ax.plot(data[:, 0], data[:, 1], data[:, 2], **kwargs)
            else:
                ax[0, 0].plot(data[:, 0], data[:, 1], data[:, 2], **kwargs)
                ax[0, 1].plot(data[:, 1], data[:, 2], **kwargs)
                ax[1, 0].plot(data[:, 0], data[:, 2], **kwargs)
                ax[1, 1].plot(data[:, 0], data[:, 1], **kwargs)
        elif limits.shape[0] == 2:
            ax.plot(data[:, 0], data[:, 1], **kwargs)


    plt.tight_layout()
    # ax.set_xticks([-2, -1, 0, 1, 2])
    # ax.set_yticks([-2, -1, 0, 1])
    return ax


# def plot_symbols(symbols: Dict, limits: ndarray, folder_save: str):
def plot_symbols(symbols, limits, folder_save):
    os.makedirs(folder_save, exist_ok=True)
    for symbol in symbols.values():
        _ = plot_symbol(symbol, limits, alpha=0.3, color=symbol.get_color())
        plt.savefig(folder_save + "/" + symbol.get_name() + ".png")


# def plot_symbol(symbol: Dict, limits: ndarray, ax=None, **kwargs) -> plt.Axes:
def plot_symbol(symbol, limits, ax=None, **kwargs):
    if ax is None:
        if limits.shape[0] == 3:
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        elif limits.shape[0] == 2:
            _, ax = plt.subplots()

    plt.ion()
    sym_limits = copy.deepcopy(limits)
    # sym_limits[symbol.get_dims(), :] = symbol.get_bnds()

    if limits.shape[0] == 3 and symbol.get_type() == 'rectangle':
        sym_limits[symbol.get_dims(), :] = symbol.get_bnds()[symbol.get_dims(), :]

        x_low = sym_limits[0, 0]
        x_high = sym_limits[0, 1]
        y_low = sym_limits[1, 0]
        y_high = sym_limits[1, 1]
        z_low = sym_limits[2, 0]
        z_high = sym_limits[2, 1]

        x = np.array(
            [
                [x_high, x_low, x_low, x_high, x_high],
                [x_high, x_low, x_low, x_high, x_high],
                [x_low, x_high, x_high, x_low, x_low],
                [x_low, x_high, x_high, x_low, x_low],
                [x_high, x_low, x_low, x_high, x_high],
            ]
        )

        y = np.array(
            [
                [y_high, y_high, y_low, y_low, y_high],
                [y_high, y_high, y_low, y_low, y_high],
                [y_low, y_low, y_high, y_high, y_low],
                [y_low, y_low, y_high, y_high, y_low],
                [y_high, y_high, y_low, y_low, y_high],
            ]
        )

        z = np.array(
            [
                [z_high, z_high, z_high, z_high, z_high],
                [z_low, z_low, z_low, z_low, z_low],
                [z_low, z_low, z_low, z_low, z_low],
                [z_high, z_high, z_high, z_high, z_high],
                [z_high, z_high, z_high, z_high, z_high],
            ]
        )

        ax.plot_surface(x, y, z, **kwargs)
        ax.set_xlim(limits[0, :])
        ax.set_xlabel('x')
        ax.set_ylim(limits[1, :])
        ax.set_ylabel('y')
        ax.set_zlim(limits[2, :])
        ax.set_zlabel('z')
    elif limits.shape[0] == 2 and symbol.get_type() == 'rectangle':
        sym_limits[symbol.get_dims(), :] = symbol.get_bnds()[symbol.get_dims(), :]

        x_low = sym_limits[0, 0]
        x_high = sym_limits[0, 1]
        y_low = sym_limits[1, 0]
        y_high = sym_limits[1, 1]
        ax.add_patch(Rectangle((x_low, y_low), x_high - x_low, y_high - y_low,
                               edgecolor='black',
                               facecolor=symbol.get_color(),
                               fill=False,
                               lw=1))
        ax.set_xlim(limits[0, :])
        ax.set_ylim(limits[1, :])
        ax.set_title(symbol.get_name())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    elif limits.shape[0] == 2 and symbol.get_type() == 'circle':
        ax.add_patch(Circle(symbol.get_center(), symbol.get_radius(),
                            edgecolor='black',
                            facecolor=symbol.get_color(),
                            fill=False,
                            lw=1))
        # ax.set_xlim([symbol.get_center()[0] - symbol.get_radius(), symbol.get_center()[0] + symbol.get_radius()])
        # ax.set_ylim([symbol.get_center()[1] - symbol.get_radius(), symbol.get_center()[1] + symbol.get_radius()])
        ax.set_xlim(limits[0, :])
        ax.set_ylim(limits[1, :])
        ax.set_title(symbol.get_name())
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    plt.tight_layout()

    return ax


# def plot_one_traj_vs_time(skill: str, arg_data: ndarray, arg_start_time: float, arg_end_time: float, arg_mean: float,
#                           arg_std: float, arg_n_stds: float, ax: plt.Axes = None, **kwargs):
def plot_one_traj_vs_time(skill, arg_data, arg_start_time, arg_end_time, arg_mean,
                          arg_std, arg_n_stds, ax=None, **kwargs):
    if ax is None:
        if arg_data.shape[1] == 8:
            fig, ax = plt.subplots(nrows=arg_data.shape[1] - 1, figsize=(15,15))
        elif arg_data.shape[1] == 3:
            fig, ax = plt.subplots(nrows=3, figsize=(5,5))
        elif arg_data.shape[1] == 2:
            fig, ax = plt.subplots(nrows=2, figsize=(5,5))

    ylabels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    if arg_data.shape[1] == 8:
        for ii in range(0, arg_data.shape[1] - 1):
            ax[ii].plot(arg_data[:, 0], arg_data[:, ii + 1], **kwargs)
            ylims = ax[ii].get_ylim()
            ax[ii].plot([arg_start_time, arg_start_time], ylims)
            ax[ii].plot([arg_end_time, arg_end_time], ylims)
            if ii == 2:
                xlims = ax[ii].get_xlim()
                ax[ii].plot(xlims, [arg_mean, arg_mean])
                ax[ii].add_patch(Rectangle((xlims[0], arg_mean - arg_std * arg_n_stds),
                                           xlims[1] - xlims[0],
                                           arg_std * arg_n_stds * 2,
                                           edgecolor='black',
                                           facecolor='blue',
                                           fill=True,
                                           lw=1))
            ax[ii].set_ylabel(ylabels[ii])
    else:
        for ii in range(0, arg_data.shape[1]):
            ax[ii].plot(np.arange(arg_data.shape[0]), arg_data[:, ii], **kwargs)
            ylims = ax[ii].get_ylim()
            ax[ii].set_ylabel(ylabels[ii])

    ax[-1].set_xlabel('Time (ms)')
    plt.suptitle(skill)
    plt.tight_layout()

    return ax
