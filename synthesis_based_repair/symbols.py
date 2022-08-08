#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import json
import copy
import os
from synthesis_based_repair.visualization import plot_cube


class Symbol:

    def __init__(self, info):
        self.name = info['name']
        self.type = info['type']
        self.factor = info['factor']
        self.color = info['color']
        self.index = info['index']
        self.dims = np.array(info['dims'])
        if 'plot_dims' in info.keys():
            self.plot_dims = info['plot_dims']
        self.transform = None
        if "transform" in info.keys():
            self.transform = info["transform"]
        if self.type == 'rectangle' or self.type == 'rectangle-ee':
            self.bounds = np.array(info['bounds'])
            self.center = np.zeros([2])
            self.radius = np.zeros([1])
        elif self.type == 'circle' or self.type == 'circle-ee':
            self.bounds = np.zeros([2, 2])
            self.center = np.array(info['center'])
            self.radius = np.array(info['radius'])

    def in_symbol(self, points):
        if points.ndim == 1:
            if self.type == 'rectangle' or (self.type == 'rectangle-ee' and points.shape[0] == 2):
                bnd_low = self.bounds[:, 0]
                bnd_high = self.bounds[:, 1]
                if points.shape[0] == bnd_low.shape[0] and np.all(bnd_low < points[:]) and np.all(points[:] < bnd_high):
                    return True
                elif points.shape[0] != bnd_low.shape[0] and np.all(bnd_low < points[self.dims]) and np.all(points[self.dims] < bnd_high):
                    return True
                else:
                    return False
            elif self.type == 'circle' or (self.type == 'circle-ee' and points.shape[0] == 2):
                if np.sqrt(np.sum(np.square(self.center - points[:]))) < self.radius:
                    return True
                else:
                    return False
            elif self.type == 'rectangle-ee' or self.type == 'circle-ee':
                l_eff = 0.1
                x_robot = points[0]
                y_robot = points[1]
                t_robot = points[2]
                l_wrist = points[3]
                z_wrist = points[4]
                t_wrist = points[5]
                x_ee = l_eff * np.cos(t_robot + t_wrist) + l_wrist * np.sin(t_robot) + x_robot
                y_ee = l_eff * np.sin(t_robot + t_wrist) - l_wrist * np.cos(t_robot) + y_robot
                points_eval = np.array([x_ee, y_ee])
                if self.type == 'rectangle-ee':
                    bnd_low = self.bounds[self.dims, 0]
                    bnd_high = self.bounds[self.dims, 1]
                    if np.all(bnd_low < points_eval) and np.all(points_eval < bnd_high):
                        return True
                    else:
                        return False
                elif self.type == 'circle-ee':
                    if np.sqrt(np.sum(np.square(self.center - points_eval))) < self.radius:
                        return True
                    else:
                        return False
        else:
            out = np.zeros(points.shape[0], dtype=bool)
            for ii, point in enumerate(points):
                out[ii] = self.in_symbol(point)
            return out

    def get_edges(self):
        if self.type == 'rectangle' or self.type == 'rectangle-ee':
            if self.bounds.shape[0] == 2:
                edges = np.hstack([np.vstack(
                    [np.linspace(self.bounds[0, 0], self.bounds[0, 1], 25), np.repeat(self.bounds[1, 0], 25)]),
                                    np.vstack([np.linspace(self.bounds[0, 0], self.bounds[0, 1], 25),
                                               np.repeat(self.bounds[1, 1], 25)]),
                                    np.vstack([np.repeat(self.bounds[0, 0], 25),
                                               np.linspace(self.bounds[1, 0], self.bounds[1, 1], 25)]),
                                    np.vstack([np.repeat(self.bounds[0, 1], 25),
                                               np.linspace(self.bounds[1, 0], self.bounds[1, 1], 25)])]).transpose()
            elif self.bounds.shape[0] == 1:
                edges = np.array([[self.bounds[0, 0]], [self.bounds[0, 1]]])
        elif self.type == 'circle' or self.type == 'circle-ee':
            edges = np.vstack([self.center[0] + self.radius * np.cos(np.linspace(0, 2 * np.pi, 100)),
                                self.center[1] + self.radius * np.sin(
                                    np.linspace(0, 2 * np.pi, 100))]).transpose()

        return edges

    def sample_from(self):
        if self.type == 'rectangle' or self.type == 'rectangle-ee':
            sample = self.bounds[:, 0] + np.random.random(self.bounds.shape[0]) * (self.bounds[:, 1] - self.bounds[:, 0])
            sample = sample.transpose()
        elif self.type == 'circle' or self.type == 'circle-ee':
            theta = 2 * np.pi * np.random.random(1)
            sample = np.array([self.center[0] + self.radius * np.cos(theta), self.center[1] + self.radius * np.sin(theta)])

        return sample

    def get_plot_bnds(self, ax):
        if 0 in self.plot_dims:
            idx = self.plot_dims.index(0)
            x_low = self.bounds[idx, 0]
            x_high = self.bounds[idx, 1]
        else:
            x_lim = ax.get_xlim()
            x_low = x_lim[0]
            x_high = x_lim[1]

        if 1 in self.plot_dims:
            idx = self.plot_dims.index(1)
            y_low = self.bounds[idx, 0]
            y_high = self.bounds[idx, 1]
        else:
            y_lim = ax.get_ylim()
            y_low = y_lim[0]
            y_high = y_lim[1]
        if 2 in self.plot_dims:
            idx = self.plot_dims.index(2)
            z_low = self.bounds[idx, 0]
            z_high = self.bounds[idx, 1]
        else:
            if ax.name == "3d":
                z_lim = ax.get_zlim()
                z_low = z_lim[0]
                z_high = z_lim[1]
            else:
                z_low = np.nan
                z_high = np.nan

        return np.array([[x_low, x_high], [y_low, y_high], [z_low, z_high]])

    def plot(self, ax, dim=2, **kwargs):
        """
        Plots the symbol on the provided axis. The user supplies the dimensionality of how the symbol should be plotted.
        If a 2 d symbol is plotted in 3d, the symbol will cover the entire z axis.

        :param ax:
        :param dim:
        :param kwargs:
        :return:
        """
        if 'alpha' not in kwargs.keys():
            kwargs["alpha"] = 0.2
        if dim == 2:
            if self.type == 'rectangle':
                plot_bnds = self.get_plot_bnds(ax)
                x_low = plot_bnds[0, 0]
                x_high = plot_bnds[0, 1]
                y_low = plot_bnds[1, 0]
                y_high = plot_bnds[1, 1]
                ax.add_patch(Rectangle((x_low, y_low), x_high - x_low, y_high - y_low,
                                       edgecolor='black',
                                       facecolor=self.color,
                                       **kwargs))
            if self.type == 'circle':
                ax.add_patch(Circle(self.center, self.radius,
                                    edgecolor='black',
                                    facecolor=self.color
                                    **kwargs))
        elif dim == 3:
            plot_bnds = self.get_plot_bnds(ax)
            x_low = plot_bnds[0, 0]
            x_high = plot_bnds[0, 1]
            y_low = plot_bnds[1, 0]
            y_high = plot_bnds[1, 1]
            z_low = plot_bnds[2, 0]
            z_high = plot_bnds[2, 1]

            plot_cube(ax, x_low, x_high, y_low, y_high, z_low, z_high, **kwargs)

    def get_factor(self):
        return self.factor

    def get_index(self):
        return self.index

    def get_type(self):
        return self.type

    def get_dims(self):
        return self.dims

    def get_plot_dims(self):
        return self.plot_dims

    def get_bnds(self):
        return self.bounds

    def get_center(self):
        return self.center

    def get_radius(self):
        return self.radius

    def get_color(self):
        return self.color


def plot_symbolic_state(state, symbols, ax, xlims, ylims, **kwargs):
    for symbol_name, truth in state.items():
        if truth:
            symbols[symbol_name].plot(ax, **kwargs)

    for symbol_name, truth in state.items():
        if not truth:
            sym_tmp = copy.deepcopy(symbols[symbol_name])
            sym_tmp.color = 'white'
            sym_tmp.plot(ax, **kwargs)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)


def symbols_intersect(symbol_one, symbol_two):

    edges_one = symbol_one.get_edges()
    edges_two = symbol_two.get_edges()

    symbol_one_intersects_two = np.any(symbol_two.in_symbol(edges_one))
    symbol_two_intersects_one = np.any(symbol_one.in_symbol(edges_two))

    return symbol_one_intersects_two or symbol_two_intersects_one


def load_symbols(file_symbols):
    fid = open(file_symbols, 'r')
    data = json.load(fid)
    fid.close()
    symbols_out = dict()
    for sym_name, sym_data in data.items():
        symbols_out[sym_name] = Symbol(sym_data)

    return symbols_out


def in_symbols(point, syms_dict, sym_defs):
    in_true_syms = True
    in_false_syms = False
    for sym, truth in syms_dict.items():
        in_sym = sym_defs[sym].in_sym(point)
        if truth:
            in_true_syms = in_true_syms and in_sym
        else:
            in_false_syms = in_false_syms or in_sym

    return in_true_syms and not in_false_syms


def find_symbols_true_and_false(point, sym_defs):
    sym_dict_out = dict()
    for sym_name, sym_def in sym_defs.items():
        sym_dict_out[sym_name] = sym_def.in_symbol(point)

    return sym_dict_out


def find_symbols_by_var(symbols, n_factors):

    syms_by_var = []
    for ii in range(n_factors):
        syms_by_var.append([])
    for symbol_name, symbol in symbols.items():
        syms_by_var[symbol.get_factor()].append(symbol_name)

    return syms_by_var


if __name__ == "__main__":
    f_symbols = "../data/nine_squares/nine_squares_symbols.json"
    os.makedirs("../data/nine_squares/plots/", exist_ok=True)
    f_plot = "../data/nine_squares/plots/nine_squares_symbols.png"
    symbols = load_symbols(f_symbols)
    fig, ax = plt.subplots()
    for sym_name, sym in symbols.items():
        sym.plot(ax, fill=True, lw=1, alpha=0.4)
    ax.set_xlim([-0.5, 3.5])
    ax.set_ylim([-0.5, 3.5])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    plt.savefig(f_plot)
