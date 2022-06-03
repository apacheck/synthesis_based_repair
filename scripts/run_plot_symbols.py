#!/usr/bin/env python
from synthesis_based_repair.tools import write_spec, json_load_wrapper
from synthesis_based_repair.symbols import load_symbols
import json
from synthesis_based_repair.skills import load_skills_from_json
import argparse
import copy
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_names", help="File names", required=True)
    parser.add_argument("--sym_opts", help="Opts involving spec writing and repair", required=True)
    parser.add_argument("--dmp_opts", help="Opts involving plotting, repair, dmps", required=True)
    args = parser.parse_args()

    file_names = json_load_wrapper(args.file_names)
    sym_opts = json_load_wrapper(args.sym_opts)
    dmp_opts = json_load_wrapper(args.dmp_opts)

    ############################
    # Load symbols and plot ####
    ############################
    symbols = load_symbols(file_names["file_symbols"])
    os.makedirs(file_names["folder_plot"], exist_ok=True)
    f_plot = file_names["folder_plot"] + "symbols.png"

    # Plot symbols together on the same figure
    dim = len(dmp_opts["plot_limits"])
    fig = plt.figure()
    if dim == 3:
        ax = plt.axes(projection="3d")
    elif dim == 2:
        ax = plt.axes()
    for sym_name, sym in symbols.items():
        sym.plot(ax, dim=dim, fill=True, lw=1, alpha=0.4)
    ax.set_xlim(dmp_opts["plot_limits"][0])
    ax.set_ylim(dmp_opts["plot_limits"][1])
    if dim == 3:
        ax.set_zlim(dmp_opts["plot_limits"][2])
    # ax.set_xticks([0, 1, 2, 3])
    # ax.set_yticks([0, 1, 2, 3])
    plt.savefig(f_plot)

    # Plot symbols individually
    for sym_name, sym in symbols.items():
        f_plot = file_names["folder_plot"] + sym_name + ".png"

        dim = len(dmp_opts["plot_limits"])
        fig = plt.figure()
        if dim == 3:
            ax = plt.axes(projection="3d")
        elif dim == 2:
            ax = plt.axes()
        sym.plot(ax, dim=dim, fill=True, lw=1, alpha=0.4)
        ax.set_xlim(dmp_opts["plot_limits"][0])
        ax.set_ylim(dmp_opts["plot_limits"][1])
        if dim == 3:
            ax.set_zlim(dmp_opts["plot_limits"][2])
        # ax.set_xticks([0, 1, 2, 3])
        # ax.set_yticks([0, 1, 2, 3])
        plt.savefig(f_plot)
