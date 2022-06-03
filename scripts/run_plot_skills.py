#!/usr/bin/env python
from synthesis_based_repair.tools import write_spec, json_load_wrapper
from synthesis_based_repair.symbols import load_symbols
from synthesis_based_repair.skills import load_skills_from_trajectories, write_skills_json
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
    parser.add_argument("--create_json", help="Create a json file at the same time?", action='store_true', default=False)
    args = parser.parse_args()

    file_names = json_load_wrapper(args.file_names)
    sym_opts = json_load_wrapper(args.sym_opts)
    dmp_opts = json_load_wrapper(args.dmp_opts)
    folder_trajectories = file_names["folder_trajectories"]

    ############################
    # Load skills and plot ####
    ############################
    symbols = load_symbols(file_names["file_symbols"])
    os.makedirs(file_names["folder_plot"], exist_ok=True)

    skills = load_skills_from_trajectories(folder_trajectories, file_names["skill_names"], symbols)
    dim = len(dmp_opts["plot_limits"])
    for skill_name, skill in skills.items():
        fig = plt.figure()
        if dim == 3:
            ax = plt.axes(projection="3d")
        elif dim == 2:
            ax = plt.axes()
        skill.plot_original(ax)
        ax.set_xlim(dmp_opts["plot_limits"][0])
        ax.set_ylim(dmp_opts["plot_limits"][1])
        if dim == 3:
            ax.set_zlim(dmp_opts["plot_limits"][2])
        plt.savefig(file_names["folder_plot"] + skill_name + ".png")
        plt.close()
    if args.create_json:
        write_skills_json(skills, file_names['file_skills'])

    ##########################################################
    # Plot one skill trajectory real nice (for the stretch) ##
    ##########################################################
    if dim == 3:
        for skill_name, skill in skills.items():
            fig = plt.figure()
            if dim == 3:
                ax = plt.axes(projection="3d")
            elif dim == 2:
                ax = plt.axes()
            skill.plot_nice(ax, dmp_opts["plot_limits"], symbols)
            ax.set_xlim(dmp_opts["plot_limits"][0])
            ax.set_ylim(dmp_opts["plot_limits"][1])
            if dim == 3:
                ax.set_zlim(dmp_opts["plot_limits"][2])
            plt.savefig(file_names["folder_plot"] + skill_name + "_nice.png")
            plt.close()
