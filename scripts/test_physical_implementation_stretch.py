#!/usr/bin/env python
import importlib as importlib

from synthesis_based_repair.tools import write_spec, clear_file, dict_to_formula, json_load_wrapper
import matplotlib.pyplot as plt
import copy
import numpy as np
from synthesis_based_repair.symbols import load_symbols
from synthesis_based_repair.skills import load_skills_from_json, Skill, write_skills_str
import json
from synthesis_based_repair.physical_implementation import run_elaborateDMP #, spoof_elaborateDMP
import argparse


if __name__ == "__main__":

    plt.close('all')
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_spec", help="Name of json file with user spec", required=True)
    parser.add_argument("--file_names", help="File names", required=True)
    parser.add_argument("--sym_opts", help="Opts involving spec writing and repair", required=True)
    parser.add_argument("--dmp_opts", help="Opts dmps", required=True)
    parser.add_argument("--n_seeds", help="Number of seeds to run", required=False, default=1, type=int)
    parser.add_argument("--loss_threshold", help="Loss threshold to accept change", required=False, default=0.9, type=float)
    args = parser.parse_args()

    file_names = json_load_wrapper(args.file_names)
    sym_opts = json_load_wrapper(args.sym_opts)
    clear_file(file_names['file_log'])
    user_spec = json_load_wrapper(args.user_spec)
    dmp_opts = json_load_wrapper(args.dmp_opts)
    dmp_opts['plot_limits'] = np.array(dmp_opts["plot_limits"])
    dmp_opts['workspace_bnds'] = np.array(dmp_opts["workspace_bnds"])

    loss_threshold = args.loss_threshold

    # Unpack some variables
    file_structured_slugs = file_names['file_structured_slugs']
    folder_trajectories = dmp_opts['demo_folder']
    change_cons = user_spec['change_cons']
    not_allowed_repair = user_spec['not_allowed_repair']
    file_log = file_names['file_log']
    folder_plot = file_names['folder_plot']
    post_first = True
    workspace_bnds = dmp_opts["workspace_bnds"]

    ############################
    # Load skills and symbols ##
    ############################
    symbols = load_symbols(file_names["file_symbols"])
    dmp_opts['symbols'] = symbols
    skills_all = load_skills_from_json(file_names["file_skills"])
    original_skills = dict()
    for skill_name in file_names["skill_names"]:
        original_skills[skill_name] = skills_all[skill_name]

    for skill_name in file_names["skill_names"]:
        original_skills[skill_name + "b"] = copy.deepcopy(original_skills[skill_name])

    fid = open(file_names["file_suggestions"], 'r')
    suggestion = json.load(fid)
    fid.close()

    iteration_count = 0

    dmp_opts['demo_folder'] = folder_trajectories + dict_to_formula(suggestion['original_skill'],
                                                                           include_false=False) + "_" + str(
        iteration_count) + "_new" + "/"
    dmp_opts['previous_skill_name'] = dict_to_formula(suggestion['original_skill'], include_false=False)
    dmp_opts['skill_name'] = dict_to_formula(suggestion['new_skill'], include_false=False) + "_" + str(
        iteration_count) + "_new"
    _, val_losses, int_sat = run_elaborateDMP(dmp_opts['previous_skill_name'],
                                                        dmp_opts['skill_name'],
                                                        suggestion, user_spec['hard_constraints'],
                                                        symbols, workspace_bnds, dmp_opts)

    print("Val_losses: {}\nInt_sat: {}".format(val_losses, int_sat))




