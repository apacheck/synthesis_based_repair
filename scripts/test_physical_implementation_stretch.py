#!/usr/bin/env python
import importlib as importlib

from synthesis_based_repair.tools import write_spec, clear_file, dict_to_formula, json_load_wrapper
import matplotlib.pyplot as plt
import copy
import numpy as np
from synthesis_based_repair.symbols import load_symbols
from synthesis_based_repair.skills import load_skills_from_json, Skill, write_skills_str, find_traj_in_syms
import json
from synthesis_based_repair.physical_implementation import learn_skill_with_constraints, fk_stretch, create_stretch_base_traj, symbols_and_workspace_to_device
import argparse
from synthesis_based_repair.visualization import plot_sat_unsat_trajectories, create_ax_array
from dl2_lfd.elaborateDMP import evaluate_constraint
from dl2_lfd.dmps.dmp import load_dmp_demos, DMP
from dl2_lfd.helper_funcs.conversions import np_to_pgpu
from torch.utils.data import TensorDataset, DataLoader
from dl2_lfd.ltl_diff.constraints import AlwaysFormula, EventuallyOrFormulas, AndEventuallyFormulas, SequenceFormulas


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

    fid = open(file_names["file_suggestions"], 'r')
    suggestions = json.load(fid)
    suggestion = suggestions['0']
    fid.close()

    iteration_count = 0

    prev_skill_name = dict_to_formula(suggestion['original_skill'], include_false=False)
    old_demo_folder = folder_trajectories + prev_skill_name + "/"
    dmp_opts['skill_name'] = dict_to_formula(suggestion['new_skill'], include_false=False) + "_" + str(iteration_count) + "_new"
    demo_folder = folder_trajectories + dmp_opts['skill_name']

    path_to_original_model = "/home/adam/repos/synthesis_based_repair/data/dmps/" + prev_skill_name + ".pt"

    ####################################################################################################################
    # Here we make sure that a sample skill plots the trajectory properly and goes through the symbols we expect       #
    # visually
    ####################################################################################################################
    v_start_states, v_pose_hists = load_dmp_demos(old_demo_folder + "/val", n_points=int(1/dmp_opts['dt']))
    v_start_states, v_pose_hists = np_to_pgpu(v_start_states), np_to_pgpu(v_pose_hists)
    val_set = TensorDataset(v_start_states, v_pose_hists)
    # losses, learned_rollouts, c_sat = evaluate_constraint(val_set, None, path_to_original_model,
    #                                                       basis_fs=dmp_opts['basis_fs'], dt=dmp_opts['dt'],
    #                                                       n_epochs=dmp_opts['n_epochs'], output_dimension=dmp_opts['dimension'])
    # symbols_to_plot = ['ee_table_1', 'ee_table_1a', 'ee_table_1b', 'ee_table_2', 'ee_table_3', 'base_1', 'base_2', 'base_3'] #, 'duck_a_held', 'duck_a_table']
    # trajectories_ee = fk_stretch(learned_rollouts)
    # trajectories_base = create_stretch_base_traj(learned_rollouts)
    # fig, ax = create_ax_array(3)
    # plot_sat_unsat_trajectories(trajectories_ee, c_sat, ax[1], ax[2])
    # plot_sat_unsat_trajectories(trajectories_base, c_sat, ax[1], ax[2])
    # for sym in symbols_to_plot:
    #     symbols[sym].plot(ax[0], dim=3)

    ####################################################################################################################
    # Here we make sure that a sample skill satisfies a constraint that it should
    ####################################################################################################################
    formula = [{'ee_table_1': True, 'ee_table_1a': False, 'ee_table_1b': False, 'ee_table_2': False, 'base_1': True},
               {'ee_table_1': False, 'ee_table_1a': True, 'ee_table_1b': False, 'ee_table_2': False},
               {'ee_table_1': False, 'ee_table_1a': False, 'ee_table_1b': True, 'ee_table_2': False},
               {'ee_table_1': False, 'ee_table_1a': False, 'ee_table_1b': False, 'ee_table_2': False},
               {'ee_table_1': False, 'ee_table_1a': False, 'ee_table_1b': False, 'ee_table_2': True}
               ]
    symbols_device, _ = symbols_and_workspace_to_device(symbols, None)
    # constraint = AlwaysFormula(symbols_device, formula, epsilon=dmp_opts['epsilon'])
    # constraint = EventuallyOrFormulas(symbols_device, formula, epsilon=dmp_opts['epsilon'])
    # constraint = AndEventuallyFormulas(symbols_device, formula, epsilon=dmp_opts['epsilon'])
    constraint = SequenceFormulas(symbols_device, formula, epsilon=dmp_opts['epsilon'])
    losses, learned_rollouts, c_sat = evaluate_constraint(val_set, constraint, path_to_original_model, basis_fs=dmp_opts['basis_fs'],
                        dt=dmp_opts['dt'], n_epochs=dmp_opts['n_epochs'], output_dimension=dmp_opts['dimension'])
    symbols_to_plot = ['ee_table_1', 'ee_table_1a', 'ee_table_1b', 'ee_table_2', 'ee_table_3', 'base_1', 'base_2', 'base_3'] #, 'duck_a_held', 'duck_a_table']
    trajectories_ee = fk_stretch(learned_rollouts)
    trajectories_base = create_stretch_base_traj(learned_rollouts)
    fig, ax = create_ax_array(3, ncols=2)
    plot_sat_unsat_trajectories(trajectories_ee, c_sat, ax[0], ax[1])
    plot_sat_unsat_trajectories(trajectories_base, c_sat, ax[0], ax[1])
    for sym in symbols_to_plot:
        symbols[sym].plot(ax[0], dim=3)
        symbols[sym].plot(ax[1], dim=3)

    ####################################################
    # Calculates which propositions the robot passes through
    #####################
    trajectories_base_and_ee = np.stack([trajectories_base, trajectories_ee])
    for trajectory in trajectories_base_and_ee:
        traj_in_syms = find_traj_in_syms(trajectory, symbols)
        reduced_traj = reduce_sym_traj(traj_in_syms)
        print(traj_in_syms)

    ###################
    # Attempt to make the model obey a new constraint
    ############
    base_folder = '../data'
    output_model_path = '../data/dmps' + dmp_opts['skill_name']
    learned_model, results_folder = learn_skill_with_constraints(dmp_opts['skill_name'], constraint,
                              base_folder, demo_folder, old_demo_folder=old_demo_folder, previous_model_path=path_to_original_model,
                                enforce_type="train", main_loss_weight=dmp_opts['m_weight'], constraint_loss_weight=dmp_opts['c_weight'], basis_fs=dmp_opts['basis_fs'],
                                dt=dmp_opts['dt'], n_epochs=dmp_opts['n_epochs'], output_dimension=dmp_opts['dimension'], epsilon=dmp_opts['epsilon'], output_model_path=output_model_path)

    losses, learned_rollouts, c_sat = evaluate_constraint(val_set, constraint, output_model_path, basis_fs=dmp_opts['basis_fs'],
                        dt=dmp_opts['dt'], n_epochs=dmp_opts['n_epochs'], output_dimension=dmp_opts['dimension'])
    symbols_to_plot = ['ee_table_1', 'ee_table_1a', 'ee_table_1b', 'ee_table_2', 'ee_table_3', 'base_1', 'base_2', 'base_3'] #, 'duck_a_held', 'duck_a_table']
    trajectories_ee = fk_stretch(learned_rollouts)
    trajectories_base = create_stretch_base_traj(learned_rollouts)
    fig, ax = create_ax_array(3, ncols=2)
    plot_sat_unsat_trajectories(trajectories_ee, c_sat, ax[0], ax[1])
    plot_sat_unsat_trajectories(trajectories_base, c_sat, ax[0], ax[1])
    for sym in symbols_to_plot:
        symbols[sym].plot(ax[0], dim=3)
        symbols[sym].plot(ax[1], dim=3)
    plt.show()



