#!/usr/bin/env python
import importlib as importlib

from synthesis_based_repair.tools import write_spec, clear_file, dict_to_formula, json_load_wrapper
import matplotlib.pyplot as plt
import copy
import numpy as np
from synthesis_based_repair.symbols import load_symbols
from synthesis_based_repair.skills import load_skills_from_json, Skill, write_skills_str, find_traj_in_syms, reduce_sym_traj, find_one_skill_intermediate_states, find_unique_states, remove_mutually_exclusive_symbols_list_of_states, remove_mutually_exclusive_symbols_intermediate_states, remove_mutually_exclusive_symbols_one_state
import json
from synthesis_based_repair.physical_implementation import learn_skill_with_constraints, fk_stretch, create_stretch_base_traj, symbols_and_workspace_to_device
import argparse
from synthesis_based_repair.visualization import plot_sat_unsat_trajectories, create_ax_array, load_intermediate_data
from dl2_lfd.elaborateDMP import evaluate_constraint, evaluate_model
from dl2_lfd.dmps.dmp import load_dmp_demos, DMP
from dl2_lfd.helper_funcs.conversions import np_to_pgpu
from torch.utils.data import TensorDataset, DataLoader
from dl2_lfd.ltl_diff.constraints import AlwaysFormula, EventuallyOrFormulas, AndEventuallyFormulas, SequenceFormulas, SkillConstraint, SequenceFormulasAndAlways, SequenceFormulasMultiplePostsAndAlways
from os.path import join


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
    losses, learned_rollouts, c_sat = evaluate_constraint(val_set, None, path_to_original_model,
                                                          basis_fs=dmp_opts['basis_fs'], dt=dmp_opts['dt'],
                                                          output_dimension=dmp_opts['dimension'])
    symbols_to_plot = ['base_B', 'base_C', 'base_F', 'base_G', 'ee_A', 'ee_B', 'ee_C', 'ee_E', 'ee_F', 'ee_G', 'ee_H']
    trajectories_ee = learned_rollouts[:, :, 2:]
    trajectories_base = create_stretch_base_traj(learned_rollouts)
    fig, ax = create_ax_array(3)
    plot_sat_unsat_trajectories(trajectories_ee, c_sat, ax[1], ax[2])
    plot_sat_unsat_trajectories(trajectories_base, c_sat, ax[1], ax[2])
    for sym in symbols_to_plot:
        symbols[sym].plot(ax[1], dim=3, alpha=0.05)


    ####################################################
    # Calculates which propositions the robot passes through
    #####################
    trajectories_base_and_ee = np.zeros([trajectories_base.shape[0], trajectories_base.shape[1], 5])
    trajectories_base_and_ee[:, :, :2] = trajectories_base[:, :, :2]
    trajectories_base_and_ee[:, :, 2:] = trajectories_ee
    # symbols_to_extract = ['ee_table_1', 'ee_table_1a', 'ee_table_1b', 'ee_table_2', 'ee_table_3', 'base_1', 'base_2',
                       # 'base_3']

    unique_traj = []
    for trajectory in trajectories_base_and_ee:
        traj_in_syms = find_traj_in_syms(trajectory, symbols)
        reduced_traj = reduce_sym_traj(traj_in_syms)
        if reduced_traj not in unique_traj:
            unique_traj.append(reduced_traj)
    for u in unique_traj:
        for s in u:
            print(remove_mutually_exclusive_symbols_one_state(s, symbols))
        print("*************")

    intermediate_states = find_one_skill_intermediate_states(old_demo_folder + "/val", symbols)
    intermediate_states = remove_mutually_exclusive_symbols_intermediate_states(intermediate_states, symbols)
    unique_states = find_unique_states(old_demo_folder + "/val", symbols)
    unique_states = remove_mutually_exclusive_symbols_list_of_states(unique_states, symbols)


    ####################################################################################################################
    # Here we make sure that a sample skill satisfies a constraint that it should
    ####################################################################################################################
    # Constraint is that ee and base are not both in E
    # bad = {'base_E': True, 'ee_B': True}
    # unique_states.remove(bad)
    # unique_states.remove({'base_F': True, 'ee_B': True})
    # unique_states.append({'base_F': True, 'ee_F': True})
    # unique_states.append({'base_F': True, 'ee_E': True})
    # for ii, (pre, posts) in enumerate(intermediate_states):
    #     if pre == bad:
    #         del intermediate_states[ii]
    #     if bad in posts:
    #         intermediate_states[ii][1].remove(bad)
    # intermediate_states[0][1] = [{'base_F': True, 'ee_F': True}]
    # intermediate_states[1][0] = {'base_F': True, 'ee_F': True}
    # intermediate_states[1][1] = [{'base_F': True, 'ee_E': True}]
    # intermediate_states.append([{'base_F': True, 'ee_E': True}, [{'base_E': True, 'ee_E': True}]])
    intermediate_states = [[{'base_C': True, 'base_G': True, 'ee_C': True, 'ee_H': True}, [{'base_C': True, 'base_G': True, 'ee_C': True, 'ee_G': True}]],
                           [{'base_C': True, 'base_G': True, 'ee_C': True, 'ee_G': True}, [{'base_C': True, 'base_G': True, 'ee_B': True, 'ee_G': True}]],
                           [{'base_C': True, 'base_G': True, 'ee_B': True, 'ee_G': True}, [{'base_B': True, 'base_G': True, 'ee_B': True, 'ee_G': True}]],
                           [{'base_B': True, 'base_G': True, 'ee_B': True, 'ee_G': True}, [{'base_B': True, 'base_G': True, 'ee_A': True, 'ee_G': True}]]]
    print("Intermiate states")
    for pre, posts in intermediate_states:
        print("pre: ", pre)
        for post in posts:
            print("post: ", post)
        print("====")

    unique_states = [{'base_C': True, 'base_G': True, 'ee_C': True, 'ee_H': True},
                     {'base_C': True, 'base_G': True, 'ee_C': True, 'ee_G': True},
                     {'base_C': True, 'base_G': True, 'ee_B': True, 'ee_G': True},
                     {'base_B': True, 'base_G': True, 'ee_B': True, 'ee_G': True},
                     {'base_B': True, 'base_G': True, 'ee_A': True, 'ee_G': True}]
    print("Unique states", unique_states)

    # formula = [
    #     {'base_F': True, 'ee_C': True},
    #     {'base_F': True, 'ee_B': True},
    #     {'base_E': True, 'ee_B': True},
    #     {'base_E': True, 'ee_A': True},
    #     {'base_H': True, 'ee_G': True}
    # ]
    # formula_always = [
    #     {'base_F': True, 'ee_C': True},
    #     {'base_F': True, 'ee_B': True},
    #     {'base_E': True, 'ee_B': True},
    #     {'base_E': True, 'ee_A': True},
    #     {'base_H': True, 'ee_G': True}]
    # formula = {'base_E': False}
    selected_symbols = {}
    for sym in symbols_to_plot:
        selected_symbols[sym] = symbols[sym]
    symbols_device, _ = symbols_and_workspace_to_device(selected_symbols, None)
    # constraint = AlwaysFormula(symbols_device, formula, epsilon=dmp_opts['epsilon'])
    # constraint = EventuallyOrFormulas(symbols_device, formula, epsilon=dmp_opts['epsilon'])
    # constraint = AndEventuallyFormulas(symbols_device, formula, epsilon=dmp_opts['epsilon'])
    # constraint = SequenceFormulas(symbols_device, formula, epsilon=dmp_opts['epsilon'])
    # constraint = SequenceFormulasAndAlways(symbols_device, formula, formula_always, epsilon=dmp_opts['epsilon'])
    constraint = SequenceFormulasMultiplePostsAndAlways(symbols_device, intermediate_states, unique_states, epsilon=dmp_opts['epsilon'])
    # constraint = SkillConstraint(symbols_device, None, suggestion['intermediate_states'], None, suggestion['unique_states'], None, workspace_bnds, dmp_opts['epsilon'], dmp_opts)

    losses, learned_rollouts, c_sat = evaluate_constraint(val_set, constraint, path_to_original_model, basis_fs=dmp_opts['basis_fs'],
                        dt=dmp_opts['dt'], output_dimension=dmp_opts['dimension'])

    trajectories_ee = learned_rollouts[:, :, 2:]
    trajectories_base = create_stretch_base_traj(learned_rollouts)
    fig, ax = create_ax_array(3, ncols=2)
    plot_sat_unsat_trajectories(trajectories_ee, c_sat, ax[0], ax[1])
    plot_sat_unsat_trajectories(trajectories_base, c_sat, ax[0], ax[1])
    for sym in symbols_to_plot:
        symbols[sym].plot(ax[0], dim=3, alpha=0.05)
        symbols[sym].plot(ax[1], dim=3, alpha=0.05)
    plt.suptitle("Before learning")
    print("losses before: {}".format(losses))

    ###################
    # Attempt to make the model obey a new constraint
    ############
    base_folder = '../data'
    output_model_path = '../data/dmps/' + dmp_opts['skill_name'] + ".pt"
    learned_model, results_folder = learn_skill_with_constraints(dmp_opts['skill_name'], constraint,
                              base_folder, demo_folder, old_demo_folder=old_demo_folder, previous_model_path=path_to_original_model,
                                enforce_type="train", main_loss_weight=dmp_opts['m_weight'], constraint_loss_weight=dmp_opts['c_weight'], basis_fs=dmp_opts['basis_fs'],
                                dt=dmp_opts['dt'], n_epochs=dmp_opts['n_epochs'], output_dimension=dmp_opts['dimension'], epsilon=dmp_opts['epsilon'], output_model_path=output_model_path)
    #
    # # Plot the intermediate states
    int_rollouts, int_sat = load_intermediate_data(results_folder)
    for ii, (learned_rollouts, c_sat) in enumerate(zip(int_rollouts, int_sat)):
        if ii % 10 != 0:
            continue
        trajectories_ee = learned_rollouts[:, :, 2:]
        trajectories_base = create_stretch_base_traj(learned_rollouts)
        fig, ax = create_ax_array(3, ncols=2)
        plot_sat_unsat_trajectories(trajectories_ee, c_sat, ax[0], ax[1])
        plot_sat_unsat_trajectories(trajectories_base, c_sat, ax[0], ax[1])
        for sym in symbols_to_plot:
            symbols[sym].plot(ax[0], dim=3, alpha=0.05)
            symbols[sym].plot(ax[1], dim=3, alpha=0.05)
        plt.suptitle("Epoch {}".format(ii))
        plt.savefig(join(results_folder, "epoch_{:03d}.png".format(ii)))
        # plt.close()


    losses, learned_rollouts_post, c_sat_post = evaluate_constraint(val_set, constraint, output_model_path, basis_fs=dmp_opts['basis_fs'],
                        dt=dmp_opts['dt'], output_dimension=dmp_opts['dimension'])
    trajectories_ee_post = learned_rollouts_post[:, :, 2:]
    trajectories_base_post = create_stretch_base_traj(learned_rollouts_post)
    fig_post, ax_post = create_ax_array(3, ncols=2)
    print(losses)
    print(c_sat_post)
    plot_sat_unsat_trajectories(trajectories_ee_post, c_sat_post, ax_post[0], ax_post[1])
    plot_sat_unsat_trajectories(trajectories_base_post, c_sat_post, ax_post[0], ax_post[1])
    for sym in symbols_to_plot:
        symbols[sym].plot(ax_post[0], dim=3, alpha=0.05)
        symbols[sym].plot(ax_post[1], dim=3, alpha=0.05)
    plt.suptitle("After learning_returned_model")

    ####################################################
    # Calculates which propositions the robot passes through
    #####################
    trajectories_base_and_ee = np.zeros([trajectories_base_post.shape[0], trajectories_base_post.shape[1], 5])
    trajectories_base_and_ee[:, :, :2] = trajectories_base_post[:, :, :2]
    trajectories_base_and_ee[:, :, 2:] = trajectories_ee_post

    unique_traj = []
    for trajectory in trajectories_base_and_ee:
        traj_in_syms = find_traj_in_syms(trajectory, symbols)
        reduced_traj = reduce_sym_traj(traj_in_syms)
        reduced_traj = remove_mutually_exclusive_symbols_list_of_states(reduced_traj, symbols)
        if reduced_traj not in unique_traj:
            unique_traj.append(reduced_traj)
    for u in unique_traj:
        for s in u:
            print(s)
        print("*************")

    plt.show()



