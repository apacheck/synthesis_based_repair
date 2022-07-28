#!/usr/bin/env python
import importlib as importlib

from synthesis_based_repair.tools import write_spec, clear_file, dict_to_formula, json_load_wrapper
import matplotlib.pyplot as plt
import copy
from synthesis_based_repair.symbolic_repair import run_repair
import numpy as np
from synthesis_based_repair.symbols import load_symbols
from synthesis_based_repair.skills import load_skills_from_json, Skill, write_skills_str, remove_mutually_exclusive_symbols_intermediate_states, remove_mutually_exclusive_symbols_list_of_states
import json
from synthesis_based_repair.physical_implementation import learn_skill_with_constraints, create_stretch_base_traj, symbols_and_workspace_to_device
from dl2_lfd.elaborateDMP import evaluate_constraint
from dl2_lfd.ltl_diff.constraints import SequenceFormulasMultiplePostsAndAlways, SequenceFormulasMultiplePostsAndAlwaysWithIK
from dl2_lfd.dmps.dmp import load_dmp_demos, DMP
from dl2_lfd.helper_funcs.conversions import np_to_pgpu
from torch.utils.data import TensorDataset, DataLoader
from synthesis_based_repair.visualization import plot_sat_unsat_trajectories, create_ax_array, load_intermediate_data, plot_sym_intersection
import argparse
from os.path import join


if __name__ == "__main__":

    plt.close('all')
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_spec", help="Name of json file with user spec", required=True)
    parser.add_argument("--file_names", help="File names", required=True)
    parser.add_argument("--sym_opts", help="Opts involving spec writing and repair", required=True)
    parser.add_argument("--dmp_opts", help="Opts dmps", required=True)
    parser.add_argument("--n_seeds", help="Number of seeds to run", required=False, default=1, type=int)
    parser.add_argument("--loss_threshold", help="Loss threshold to accept change", required=False, default=0.8, type=float)
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

    symbols_device, _ = symbols_and_workspace_to_device(symbols, None)


    ###################################
    # Iterate through different seeds #
    ###################################
    for seed in range(args.n_seeds):
        iteration_count = 1
        attempt_cnt = 1
        np.random.seed(seed)

        is_realizable = False
        skills = copy.deepcopy(original_skills)
        while not is_realizable:

            write_spec(file_structured_slugs, symbols, skills, user_spec, change_cons, not_allowed_repair, sym_opts)

            ###########################
            # SYMBOLIC REPAIR #########
            ###########################
            sym_opts['only_synthesis'] = True
            is_realizable, _ = run_repair(file_structured_slugs, sym_opts)
            if is_realizable:
                fid = open(file_log, 'a')
                fid.write("Seed {} took {} iterations\n".format(seed, iteration_count))
                fid.close()
                write_skills_str(skills, file_log, only_suggestions=True)

                # for skill_name, skill in skills.items():
                #     if skill.is_suggestion():
                #         fig, ax =
                #         for pre, posts in skill.get_intermediate_states().items():
                #         fig, ax = skill.plot_symbolic_skill(symbols, [-0.5, 3.5], [-0.5, 3.5])
                #         plt.savefig(folder_plot + "seed_" + str(seed) + "_" + skill_name)

                break

            # skills = copy.deepcopy(original_skills)
            # write_spec(file_structured_slugs, symbols, skills, user_spec, change_cons, not_allowed_repair, sym_opts)
            sym_opts['only_synthesis'] = False
            found_suggestion = False
            # np.random.seed(seed)
            while not found_suggestion:
                try:
                    sym_opts['post_first'] = bool((post_first + attempt_cnt) % 2)
                    attempt_cnt += 1
                    _, suggestions = run_repair(file_structured_slugs, sym_opts)
                    found_suggestion = True
                except Exception as e:
                    print(e)

            val_losses = dict()
            int_sat = dict()

            # Saving the constraint and the losses
            fid = open(file_log, 'a')
            fid.write("Seed {} Suggestion # {}\n".format(seed, iteration_count))
            fid.close()
            write_skills_str(skills, file_log, only_suggestions=True)
            skill_suggestions = dict()
            for idx, suggestion in suggestions.items():
                skill_suggestions[suggestion['name']] = Skill(suggestion, True)
            write_skills_str(skills, file_log, only_suggestions=True)

            # The actual physical repair
            for idx, suggestion in suggestions.items():
                dmp_opts['previous_skill_name'] = dict_to_formula(suggestion['original_skill'], include_false=False)
                dmp_opts['skill_name'] = dict_to_formula(suggestion['new_skill'], include_false=False) + "_" + str(
                    iteration_count) + "_new"

                old_demo_folder = folder_trajectories + dmp_opts['previous_skill_name'] + "/"
                demo_folder = folder_trajectories + dmp_opts['skill_name']

                path_to_original_model = "/home/adam/repos/synthesis_based_repair/data/dmps/" + dmp_opts['previous_skill_name'] + ".pt"

                intermediate_states = remove_mutually_exclusive_symbols_intermediate_states(suggestion['intermediate_states'],
                                                                                            symbols)
                unique_states = remove_mutually_exclusive_symbols_list_of_states(suggestion['unique_states'], symbols)

                constraint = SequenceFormulasMultiplePostsAndAlwaysWithIK(symbols_device, intermediate_states, unique_states,
                                                                    epsilon=dmp_opts['epsilon'])

                base_folder = '../data'
                output_model_path = '../data/dmps/' + dmp_opts['skill_name'] + ".pt"
                _, results_folder = learn_skill_with_constraints(dmp_opts['skill_name'], constraint,
                                                                             base_folder, demo_folder,
                                                                             old_demo_folder=old_demo_folder,
                                                                             previous_model_path=path_to_original_model,
                                                                             enforce_type="train",
                                                                             main_loss_weight=dmp_opts['m_weight'],
                                                                             constraint_loss_weight=dmp_opts[
                                                                                 'c_weight'],
                                                                             basis_fs=dmp_opts['basis_fs'],
                                                                             dt=dmp_opts['dt'],
                                                                             n_epochs=dmp_opts['n_epochs'],
                                                                             output_dimension=dmp_opts['dimension'],
                                                                             epsilon=dmp_opts['epsilon'],
                                                                             output_model_path=output_model_path)

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
                    for sym in symbols:
                        symbols[sym].plot(ax[0], dim=3, alpha=0.05)
                        symbols[sym].plot(ax[1], dim=3, alpha=0.05)
                    plt.suptitle("Epoch {}".format(ii))
                    plt.savefig(join(results_folder, "epoch_{:03d}.png".format(ii)))


                ################################################
                # Determine which changes are physically valid #
                ################################################
                v_start_states, v_pose_hists = load_dmp_demos(old_demo_folder + "/val",
                                                              n_points=int(1 / dmp_opts['dt']))
                v_start_states, v_pose_hists = np_to_pgpu(v_start_states), np_to_pgpu(v_pose_hists)
                val_set = TensorDataset(v_start_states, v_pose_hists)
                losses, learned_rollouts_post, c_sat_post = evaluate_constraint(val_set, constraint, output_model_path,
                                                                                basis_fs=dmp_opts['basis_fs'],
                                                                                dt=dmp_opts['dt'],
                                                                                output_dimension=dmp_opts['dimension'])



                fid = open(file_log, 'a')
                fid.write("Val_losses: {}\n".format(losses))
                # fid.write("intermediate sat: {}\n".format(int_sat))
                fid.write("Additional not allowed repair:\n")

                # If val_loss for a skill is acceptable, save that skill and add it to the specification, along with a
                # constraint not to change it
                if losses[0][1] > loss_threshold:
                    new_skill_name = suggestions[idx]['name']
                    skills[new_skill_name] = Skill(suggestions[idx], True)
                    user_spec['sys_init_false'].append(new_skill_name)
                    not_allowed_repair += "\n!" + new_skill_name
                    sym_opts['existing_skills'].append(new_skill_name)
                # If a val loss is not acceptable, check each of the intermediate constraints and restrict based off which ones
                # do not satisfy
                else:
                    for transition in suggestion['intermediate_states']:
                        intermediate_constraint = SequenceFormulasMultiplePostsAndAlways(symbols_device, [transition],
                                                                            None,
                                                                            epsilon=dmp_opts['epsilon'])

                        losses, learned_rollouts, c_sat = evaluate_constraint(val_set, intermediate_constraint, output_model_path,
                                                                              basis_fs=dmp_opts['basis_fs'],
                                                                              dt=dmp_opts['dt'],
                                                                              output_dimension=dmp_opts['dimension'])
                        if losses[0][1] < loss_threshold:
                            if len(transition[1]) > 1:
                                continue
                            for post_dict in transition[1]:
                                additional_not_allowed_repair = "!(" + dict_to_formula(transition[0], include_false=False) + " & " \
                                                                + dict_to_formula(suggestions[idx]['original_skill'],
                                                                                  include_false=False) + " & " \
                                                                + dict_to_formula(post_dict, prime=True, include_false=False) + ")"
                                not_allowed_repair += "\n" + additional_not_allowed_repair
                                fid.write("Percent satisfy: {:1.2f}".format(losses[0][1]) + additional_not_allowed_repair + "\n")
                fid.close()
            fid = open(file_log, 'a')
            fid.write("\n********************\n")
            fid.close()

            iteration_count += 1
