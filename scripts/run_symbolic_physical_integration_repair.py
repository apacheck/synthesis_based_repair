#!/usr/bin/env python
import importlib as importlib

from synthesis_based_repair.tools import write_spec, clear_file, dict_to_formula, json_load_wrapper
import matplotlib.pyplot as plt
import copy
from synthesis_based_repair.symbolic_repair import run_repair
import numpy as np
from synthesis_based_repair.symbols import load_symbols
from synthesis_based_repair.skills import load_skills_from_json, Skill, write_skills_str
import json
from synthesis_based_repair.physical_implementation import run_elaborateDMP
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

                for skill_name, skill in skills.items():
                    if skill.is_suggestion():
                        fig, ax = skill.plot_symbolic_skill(symbols, [-0.5, 3.5], [-0.5, 3.5])
                        plt.savefig(folder_plot + "seed_" + str(seed) + "_" + skill_name)

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
            for idx, suggestion in suggestions.items():
                dmp_opts['demo_folder'] = folder_trajectories + dict_to_formula(suggestion['original_skill'],
                                                                                       include_false=False) + "_" + str(
                    iteration_count) + "_new" + "/"
                dmp_opts['previous_skill_name'] = dict_to_formula(suggestion['original_skill'], include_false=False)
                dmp_opts['skill_name'] = dict_to_formula(suggestion['new_skill'], include_false=False) + "_" + str(
                    iteration_count) + "_new"
                _, val_losses[idx], int_sat[idx] = run_elaborateDMP(dmp_opts['previous_skill_name'],
                                                                    dmp_opts['skill_name'],
                                                                    suggestion, user_spec['hard_constraints'],
                                                                    symbols, workspace_bnds, dmp_opts)
                # _, val_losses[idx], int_sat[idx] = spoof_elaborateDMP(dmp_opts['previous_skill_name'],
                #                                                       dmp_opts['skill_name'],
                #                                                       suggestion, user_spec['hard_constraints'],
                #                                                       symbols, workspace_bnds, dmp_opts)


            ################################################
            # Determine which changes are physically valid #
            ################################################
            fid = open(file_log, 'a')
            fid.write("Seed {} Suggestion # {}\n".format(seed, iteration_count))
            fid.close()
            write_skills_str(skills, file_log, only_suggestions=True)
            skill_suggestions = dict()
            for idx, suggestion in suggestions.items():
                skill_suggestions[suggestion['name']] = Skill(suggestion, True)
            write_skills_str(skills, file_log, only_suggestions=True)
            fid = open(file_log, 'a')
            fid.write("Val_losses: {}\n".format(val_losses))
            fid.write("intermediate sat: {}\n".format(int_sat))
            fid.write("Additional not allowed repair:\n")
            # If val_loss for a skill is acceptable, save that skill and add it to the specification, along with a
            # constraint not to change it
            # If a val loss is not acceptable, check each of the intermediate constraints and restrict based off which ones
            # do not satisfy
            for idx, val_loss in val_losses.items():
                if val_loss[-1][-1] > loss_threshold:
                    new_skill_name = suggestions[idx]['name']
                    skills[new_skill_name] = Skill(suggestions[idx], True)
                    user_spec['sys_init_false'].append(new_skill_name)
                    not_allowed_repair += "!" + new_skill_name + "\n"
                    sym_opts['existing_skills'].append(new_skill_name)
                else:
                    for one_constraint_sat, transition in zip(int_sat[idx], suggestions[idx]['intermediate_states']):
                        if one_constraint_sat[-1][-1] < loss_threshold:
                            for post_dict in transition[1]:
                                additional_not_allowed_repair = "!(" + dict_to_formula(transition[0]) + " & " \
                                                                + dict_to_formula(suggestions[idx]['original_skill'],
                                                                                  include_false=False) + " & " \
                                                                + dict_to_formula(post_dict, prime=True) + ")\n"
                                not_allowed_repair += additional_not_allowed_repair
                                fid.write(additional_not_allowed_repair)
            fid.write("********************\n")
            fid.close()

            iteration_count += 1
