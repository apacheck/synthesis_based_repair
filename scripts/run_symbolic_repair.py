#!/usr/bin/env python
import importlib as importlib

from synthesis_based_repair.tools import write_spec, clear_file, json_load_wrapper
import matplotlib.pyplot as plt
import copy
from synthesis_based_repair.symbolic_repair import run_repair
import numpy as np
from synthesis_based_repair.symbols import load_symbols
from synthesis_based_repair.skills import load_skills_from_json, Skill, write_skills_str
import json
import argparse

if __name__ == "__main__":

    plt.close('all')

    parser = argparse.ArgumentParser()
    parser.add_argument("--user_spec", help="Name of json file with user spec", required=True)
    parser.add_argument("--file_names", help="File names", required=True)
    parser.add_argument("--sym_opts", help="Opts involving spec writing and repair", required=True)
    parser.add_argument("--n_seeds", help="Number of seeds to run", required=False, default=10, type=int)
    args = parser.parse_args()

    file_names = json_load_wrapper(args.file_names)
    sym_opts = json_load_wrapper(args.sym_opts)
    clear_file(file_names['file_log'])

    post_first = False

    ############################
    # Load skills and symbols ##
    ############################
    symbols = load_symbols(file_names["file_symbols"])
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
        # Load the specification at the start of each round
        user_spec = json_load_wrapper(args.user_spec)

        iteration_count = 1
        attempt_cnt = 1
        np.random.seed(seed)

        is_realizable = False
        skills = copy.deepcopy(original_skills)
        # While specification is not realizable, write the current specification with the current skills.
        # Check to see if it is realizable. If it is, return the current suggestions
        # If the specificaiton is not realizable, perform one iteration of repair

        while not is_realizable:

            write_spec(file_names["file_structured_slugs"], symbols, skills, user_spec, user_spec["change_cons"], user_spec["not_allowed_repair"], sym_opts)

            ###########################
            # SYMBOLIC REPAIR #########
            ###########################
            # Checking if the specification is realizable
            sym_opts['only_synthesis'] = True
            is_realizable, _ = run_repair(file_names["file_structured_slugs"], sym_opts)
            if is_realizable:
                fid = open(file_names["file_log"], 'a')
                fid.write("Seed {} took {} iterations\n".format(seed, iteration_count))
                fid.close()
                write_skills_str(skills, file_names["file_log"], only_suggestions=True)

                # Plot the suggestions
                for skill_name, skill in skills.items():
                    if skill.is_suggestion():
                        fig, ax = skill.plot_symbolic_skill(symbols, [-0.5, 3.5], [-0.5, 3.5])
                        plt.savefig(file_names["folder_plot"] + "seed_" + str(seed) + "_" + skill_name)

                break

            # Perform the actual repair
            skills = copy.deepcopy(original_skills)
            write_spec(file_names["file_structured_slugs"], symbols, skills, user_spec, user_spec["change_cons"], user_spec["not_allowed_repair"], sym_opts)
            sym_opts['only_synthesis'] = False
            found_suggestion = False
            # np.random.seed(seed)
            while not found_suggestion:
                try:
                    sym_opts['post_first'] = bool((post_first + attempt_cnt) % 2)
                    attempt_cnt += 1
                    _, suggestions = run_repair(file_names["file_structured_slugs"], sym_opts)
                    found_suggestion = True
                except Exception as e:
                    print(e)

            for idx, suggestion in suggestions.items():
                skills[suggestion['name']] = Skill(suggestion, True)
                user_spec['sys_init_false'].append(suggestion['name'])
