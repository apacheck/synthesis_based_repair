#!/usr/bin/env python
import importlib as importlib

from tools import write_spec, clear_file, dict_to_formula
import matplotlib.pyplot as plt
import copy
from symbolic_repair import run_repair
import numpy as np
from symbols import load_symbols
from skills import load_skills_from_json, Skill, write_skills_str
import json
from physical_implementation import run_elaborateDMP


if __name__ == "__main__":

    plt.close('all')
    file_symbols = "../data/nine_squares/nine_squares_symbols.json"
    file_skills = "../data/nine_squares/nine_squares_skills.json"
    folder_trajectories = "../data/nine_squares/trajectories/"
    folder_plot = "../data/nine_squares/plots/"
    skill_names = ['skill0', 'skill1']
    file_structured_slugs = '../data/nine_squares/nine_squares_a.structuredslugs'
    file_log = '../data/nine_squares/log_a.txt'
    clear_file(file_log)
    workspace_bnds = np.array([[-0.5, 3.5], [-0.5, 3.5]])
    loss_threshold = 0.8

    f_user_spec = '../data/nine_squares/nine_squares_a.json'
    fid = open(f_user_spec, 'r')
    user_spec = json.load(fid)
    fid.close()

    post_first = False

    change_cons = "!(x0' & x1')\n" \
                  "!(x0' & x2')\n" \
                  "!(x1' & x2')\n" \
                  "!(y0' & y1')\n" \
                  "!(y0' & y2')\n" \
                  "!(y1' & y2')\n" \
                  "(x0' | x1' | x2')\n" \
                  "(y0' | y1' | y2')\n" \
                  "!(x0 & x1)\n" \
                  "!(x0 & x2)\n" \
                  "!(x1 & x2)\n" \
                  "!(y0 & y1)\n" \
                  "!(y0 & y2)\n" \
                  "!(y1 & y2)\n" \
                  "(x0 | x1 | x2)\n" \
                  "(y0 | y1 | y2)\n" \
                  "(x0 <-> !x0') | (x1 <-> !x1') | (x2 <-> !x2') | ((x0 <-> x0') & (x1 <-> x1') & (x2 <-> x2'))\n" \
                  "(y0 <-> !y0') | (y1 <-> !y1') | (y2 <-> !y2') | ((y0 <-> y0') & (y1 <-> y1') & (y2 <-> y2'))\n" \
                  "!((x0 <-> x0') & (x1 <-> x1') & (x2 <-> x2') & (y0 <-> y0') & (y1 <-> y1') & (y2 <-> y2'))\n"
    not_allowed_repair = "!skill0b\n" \
                         "!skill1b\n"

    sym_opts = {'return_with_one_repair': True,
                'run_original': False,
                'only_synthesis': False,
                'enforce_reactive_variables': False,
                'suggestions': None,
                'do_names': True,
                'post_first': False,
                'post_repair_cnt': 0,
                'to_file': False,
                'fid_base': None,
                'existing_skills': ['skill0', 'skill1', 'skill0b', 'skill1b'],
                'reactive_variables': [],
                'reactive_variables_current': [],
                'n_factors': 2,
                'generate_figure': ''
                }

    dmp_opts = {'enforce_type': 'adversarial',
                'n_train_trajs': 32,
                'n_val_trajs': 32,
                'demo_folder': folder_trajectories,
                'basis_fs': 30,
                'dmp_folder': '../data/dmps/',
                'dt': 0.01,
                'c_weight': 50,
                'm_weight': 1,
                'epsilon': 1E-6,
                'plt_background': None,
                'plot_limits': workspace_bnds,
                'n_epochs': [50],
                'start_dimension': 4,
                'dimension': 2,
                'n_states': 2,
                'base_folder': '../data',
                'use_previous': True,
                # 'symbols': symbols,
                'file_physical_log': "../data/logs/a_nine_squares_log.txt",
                'constraints': [['implication_next', 'always']]
                }

    ############################
    # Load skills and symbols ##
    ############################
    symbols = load_symbols(file_symbols)
    dmp_opts['symbols'] = symbols
    skills_all = load_skills_from_json(file_skills)
    original_skills = dict()
    for skill_name in skill_names:
        original_skills[skill_name] = skills_all[skill_name]

    for skill_name in skill_names:
        original_skills[skill_name + "b"] = copy.deepcopy(original_skills[skill_name])

    ###################################
    # Iterate through different seeds #
    ###################################
    for seed in range(1):
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
                #                                                       suggestion, arg_user_spec['hard_constraints'],
                #                                                       symbols, workspace_bnds, dmp_opts)


            ################################################
            # Determine which changes are physically valid #
            ################################################
            fid = open(file_log, 'a')
            fid.write("Seed {} Suggestion # {}\n".format(seed, iteration_count))
            fid.close()
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




