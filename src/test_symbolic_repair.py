#!/usr/bin/env python
import importlib as importlib

from tools import write_spec, clear_file
import matplotlib.pyplot as plt
import copy
from symbolic_repair import run_repair
import numpy as np
from symbols import load_symbols
from skills import load_skills_from_json, Skill, write_skills_str
import json

if __name__ == "__main__":

    plt.close('all')
    file_symbols = "../data/nine_squares/nine_squares_symbols.json"
    file_skills = "../data/nine_squares/nine_squares_skills.json"
    folder_plot = "../data/nine_squares/plots/"
    skill_names = ['skill0', 'skill1']
    file_structured_slugs = '../data/nine_squares/nine_squares_a.structuredslugs'
    file_log = '../data/nine_squares/log_a.txt'
    clear_file(file_log)

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

    ############################
    # Load skills and symbols ##
    ############################
    symbols = load_symbols(file_symbols)
    skills_all = load_skills_from_json(file_skills)
    original_skills = dict()
    for skill_name in skill_names:
        original_skills[skill_name] = skills_all[skill_name]

    for skill_name in skill_names:
        original_skills[skill_name + "b"] = copy.deepcopy(original_skills[skill_name])

    ###################################
    # Iterate through different seeds #
    ###################################
    for seed in range(2):
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

            skills = copy.deepcopy(original_skills)
            write_spec(file_structured_slugs, symbols, skills, user_spec, change_cons, not_allowed_repair, sym_opts)
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

            for idx, suggestion in suggestions.items():
                skills[suggestion['name']] = Skill(suggestion, True)



