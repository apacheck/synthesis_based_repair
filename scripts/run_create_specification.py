#!/usr/bin/env python
from src.tools import write_spec, json_load_wrapper
from src.symbols import load_symbols
import json
from src.skills import load_skills_from_json
import argparse
import copy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_spec", help="Name of json file with user spec", required=True)
    parser.add_argument("--file_names", help="File names", required=True)
    parser.add_argument("--sym_opts", help="Opts involving spec writing and repair", required=True)
    args = parser.parse_args()

    user_spec = json_load_wrapper(args.user_spec)
    file_names = json_load_wrapper(args.file_names)
    sym_opts = json_load_wrapper(args.sym_opts)

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

    write_spec(file_names["file_structured_slugs"], symbols, original_skills, user_spec, user_spec['change_cons'], user_spec['not_allowed_repair'], sym_opts)