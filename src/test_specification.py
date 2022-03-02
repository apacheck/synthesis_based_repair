#!/usr/bin/env python
from tools import write_spec
from symbols import load_symbols
import json
from skills import load_skills_from_json


if __name__ == "__main__":
    f_user_spec = '../data/nine_squares/nine_squares_a.json'
    fid = open(f_user_spec, 'r')
    user_spec = json.load(fid)
    fid.close()

    file_spec = '../data/nine_squares/nine_squares_a.structuredslugs'
    file_symbols = '../data/nine_squares/nine_squares_symbols.json'
    file_skills = '../data/nine_squares/nine_squares_skills.json'
    syms = load_symbols(file_symbols)
    skills = load_skills_from_json(file_skills)

    opts = {'n_factors': 2}

    write_spec(file_spec, syms, skills, user_spec, user_spec['change_cons'], user_spec['not_allowed_repair'], opts)