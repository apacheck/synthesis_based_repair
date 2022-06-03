#!/usr/bin/env python

import sys
import torch
from torch import nn, optim, autograd
import os
from dl2_lfd.ltl_diff import oracle, constraints
from dl2_lfd.dmps.dmp import load_dmp_demos, DMP
from dl2_lfd.helper_funcs.conversions import np_to_pgpu
from dl2_lfd.helper_funcs.utils import t_stamp
from torch.utils.data import TensorDataset, DataLoader
from synthesis_based_repair.tools import write_spec, clear_file, dict_to_formula
import copy
import numpy as np
from synthesis_based_repair.symbols import load_symbols
from synthesis_based_repair.skills import load_skills_from_json, Skill, write_skills_str
import json
from synthesis_based_repair.physical_implementation import rollout_error
from dl2_lfd.nns.dmp_nn import DMPNN
# from dl2_lfd.ltl_diff import oracle

DEVICE = "cpu"


if __name__ == "__main__":

    file_symbols = "../data/stretch/stretch_symbols.json"
    file_skills = "../data/stretch/stretch_skills.json"
    folder_trajectories = "../data/stretch/trajectories/skillStretch1/"
    folder_plot = "../data/stretch/plots/"
    skill_names = ['skillStretch0', 'skillStretch1']
    file_structured_slugs = '../data/stretch/stretch.structuredslugs'
    file_log = '../data/stretch/log_stretch.txt'
    clear_file(file_log)
    plot_limits = np.array([[-0.5, 3.5], [-0.5, 2.5], [-0.1, 0.5]])
    workspace_bnds = np.array([[0, 3], [0, 2], [0, 6.28], [0, 0.2], [0.2, 0.4], [0, 6.28]])
    loss_threshold = 0.8

    opts = {'enforce_type': 'adversarial',
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
            # 'plot_limits': plot_limits,
            'n_epochs': [50],
            'start_dimension': 12,
            'dimension': 6,
            'n_states': 2,
            'base_folder': '../data',
            'use_previous': False,
            # 'symbols': symbols,
            'file_physical_log': "../data/logs/stretch_log.txt",
            'constraints': [['implication_next', 'always', 'stretch']]
            }

    symbols = load_symbols(file_symbols)
    opts['symbols'] = symbols
    skills_all = load_skills_from_json(file_skills)
    original_skills = dict()
    for skill_name in skill_names:
        original_skills[skill_name] = skills_all[skill_name]

    for skill_name in skill_names:
        original_skills[skill_name + "b"] = copy.deepcopy(original_skills[skill_name])

    f_suggestion = '/home/adam/repos/synthesis_based_repair/data/stretch/suggestion.json'
    fid = open(f_suggestion, 'r')
    suggestions = json.load(fid)
    fid.close()

    results_root = opts['base_folder'] + "/logs/generalized-exps-{}".format(t_stamp())
    # os.makedirs(opts['demo_folder'], exist_ok=True)
    # os.makedirs(opts['demo_folder'] + "/train", exist_ok=True)
    # os.makedirs(opts['demo_folder'] + "/val", exist_ok=True)

    if opts['enforce_type'] == "unconstrained":
        enforce, adversarial = False, False
    if opts['enforce_type'] == "train":
        enforce, adversarial = True, False
    if opts['enforce_type'] == "adversarial":
        enforce, adversarial = True, True

    # t_start_states, t_pose_hists = load_dmp_demos(opts['demo_folder'] + "/train", n_interp_pts=int(1/opts['dt']), opts=opts)
    t_start_states, t_pose_hists = load_dmp_demos(opts['demo_folder'] + "/train")
    t_start_states, t_pose_hists = np_to_pgpu(t_start_states), np_to_pgpu(t_pose_hists)
    train_set = TensorDataset(t_start_states, t_pose_hists)

    symbols_device = dict()
    for sym, data in symbols.items():
        symbols_device[sym] = copy.deepcopy(symbols[sym])
        if symbols[sym].get_type() == 'rectangle' or symbols[sym].get_type() == 'rectangle-ee':
            symbols_device[sym].bounds = torch.from_numpy(symbols[sym].bounds).to(DEVICE)
        elif symbols[sym].get_type() == 'circle' or symbols[sym].get_type() == 'circle-ee':
            symbols_device[sym].center = torch.from_numpy(symbols[sym].center).to(DEVICE)
            symbols_device[sym].radius = torch.from_numpy(symbols[sym].radius).to(DEVICE)
    workspace_bnds_device = torch.from_numpy(workspace_bnds).to(DEVICE)
    constraint_list = []
    # suggestions['0']['unique_states'] = [{'a': False, 'b': False, 'c': False, 'd': True, 'aee': False, 'bee': False, 'cee': False, 'dee': True}]
    suggestion = suggestions['0']
    for constraint_type in opts['constraints']:
        constraint_list.append(constraints.AutomaticSkill(symbols_device, suggestion['intermediate_states_all_pres'],
                                                suggestion['intermediate_states'],
                                                suggestion['final_postconditions'], suggestion['unique_states'],
                                                suggestion['avoid_states'], workspace_bnds_device, opts['epsilon'], constraint_type))

    model = DMPNN(opts['start_dimension'], 1024, opts['dimension'], opts['basis_fs']).to(DEVICE)
    if opts['use_previous']:
        model.load_state_dict(torch.load(opts['dmp_folder'] + opts['previous_skill_name'] + ".pt"))
    # model = DMPNN(in_dim, 1024, t_pose_hists.shape[2], basis_fs).to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    loss_fn = rollout_error
    train_loader = DataLoader(train_set, shuffle=False, batch_size=32)
    train_losses, val_losses = [], []

    for batch_idx, (starts, rollouts) in enumerate(train_loader):
        batch_size, T, dims = rollouts.shape

        learned_weights = model(starts)
        dmp = DMP(opts['basis_fs'], opts['dt'], dims)

        c_loss, c_sat = oracle.evaluate_constraint(
            starts, rollouts, constraint_list[0], model, dmp.rollout_torch, adversarial)

        print(c_sat)



