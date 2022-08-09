#!/usr/bin/env python

import sys
import torch
import os
from torch import nn, optim, autograd
from dl2_lfd.ltl_diff import oracle, constraints
from os.path import join
from dl2_lfd.nns.dmp_nn import DMPNN
from dl2_lfd.dmps.dmp import load_dmp_demos, DMP
from dl2_lfd.helper_funcs.conversions import np_to_pgpu
from dl2_lfd.helper_funcs.utils import t_stamp
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import dl2_lfd.ltl_diff.ltldiff as ltd
from dl2_lfd.elaborateDMP import training_loop
import time
import copy
import shutil
from synthesis_based_repair.skills import find_traj_in_syms
from synthesis_based_repair.tools import dict_to_formula, fk_stretch

DEVICE = "cpu"


def generate_trajectory(skill_name, dmp_folder, symbols, workspace_bnds, suggestions_pre, suggestions_post, folder_save, opts):
    """
    Generate trajectories based on the dmp and randomly sampling from the preconditions and postconditions

    :param skill_name:
    :param dmp_folder:
    :param symbols:
    :param workspace_bnds:
    :param suggestions_pre:
    :param suggestions_post:
    :param folder_save:
    :param opts:
    :return:
    """
    os.makedirs(folder_save, exist_ok=True)
    model = DMPNN(opts['start_dimension'], 1024, opts['dimension'], opts['basis_fs']).to(DEVICE)
    model.load_state_dict(torch.load(dmp_folder + skill_name + ".pt"))
    pre_cons = constraints.States(symbols, suggestions_pre, epsilon=opts['epsilon'], buffer=0.1)
    post_cons = constraints.States(symbols, suggestions_post, epsilon=opts['epsilon'], buffer=0.1)
    start_pose_in = workspace_bnds[:, 0] + np.random.random(opts['dimension']) * (workspace_bnds[:, 1] - workspace_bnds[:, 0])
    prec = pre_cons.condition(ltd.TermStatic(torch.from_numpy(start_pose_in[np.newaxis, :])))
    while not prec.satisfy(0):
        start_pose_in = workspace_bnds[:, 0] + np.random.random(opts['dimension']) * (
                workspace_bnds[:, 1] - workspace_bnds[:, 0])
        prec = pre_cons.condition(ltd.TermStatic(torch.from_numpy(start_pose_in[np.newaxis, :])))

    end_pose_in = workspace_bnds[:, 0] + np.random.random(opts['dimension']) * (workspace_bnds[:, 1] - workspace_bnds[:, 0])
    postc = post_cons.condition(ltd.TermStatic(torch.from_numpy(end_pose_in[np.newaxis, :])))
    while not postc.satisfy(0):
        end_pose_in = workspace_bnds[:, 0] + np.random.random(opts['dimension']) * (
                workspace_bnds[:, 1] - workspace_bnds[:, 0])
        postc = post_cons.condition(ltd.TermStatic(torch.from_numpy(end_pose_in[np.newaxis, :])))

    start_pose = np.zeros([1, int(opts['start_dimension']/opts['dimension']), opts['dimension']], dtype=float)
    start_pose[0, 0, :start_pose_in.shape[0]] = start_pose_in
    start_pose[0, -1, :end_pose_in.shape[0]] = end_pose_in
    learned_weights = model(np_to_pgpu(start_pose))
    # print("Learned weights: {}".format(learned_weights))
    dmp = DMP(opts['basis_fs'], opts['dt'], opts['dimension'])
    # print("Calculating rollout")
    learned_rollouts, _, _ = \
        dmp.rollout_torch(torch.tensor(start_pose[:, 0, :]).to(DEVICE), torch.tensor(start_pose[:, -1, :]).to(DEVICE), learned_weights)

    np.savetxt(folder_save + "/rollout-" + opts['f_name_add'] + ".txt",
               learned_rollouts[0][:, :].cpu().detach().numpy(), delimiter=" ")
    np.savetxt(folder_save + "/start-state-" + opts['f_name_add'] + ".txt",
               start_pose[0, :, :], delimiter=" ")


def learn_skill_with_constraints(skill_name, constraint, base_folder, demo_folder, old_demo_folder=None, previous_model_path=None,
                                enforce_type="train", main_loss_weight=1.0, constraint_loss_weight=5.0, basis_fs=30,
                                dt=0.01, n_epochs=200, output_dimension=6, epsilon=0.01, output_model_path=None):
    """
    Create a dynamic motion primite that will generate trajectories that mimic the initial trajectories while also
    obeying the hard constraints if desired

    :param old_skill:
    :param new_skill:
    :param suggestion:
    :param hard_constraints:
    :param symbols:
    :param workspace_bnds:
    :param opts:
    :return:
    """

    if enforce_type == "unconstrained":
        enforce, adversarial = False, False
    elif enforce_type == "train":
        enforce, adversarial = True, False
    elif enforce_type == "adversarial":
        enforce, adversarial = True, True
    else:
        raise Exception("enforce_type " + enforce_type + "invalid, must be unconstrained. train, or adversarial")

    if not ((old_demo_folder is None) == (previous_model_path is None)):
        raise Exception("If using old data, must supply the old model as well")

    # Create folder for the outputs and duplicate the data
    results_folder = join(base_folder, "logs/learn_skill-{}-at-{}".format(skill_name, t_stamp()))
    os.makedirs(results_folder, exist_ok=False)
    # TODO: dump options to text file in folder

    # When learning a new skill based on an old skill, copy the original skill trajectories
    if old_demo_folder is not None and not os.path.isdir(demo_folder):
        print("Creating {} and copying data from {}".format(demo_folder, old_demo_folder))
        shutil.copytree(old_demo_folder, demo_folder)

    t_start_states, t_pose_hists = load_dmp_demos(demo_folder + "/train", n_points=int(1/dt))
    t_start_states, t_pose_hists = np_to_pgpu(t_start_states), np_to_pgpu(t_pose_hists)
    train_set = TensorDataset(t_start_states, t_pose_hists)

    v_start_states, v_pose_hists = load_dmp_demos(demo_folder + "/val", n_points=int(1/dt))
    v_start_states, v_pose_hists = np_to_pgpu(v_start_states), np_to_pgpu(v_pose_hists)
    val_set = TensorDataset(v_start_states, v_pose_hists)

    learned_model = training_loop(train_set, val_set, constraint, enforce, adversarial, results_folder, main_loss_weight=main_loss_weight,
                  constraint_loss_weight=constraint_loss_weight, basis_fs=basis_fs, dt=dt, n_epochs=n_epochs, output_dimension=output_dimension, previous_model_path=previous_model_path, output_model_path=output_model_path)

    return learned_model, results_folder


def symbols_and_workspace_to_device(symbols, workspace_bnds, device=DEVICE):
    """
    Puts the symbols on the device that is being used (cpu or cuda)
    :param symbols:
    :return:
    """
    symbols_device = dict()
    for sym, data in symbols.items():
        symbols_device[sym] = copy.deepcopy(symbols[sym])
        if symbols[sym].get_type() == 'rectangle':
            symbols_device[sym].bounds = torch.from_numpy(symbols[sym].bounds).to(DEVICE)
        elif symbols[sym].get_type() == 'circle':
            symbols_device[sym].center = torch.from_numpy(symbols[sym].center).to(DEVICE)
            symbols_device[sym].radius = torch.from_numpy(symbols[sym].radius).to(DEVICE)
    if workspace_bnds is None:
        workspace_bnds_device = None
    else:
        workspace_bnds_device = torch.from_numpy(workspace_bnds).to(DEVICE)

    return symbols_device, workspace_bnds_device


def generate_constraints_from_suggestion(suggestion, symbols, workspace_bnds):
    """
    Generates constraints from the suggestion given by the symbolic repair

    :param suggestion:
    :param symbols:
    :param workspace_bnds:
    :return:
    """
    # Create the symbols as constraints
    symbols_device, workspace_bnds_device = symbols_and_workspace_to_device(symbols, workspace_bnds)

    # Constraints on transitioning between the correct states
    constraint = constraints.AutomaticSkill(symbols_device, suggestion['intermediate_states_all_pres'],
                                                suggestion['intermediate_states'],
                                                suggestion['final_postconditions'], suggestion['unique_states'],
                                                suggestion['avoid_states'], workspace_bnds_device, epsilon, constraint_type)

    # Constraints on always being in one of the intermediate states
    intermediate_constraints = []
    for suggestion_intermediate_all_posts in suggestion['intermediate_states']:
        intermediate_constraints.append(constraints.AutomaticIntermediateSteps(symbols_device,
                                                                               suggestion_intermediate_all_posts,
                                                                               suggestion['unique_states'],
                                                                               suggestion['avoid_states'],
                                                                               workspace_bnds_device,
                                                                               opts['epsilon']))

    return constraint, intermediate_constraints


def create_stretch_base_traj(rollouts):
    """
    Selects only the xy out of the rollout for the stretch base and adds a z of 0.1

    :param rollouts:
    :return:
    """
    out = 0.1 * np.ones([rollouts.shape[0], rollouts.shape[1], 3])
    out[:, :, :2] = rollouts[:, :, :2]

    return out



