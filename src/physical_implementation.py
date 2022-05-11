#!/usr/bin/env python

import sys
import torch
import os
from torch import nn, optim, autograd
from dl2_lfd.ltl_diff import oracle
import additional_constraints as constraints
from os.path import join
from dl2_lfd.nns.dmp_nn import DMPNN
from dl2_lfd.dmps.dmp import load_dmp_demos, DMP
from dl2_lfd.helper_funcs.conversions import np_to_pgpu
from dl2_lfd.helper_funcs.utils import t_stamp
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import dl2_lfd.ltl_diff.ltldiff as ltd
import additional_ltldiff as ltd2
import time
import copy
from visualization import plot_one_skill_trajectories_and_symbols_numpy

DEVICE = "cpu"


def rollout_error(output_roll, target_roll):
    return torch.norm(output_roll - target_roll, dim=2).mean()


def training_loop(train_set, val_set, constraint_list, enforce_constraint, adversarial, t_pose_hists, results_folder, intermediate_constraints, opts):
    model = DMPNN(opts['start_dimension'], 1024, opts['dimension'], opts['basis_fs']).to(DEVICE)
    if opts['use_previous']:
        model.load_state_dict(torch.load(opts['dmp_folder'] + opts['previous_skill_name'] + ".pt"))
    # model = DMPNN(in_dim, 1024, t_pose_hists.shape[2], basis_fs).to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    loss_fn = rollout_error
    train_loader = DataLoader(train_set, shuffle=True, batch_size=32)
    if val_set is not None:
        val_loader = DataLoader(val_set, shuffle=False, batch_size=32)
    else:
        val_loader = None

    train_losses, val_losses = [], []

    def batch_learn(arg_constraint, data_loader, enf_c, adv, optimize=False, do_plot=False, only_sat=False):
        losses = []
        for batch_idx, (starts, rollouts) in enumerate(data_loader):
            batch_size, T, dims = rollouts.shape

            learned_weights = model(starts)
            dmp = DMP(opts['basis_fs'], opts['dt'], dims)
            learned_rollouts = dmp.rollout_torch(starts[:, 0], starts[:, -1], learned_weights)[0]

            main_loss = loss_fn(learned_rollouts, rollouts)

            if arg_constraint is None:
                c_loss, c_sat = torch.tensor(0), torch.tensor(0)
            else:
                c_loss, c_sat = oracle.evaluate_constraint(
                    starts, rollouts, arg_constraint, model, dmp.rollout_torch, adv)

            if do_plot:
                if rollouts.shape[2] == 2:
                    _, ax = plt.subplots(ncols=3, figsize=(12,4))
                else:
                    fig = plt.figure(figsize=(15, 5))
                    ax = [None, None, None]
                    ax[0] = fig.add_subplot(1, 3, 1, projection='3d')
                    ax[1] = fig.add_subplot(1, 3, 2, projection='3d')
                    ax[2] = fig.add_subplot(1, 3, 3, projection='3d')
                    ax = np.array(ax)
                    # ax = [[None, None], [None, None]]
                    # ax[0][0] = fig.add_subplot(2, 2, 1, projection='3d')
                    # ax[0][1] = fig.add_subplot(2, 2, 2)
                    # ax[1][0] = fig.add_subplot(2, 2, 3)
                    # ax[1][1] = fig.add_subplot(2, 2, 4)
                    # ax = np.array(ax)
                plot_one_skill_trajectories_and_symbols_numpy(None, None, rollouts.cpu().detach().numpy(), opts['symbols'], opts['plot_limits'], ax=ax[0], color='b', linestyle='--')
                if arg_constraint is not None:
                    plot_one_skill_trajectories_and_symbols_numpy(None, None, learned_rollouts.cpu().detach().numpy()[c_sat.cpu().detach().numpy().astype(bool)], opts['symbols'], opts['plot_limits'], ax=ax[1], color='g')
                    plot_one_skill_trajectories_and_symbols_numpy(None, None, learned_rollouts.cpu().detach().numpy()[np.logical_not(c_sat.cpu().detach().numpy().astype(bool))], opts['symbols'], opts['plot_limits'], ax=ax[-1], color='r')
                else:
                    plot_one_skill_trajectories_and_symbols_numpy(None, None, learned_rollouts.cpu().detach().numpy(), opts['symbols'], opts['plot_limits'], ax=ax[1], color='g')
                # plot_one_skill_trajectories_and_symbols_numpy(None, None, rollouts.cpu().detach().numpy(),
                #                                               opts['symbols'], opts['plot_limits'], ax=ax, color='b',
                #                                               linestyle='--')
                # if arg_constraint is not None:
                #     plot_one_skill_trajectories_and_symbols_numpy(None, None, learned_rollouts.cpu().detach().numpy()[
                #         c_sat.cpu().detach().numpy().astype(bool)], opts['symbols'], opts['plot_limits'], ax=ax,
                #                                                   color='g')
                #     plot_one_skill_trajectories_and_symbols_numpy(None, None, learned_rollouts.cpu().detach().numpy()[
                #         np.logical_not(c_sat.cpu().detach().numpy().astype(bool))], opts['symbols'],
                #                                                   opts['plot_limits'], ax=ax, color='r')
                # else:
                #     plot_one_skill_trajectories_and_symbols_numpy(None, None, learned_rollouts.cpu().detach().numpy(),
                #                                                   opts['symbols'], opts['plot_limits'], ax=ax,
                #                                                   color='g')

            if enf_c:
                full_loss = opts['m_weight'] * main_loss + opts['c_weight'] * c_loss
            else:
                full_loss = main_loss

            losses.append([main_loss.item(), c_loss.item(), full_loss.item(), np.mean(c_sat.cpu().detach().numpy())])

            if optimize:
                optimizer.zero_grad()
                full_loss.backward()
                optimizer.step()

        return np.mean(losses, 0, keepdims=True)

    # int_sat = []
    constraint_idx = 0
    for epoch in range(sum(opts['n_epochs'])):
        epoch_start = time.time()
        do_plot = False
        if (epoch + 1) % 1 == 0 or epoch == 0:
            do_plot = True
        # if (epoch) == opts['n_epochs'] / 2:
        #     opts['c_weight'] = opts['c_weight'] * 10

        # Train loop
        model.train()
        if epoch > sum(opts['n_epochs'][:constraint_idx]) + opts['n_epochs'][constraint_idx]:
            constraint_idx += 1
        avg_train_loss = batch_learn(constraint_list[constraint_idx], train_loader, enforce_constraint, adversarial, True, do_plot=do_plot)
        if do_plot:
            plt.savefig(results_folder + "/train_epoch_" + str(epoch) + ".png")
        train_losses.append(avg_train_loss[0])

        # Validation Loop
        if val_loader is not None and (epoch == sum(opts['n_epochs'])-1):
            model.eval()
            avg_val_loss = batch_learn(constraint_list[constraint_idx], val_loader, True, False, False, do_plot=do_plot)
            if do_plot:
                plt.savefig(results_folder + "/val_epoch_" + str(epoch) + ".png")
            val_losses.append(avg_val_loss[0])

            print("e{}\t t: {} v: {}".format(epoch, avg_train_loss[0, :], avg_val_loss[0, :]))
        else:
            print("e{}\t t: {}".format(epoch, avg_train_loss[0, :]))

        # int_sat = []
        # for int_constraint in intermediate_constraints:
        #     int_sat.append(batch_learn(int_constraint, val_loader, True, False, False, do_plot=False, only_sat=True))
        #
        # print("intermediate sat: ", end="")
        # for i_s in int_sat:
        #     print("{}\t".format(i_s[-1][3]), end="")
        # print("")

        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_{}.pt".format(epoch)))
        plt.close('all')
        print("epoch time: {}".format(time.time() - epoch_start))

    os.makedirs(opts['dmp_folder'], exist_ok=True)
    torch.save(model.state_dict(), join(opts['dmp_folder'], opts['skill_name'] + ".pt"))
    os.makedirs(opts['dmp_folder'] + "/" + opts['skill_name'], exist_ok=True)
    np.savetxt(join(opts['dmp_folder'] + "/" + opts['skill_name'], "train_losses.txt"), train_losses)
    np.savetxt(join(opts['dmp_folder'] + "/" + opts['skill_name'], "val_losses.txt"), val_losses)

    # Check time
    for batch_idx, (starts, rollouts) in enumerate(train_loader):
        _, _, dims = rollouts.shape
        learned_weights = model(starts)
        dmp = DMP(opts['basis_fs'], opts['dt'], dims)
        learned_rollouts = dmp.rollout_torch(starts[:, 0], starts[:, -1], learned_weights)[0]
        # for kk, learned_rollout in enumerate(learned_rollouts):
        #     plot_one_traj_vs_time(opts['skill_name'], learned_rollout.cpu().detach().numpy(), None, None, None, None, None)
        #     plt.savefig(results_folder + "/time_" + str(kk) + ".png")
        #     plt.close()

    int_sat = []
    for int_constraint in intermediate_constraints:
        int_sat.append(batch_learn(int_constraint, val_loader, True, False, False, do_plot=False, only_sat=True))

    return model, val_losses, int_sat


def parse_user_symbols(arg_user_symbols):
    syms_out = []
    syms_split = arg_user_symbols.split(", ")
    if len(syms_split) == 0:
        return []
    for one_sym in syms_split:
        sym_split = one_sym.split(" ")
        name = sym_split[0]
        m = float(sym_split[1])
        sd = float(sym_split[2])
        var = int(sym_split[3])
        syms_out.append([[m, sd], var, name])

    return syms_out


def generate_trajectory(skill_name, dmp_folder, symbols, workspace_bnds, suggestions_pre, suggestions_post, folder_save, opts):
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


def generate_trajectory_baxter(skill_name, dmp_folder, symbols, workspace_bnds, suggestions_pre, suggestions_post, folder_save, opts):
    os.makedirs(folder_save, exist_ok=True)
    model = DMPNN(opts['start_dimension'], 1024, opts['dimension'], opts['basis_fs']).to(DEVICE)
    model.load_state_dict(torch.load(dmp_folder + skill_name + ".pt"))
    pre_cons = constraints.States(symbols, suggestions_pre, epsilon=opts['epsilon'], buffer=0.2)
    post_cons = constraints.States(symbols, suggestions_post, epsilon=opts['epsilon'], buffer=0.2)
    true_sym_start_bnds = np.zeros([opts['dimension'], 2])
    for sym, val in suggestions_pre[np.random.randint(len(suggestions_pre))].items():
        if val:
            true_sym_start_bnds[symbols[sym].dims, :] = symbols[sym].bounds[symbols[sym].dims, :]
            # true_sym_start_bnds = symbols[sym].bounds
    true_sym_end_bnds = np.zeros([opts['dimension'], 2])
    for sym, val in suggestions_post[np.random.randint(len(suggestions_post))].items():
        if val:
            true_sym_end_bnds[symbols[sym].dims, :] = symbols[sym].bounds[symbols[sym].dims, :]
            # true_sym_end_bnds = symbols[sym].bounds
    start_pose_in = true_sym_start_bnds[:, 0] + 0.2 * (true_sym_start_bnds[:, 1] - true_sym_start_bnds[:, 0]) + 0.6 * np.random.random(opts['dimension']) * (true_sym_start_bnds[:, 1] - true_sym_start_bnds[:, 0])
    prec = pre_cons.condition(ltd.TermStatic(torch.from_numpy(start_pose_in[np.newaxis, :])))
    while not prec.satisfy(0):
        start_pose_in = true_sym_start_bnds[:, 0] + 0.2 * (
                    true_sym_start_bnds[:, 1] - true_sym_start_bnds[:, 0]) + 0.6 * np.random.random(
            opts['dimension']) * (true_sym_start_bnds[:, 1] - true_sym_start_bnds[:, 0])
        prec = pre_cons.condition(ltd.TermStatic(torch.from_numpy(start_pose_in[np.newaxis, :])))

    end_pose_in = true_sym_end_bnds[:, 0] + np.random.random(opts['dimension']) * (true_sym_end_bnds[:, 1] - true_sym_end_bnds[:, 0])
    postc = post_cons.condition(ltd.TermStatic(torch.from_numpy(end_pose_in[np.newaxis, :])))
    while not postc.satisfy(0):
        end_pose_in = true_sym_end_bnds[:, 0] + np.random.random(opts['dimension']) * (
                true_sym_end_bnds[:, 1] - true_sym_end_bnds[:, 0])
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


def generate_trajectory_find(skill_name, dmp_folder, symbols, workspace_bnds, suggestions_pre, suggestions_post, folder_save, opts):
    # For the stretch. Temporarily making the robot only move straight
    os.makedirs(folder_save, exist_ok=True)
    model = DMPNN(opts['start_dimension'], 1024, opts['dimension'], opts['basis_fs']).to(DEVICE)
    model.load_state_dict(torch.load(dmp_folder + skill_name + ".pt"))
    pre_cons = constraints.States(symbols, suggestions_pre, epsilon=opts['epsilon'], buffer=0.05)
    post_cons = constraints.States(symbols, suggestions_post, epsilon=opts['epsilon'], buffer=0.05)
    start_pose_in = workspace_bnds[:, 0] + (np.random.random(opts['dimension'])) * (workspace_bnds[:, 1] - workspace_bnds[:, 0])
    prec = pre_cons.condition(ltd.TermStatic(torch.from_numpy(start_pose_in[np.newaxis, :])))
    while not prec.satisfy(0):
        start_pose_in = workspace_bnds[:, 0] + np.random.random(opts['dimension']) * (
                    workspace_bnds[:, 1] - workspace_bnds[:, 0])
        prec = pre_cons.condition(ltd.TermStatic(torch.from_numpy(start_pose_in[np.newaxis, :])))

    end_pose_in = workspace_bnds[:, 0] + (0.1 + 0.8 * np.random.random(opts['dimension'])) * (workspace_bnds[:, 1] - workspace_bnds[:, 0])
    end_pose_in[[1, 2, 3, 4, 5]] = start_pose_in[[1, 2, 3, 4, 5]]

    # end_pose_in = np.copy(start_pose_in)
    # end_pose_in[0, 0] += 1.5
    postc = post_cons.condition(ltd.TermStatic(torch.from_numpy(end_pose_in[np.newaxis, :])))
    while not postc.satisfy(0):
        end_pose_in = workspace_bnds[:, 0] + np.random.random(opts['dimension']) * (
                    workspace_bnds[:, 1] - workspace_bnds[:, 0])
        end_pose_in[[1, 2, 3, 4, 5]] = start_pose_in[[1, 2, 3, 4, 5]]
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


def run_elaborateDMP(old_skill, new_skill, suggestion, hard_constraints, symbols, workspace_bnds, opts):
    results_root = opts['base_folder'] + "/logs/generalized-exps-{}".format(t_stamp())
    os.makedirs(opts['demo_folder'], exist_ok=True)
    os.makedirs(opts['demo_folder'] + "/train", exist_ok=True)
    os.makedirs(opts['demo_folder'] + "/val", exist_ok=True)

    if opts['enforce_type'] == "unconstrained":
        enforce, adversarial = False, False
    if opts['enforce_type'] == "train":
        enforce, adversarial = True, False
    if opts['enforce_type'] == "adversarial":
        enforce, adversarial = True, True

    if opts['enforce_type'] == 'adversarial':
        # Find the trajectory from when the block is grasped
        for ii in range(opts['n_train_trajs'] + opts['n_val_trajs']):
            if ii < opts['n_train_trajs']:
                folder_train_val = 'train'
            else:
                folder_train_val = 'val'
            opts['f_name_add'] = str(ii)
            generate_trajectory_find(old_skill, opts['dmp_folder'], symbols, workspace_bnds,
                                suggestion['initial_preconditions'], suggestion['final_postconditions'],
                                opts['demo_folder'] + "/" + folder_train_val, opts)

    # t_start_states, t_pose_hists = load_dmp_demos(opts['demo_folder'] + "/train", n_interp_pts=int(1/opts['dt']), opts=opts)
    t_start_states, t_pose_hists = load_dmp_demos(opts['demo_folder'] + "/train")
    t_start_states, t_pose_hists = np_to_pgpu(t_start_states), np_to_pgpu(t_pose_hists)
    train_set = TensorDataset(t_start_states, t_pose_hists)

    # v_start_states, v_pose_hists = load_dmp_demos(opts['demo_folder'] + "/val", n_interp_pts=int(1/opts['dt']), opts=opts)
    v_start_states, v_pose_hists = load_dmp_demos(opts['demo_folder'] + "/val")
    v_start_states, v_pose_hists = np_to_pgpu(v_start_states), np_to_pgpu(v_pose_hists)
    val_set = TensorDataset(v_start_states, v_pose_hists)

    results_folder = join(results_root, "{}-{}-{}".format(new_skill, opts['enforce_type'], opts['c_weight']))
    os.makedirs(results_folder, exist_ok=True)

    if opts['enforce_type'] == 'unconstrained':
        constraint_list = [None]
        intermediate_constraints = [None]
    else:
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
        for constraint_type in opts['constraints']:
            constraint_list.append(constraints.AutomaticSkill(symbols_device, suggestion['intermediate_states_all_pres'],
                                                    suggestion['intermediate_states'],
                                                    suggestion['final_postconditions'], suggestion['unique_states'],
                                                    suggestion['avoid_states'], workspace_bnds_device, opts['epsilon'], constraint_type))
        intermediate_constraints = []
        for suggestion_intermediate_all_posts in suggestion['intermediate_states']:
            intermediate_constraints.append(constraints.AutomaticIntermediateSteps(symbols_device,
                                                                                   suggestion_intermediate_all_posts,
                                                                                   suggestion['unique_states'],
                                                                                   suggestion['avoid_states'],
                                                                                   workspace_bnds_device,
                                                                                   opts['epsilon']))

    learned_model, val_losses, intermediate_sat = training_loop(train_set, val_set, constraint_list, enforce, adversarial, t_pose_hists, results_folder, intermediate_constraints, opts)

    return learned_model, val_losses, intermediate_sat


def ik(limb, pose, seed_angles=None, b=0.001, br=0.01, qinit_in=None, solve_type="Speed"):
    """
    limb: which limb
    pose: Pose msg
    returns: joints and joint angles
    """

    urdf = rospy.get_param('/robot_description')
    ik_solver = IK("base", limb + "_gripper", urdf_string=urdf, timeout=0.1, epsilon=1e-5, solve_type=solve_type)
    bx = by = bz = b
    brx = bry = brz = br
    if limb == 'left':
        joint_names = ['left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
    else:
        joint_names = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']

    if type(pose) == torch.Tensor or type(pose) == np.ndarray:
        x = float(pose[0])
        y = float(pose[1])
        z = float(pose[2])
        # rx = ry = rz = 0.
        # rw = -1.
        rx = -0.039
        ry = 0.998
        rz = 0.049
        rw = -0.030
    else:
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z
        rx = pose.orientation.x
        ry = pose.orientation.y
        rz = pose.orientation.z
        rw = pose.orientation.w

    if qinit_in is None:
        qinit = [0.] * 7
    else:
        qinit = qinit_in

    # print("Solving for one pose")
    # print(f"qinit {qinit}")
    # print(f"x: {x}, y: {y}, z: {z}, rx: {rx}, ry: {ry}, rz: {rz}, rw: {rw}")
    # print(f"bx: {bx}, by: {by}, bz: {bz}, brx: {brx}, bry: {bry}, brz: {brz}")
    # print("Internal joint names: {}".format(ik_solver.joint_names))
    sol = ik_solver.get_ik(qinit,
                           x, y, z,
                           rx, ry, rz, rw,
                           bx, by, bz,
                           brx, bry, brz)

    if sol is not None and len(sol) > 0:
        limb_joints = dict(zip(joint_names, sol))
    else:
        print("No valid joint solution found")
        limb_joints = dict(zip(joint_names, [np.NaN] * len(joint_names)))
        # raise Exception("No valid joint solution found")

    return limb_joints


def ik_trajs(limb, poses, b=0.001, br=0.01, qinit_in=None, solve_type="Speed"):
    n_trajs, n_points, _ = poses.shape
    # print("Setting n_traj to 1 for debugging")
    # n_trajs = 1
    # n_points = 2
    out_joints = np.zeros([n_trajs, n_points, 7])
    joint_names = []
    print("Beginning solve for ik")
    for e in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']:
        joint_names.append(limb + "_" + e)
    for ii in range(n_trajs):
        print("Trajectory {}".format(ii))
        qinit = qinit_in
        if qinit_in is None:
            qinit = [0.] * 7
        # qinit = [0.] * 7
        for jj in range(n_points):
            print("Time point: {}".format(jj))
            pose = poses[ii, jj, :]
            print("Poses going in: {}".format(pose))
            joints = ik(limb, pose, b=b, br=br, qinit_in=qinit, solve_type=solve_type)
            # print("Found joints: {}".format(joints))
            for kk, joint_name in enumerate(joint_names):
                out_joints[ii, jj, kk] = joints[joint_name]
                qinit[kk] = joints[joint_name]

    return out_joints


class StateValidity():
    # Adapted from https://answers.ros.org/question/203633/collision-detection-in-python/
    def __init__(self):
        # subscribe to joint joint states
        rospy.Subscriber("joint_states", JointState, self.jointStatesCB, queue_size=1)
        # prepare service for collision check
        self.sv_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        # wait for service to become available
        self.sv_srv.wait_for_service()
        rospy.loginfo('service is avaiable')
        # prepare msg to interface with moveit
        self.rs = RobotState()
        joint_state_names = []
        # Need to include left and right arm to find when they intersect
        for a in ['left', 'right']:
            for e in ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']:
                joint_state_names.append(a + "_" + e)
        self.rs.joint_state.name = joint_state_names
        self.rs.joint_state.position = [0.] * len(self.rs.joint_state.name)
        self.joint_states_received = False

    def jointStatesCB(self, msg):
        '''
        update robot state
        '''
        self.rs.joint_state.position = [msg.position[0], msg.position[1]]
        self.joint_states_received = True

    def getStateValidity(self, joint_positions, constraints=None):
        '''
        Given a RobotState and a group name and an optional Constraints
        return the validity of the State
        '''
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = self.rs
        gsvr.robot_state.joint_state.position = joint_positions
        gsvr.group_name = 'baxter'
        if constraints != None:
            gsvr.constraints = constraints
        result = self.sv_srv.call(gsvr)
        return result
