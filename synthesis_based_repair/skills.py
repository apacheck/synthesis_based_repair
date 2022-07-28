#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from synthesis_based_repair.tools import dict_to_formula, pre_posts_to_env_formula, sym_list_to_dict
import os
from synthesis_based_repair.symbols import in_symbols, find_symbols_true_and_false, load_symbols, plot_symbolic_state
import json


class Skill:

    def __init__(self, info, suggestion=False, stretch_eff=0.23):
        self.name = info['name']
        self.suggestion = suggestion
        self.stretch_eff = stretch_eff
        if suggestion:
            self.original_skill = info['original_skill']
            self.new_skill = info['new_skill']
        self.intermediate_states = info['intermediate_states']
        self.init_pres = info['initial_preconditions']
        self.final_posts = info['final_postconditions']
        # self.unique = info['unique_states']
        self.folder_train = info['folder_train']
        self.folder_val = info['folder_val']

    def get_name(self):
        return self.name

    def get_skill_str(self, include_false=True):
        out = "Skill name: " + self.name + "\n"
        if self.suggestion:
            out += "Original name: " + dict_to_formula(self.original_skill, include_false=False) + "\n"
        out += "Initial preconditions: \n"
        for init_pre in self.init_pres:
            out += "{}\n".format(dict_to_formula(init_pre, prime=False, include_false=include_false))
        for int_state in self.intermediate_states:
            out += "{}\n".format(
                pre_posts_to_env_formula(self.name, int_state[0], int_state[1]))
        out += "Final Postconditions:\n"
        for post in self.final_posts:
            out += "{}\n".format(dict_to_formula(post))

        return out

    def write_skill_str(self, file_name, include_false=True):
        fid = open(file_name, "a")
        fid.write(self.get_skill_str(include_false=include_false))
        fid.write('**********************\n')
        fid.close()

    def stretch_joints_to_cartesian(self, data):
        """ Convert the stretch joint angles to cartesian end effector positions"""
        # x, y, theta, arm length, z, wrist theta
        x = data[:, 0]
        y = data[:, 1]
        theta = data[:, 2]
        arm = data[:, 3]
        z = data[:, 4]
        theta_wrist = data[:, 5]

        x_ee = self.stretch_eff * np.cos(theta + theta_wrist - np.pi/2) + arm * np.cos(theta - np.pi/2) + x
        y_ee = self.stretch_eff * np.sin(theta + theta_wrist - np.pi/2) + arm * np.sin(theta - np.pi/2) + y
        z_ee = z

        out = np.hstack((x_ee[:, np.newaxis], y_ee[:, np.newaxis], z_ee[:, np.newaxis]))
        return out

    def plot(self, ax, data, **kwargs):
        for d in data:
            if d.shape[1] == 2:
                ax.plot(d[:, 0], d[:, 1], **kwargs)
            elif d.shape[1] == 3:
                ax.plot(d[:, 0], d[:, 1], d[:, 2], **kwargs)
            elif d.shape[1] == 6:
                ee = self.stretch_joints_to_cartesian(d)
                ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], **kwargs)
                ax.plot(d[:, 0], d[:, 1], np.zeros(ee[:, 1].shape), **kwargs)

    def plot_original(self, ax, train=True, **kwargs):
        if train:
            folder_trajectories = self.folder_train
        else:
            folder_trajectories = self.folder_val
        files_folder = [f for f in os.listdir(folder_trajectories) if
                        os.path.isfile(os.path.join(folder_trajectories, f))]
        files_traj = [f for f in files_folder if 'rollout' in f]
        data = np.stack([np.loadtxt(os.path.join(folder_trajectories, rp), ndmin=2) for rp in files_traj])

        self.plot(ax, data, **kwargs)

    def get_original_trajectories(self, train=True, **kwargs):
        if train:
            folder_trajectories = self.folder_train
        else:
            folder_trajectories = self.folder_val
        files_folder = [f for f in os.listdir(folder_trajectories) if
                        os.path.isfile(os.path.join(folder_trajectories, f))]
        files_traj = [f for f in files_folder if 'rollout' in f]
        data = np.stack([np.loadtxt(os.path.join(folder_trajectories, rp), ndmin=2) for rp in files_traj])

        return data

    def plot_nice(self, ax, plot_limits, symbols, train=True, idx=0, **kwargs):
        if train:
            folder_trajectories = self.folder_train
        else:
            folder_trajectories = self.folder_val
        files_folder = [f for f in os.listdir(folder_trajectories) if
                        os.path.isfile(os.path.join(folder_trajectories, f))]
        files_traj = [f for f in files_folder if 'rollout' in f]
        data = np.stack([np.loadtxt(os.path.join(folder_trajectories, rp), ndmin=2) for rp in files_traj])
        d = data[idx, :, :]
        ee = self.stretch_joints_to_cartesian(d)

        ax.set_xlim(plot_limits[0])
        ax.set_ylim(plot_limits[1])
        ax.set_zlim(plot_limits[2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        for sym_name, sym in symbols.items():
            sym.plot(ax, dim=3, fill=True, lw=1, alpha=0.1)

        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color='g')
        ax.plot(ee[:, 0], ee[:, 1], np.zeros(ee[:, 1].shape), ':', color='g')
        ax.plot(d[:, 0], d[:, 1], np.zeros(ee[:, 1].shape), color='r')
        for ii in range(0, np.shape(d)[0], 20):
            ax.plot(np.array([d[ii, 0], d[ii, 0]]), np.array([d[ii, 1], d[ii, 1]]), np.array([0, 0.4]), color='k')
            ax.plot(np.array([d[ii, 0], ee[ii, 0]]), np.array([d[ii, 1], ee[ii, 1]]), np.array([d[ii, 4], d[ii, 4]]), color='k')

    def sample_start_end_states(self, limits, sym_defs, n_points=32, n_rows=2, dim=2):
        out_points = np.zeros([n_rows, dim, n_points])

        # Preconditions
        for ii in range(n_points):
            point = limits[:, 0] + np.random.random(dim) * (limits[:, 1] - limits[:, 0])
            pre_dict = self.init_pres[np.random.randint(len(self.init_pres))]
            while not in_symbols(point, pre_dict, sym_defs):
                point = limits[:, 0] + np.random.random(dim) * (limits[:, 1] - limits[:, 0])
            out_points[0, :, ii] = point

        # Postconditions
        for ii in range(n_points):
            point = limits[:, 0] + np.random.random(dim) * (limits[:, 1] - limits[:, 0])
            post_dict = self.init_pres[np.random.randint(len(self.final_posts))]
            while not in_symbols(point, post_dict, sym_defs):
                point = limits[:, 0] + np.random.random(dim) * (limits[:, 1] - limits[:, 0])
            out_points[1, :, ii] = point

        return out_points

    def plot_dmp(self, ax, start_end_states):
        pass

    def get_skill_dict(self):
        pass

    def to_json(self):
        outdict = dict()
        outdict['name'] = self.name
        if self.suggestion:
            outdict['original_skill'] = self.original_skill
            outdict['new_skill'] = self.new_skill
        outdict['intermediate_states'] = self.intermediate_states
        outdict['initial_preconditions'] = self.init_pres
        outdict['final_postconditions'] = self.final_posts
        outdict['folder_train'] = self.folder_train
        outdict['folder_val'] = self.folder_val
        return outdict

    def get_intermediate_states(self):
        return self.intermediate_states

    def get_final_posts(self):
        return self.final_posts

    def get_initial_pres(self):
        return self.init_pres

    def is_suggestion(self):
        return self.suggestion

    def plot_symbolic_skill(self, symbols, xlims, ylims, **kwargs):
        ncols = 2
        for _, posts in self.intermediate_states:
            ncols = max([ncols, len(posts) + 1])
        nrows = len(self.intermediate_states)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        ax[0, 0].set_title("Precondition")
        for ii in range(1, ncols):
            ax[ii, 0].set_title("Postcondition")
        for ii, (pre, posts) in enumerate(self.intermediate_states):
            plot_symbolic_state(pre, symbols, ax[nrows-(ii+1), 0], xlims, ylims, **kwargs)
            ax[ii, 0].set_xticks([])
            ax[ii, 0].set_yticks([])
            for jj, post in enumerate(posts):
                plot_symbolic_state(post, symbols, ax[nrows-(ii+1), jj+1], xlims, ylims, **kwargs)
                ax[ii, jj+1].set_xticks([])
                ax[ii, jj+1].set_yticks([])
        return fig, ax

    def get_ee_final_symbol(self):
        for ee in ['ee_table_1', 'ee_table_2', 'ee_table_3']:
            if self.final_posts[0][ee]:
                return ee

    def get_ee_final_region(self):
        ee = self.get_ee_final_symbol()
        return ee.split("_")[-1]

    def get_final_robot_pose(self, joint_state, world_state, symbols):
        world_state = np.squeeze(world_state)
        if self.name.split("_")[0] in ['skillStretch3to1', 'skillStretch1to2', 'skillStretch2to3']:
            ee_final_region = self.get_ee_final_region()
            ee_final_symbol = self.get_ee_final_symbol()
            if self.final_posts[0]['duck_a_' + ee_final_region] and self.final_posts[0]['duck_a_table']:
                final_ee_xyz = np.copy(world_state[6:9])
                final_ee_xyz[2] += 0.2
            elif self.final_posts[0]['duck_b_' + ee_final_region] and self.final_posts[0]['duck_b_table']:
                final_ee_xyz = np.copy(world_state[9:12])
                final_ee_xyz[2] += 0.2
            else:
                final_ee_xyz = symbols[ee_final_symbol].sample_from()
            final_robot = np.zeros([6])
            if ee_final_region == "1":
                final_robot[0] = final_ee_xyz[0] - 0.04
                final_robot[1] = final_ee_xyz[1] - 0.75
                final_robot[2] = np.pi
            elif ee_final_region == "2":
                final_robot[0] = final_ee_xyz[0] + 0.75
                final_robot[1] = final_ee_xyz[1] - 0.04
                final_robot[2] = 3 * np.pi / 2
            elif ee_final_region == "3":
                final_robot[0] = final_ee_xyz[0] + 0.04
                final_robot[1] = final_ee_xyz[1] + 0.75
                final_robot[2] = 0
            final_robot[3] = 0.75 - self.stretch_eff
            final_robot[4] = 0.9
            final_robot[5] = 0
        else:
            final_robot = np.copy(joint_state)

        return final_robot


def write_skills_str(skills, file_str, only_suggestions=False, include_false=False):
    for _, skill in skills.items():
        if only_suggestions and not skill.is_suggestion():
            continue
        skill.write_skill_str(file_str, include_false=include_false)

    fid = open(file_str, 'a')
    fid.write("================================\n")
    fid.close()

def write_skills_json(skills, file_json):
    outdict = dict()
    for skill_name, skill in skills.items():
        outdict[skill_name] = skill.to_json()
    with open(file_json, "w") as outfile:
        json.dump(outdict, outfile, indent=1)


def find_one_skill_intermediate_states(arg_folder_traj, arg_symbols):
    files_folder = [f for f in os.listdir(arg_folder_traj) if os.path.isfile(os.path.join(arg_folder_traj, f))]
    files_traj = [f for f in files_folder if 'rollout' in f]

    intermediate_states = []
    poss_changes = []

    unique_traj = []
    for f in files_traj:
        data = np.loadtxt(arg_folder_traj + "/" + f, delimiter=" ", dtype=float)
        traj_syms = find_traj_in_syms(data, arg_symbols)
        traj_syms = reduce_sym_traj(traj_syms)
        if traj_syms not in unique_traj:
            unique_traj.append(traj_syms)
        for ii in range(0, len(traj_syms) - 1):
            if [traj_syms[ii], [traj_syms[ii+1]]] not in poss_changes:
                poss_changes.append([traj_syms[ii], [traj_syms[ii + 1]]])

    # print("unique trajs " + arg_folder_traj)
    # for u in unique_traj:
    #     for s in u:
    #         print(s)
    #     print("*************")

    for poss_change in poss_changes:
        pre_already_entered = False
        for ii, (pre, posts) in enumerate(intermediate_states):
            if pre == poss_change[0]:
                pre_already_entered = True
                post_entered = False
                for post in posts:
                    if post == poss_change[1][0]:
                        post_entered = True
                if not post_entered:
                    intermediate_states[ii][1].append(poss_change[1][0])
                break
        if not pre_already_entered:
            intermediate_states.append(poss_change)

    return intermediate_states


def find_unique_states(arg_folder_traj, arg_symbols):
    files_folder = [f for f in os.listdir(arg_folder_traj) if os.path.isfile(os.path.join(arg_folder_traj, f))]
    files_traj = [f for f in files_folder if 'rollout' in f]

    intermediate_states = []
    poss_changes = []

    unique_states = []
    for f in files_traj:
        data = np.loadtxt(arg_folder_traj + "/" + f, delimiter=" ", dtype=float)
        sym_traj = find_traj_in_syms(data, arg_symbols)
        reduced_traj = reduce_sym_traj(sym_traj)
        for r in reduced_traj:
            if r not in unique_states:
                unique_states.append(r)

    return unique_states


def find_traj_in_syms(arg_data, arg_symbols):
    """
    Finds what symbols are true and false at each state in a trajectory

    :param arg_data:
    :param arg_symbols:
    :return:
    """
    syms_out = []
    for ii, d in enumerate(arg_data):
        syms_out.append(find_symbols_true_and_false(d, arg_symbols))

    return syms_out


def reduce_sym_traj(sym_traj):
    """
    Reduces a list of symbolic states to just the ones that change

    Example: reduce_sym_traj([{'a': True, 'b':False}, {'a':True, 'b':False}, {'a':True, 'b':True}])
    returns [{'a':True, 'b':False}, {'a':True, 'b':True}]

    :param sym_traj:
    :return:
    """
    sym_sequence = [sym_traj[0]]
    for sym_state in sym_traj:
        if sym_state != sym_sequence[-1]:
            sym_sequence.append(sym_state)

    return sym_sequence


def remove_mutually_exclusive_symbols_one_state(sym_state, mx_symbols):
    out = dict()
    for sym, truth_value in sym_state.items():
        if truth_value or sym not in mx_symbols:
            out[sym] = truth_value

    return out


def remove_mutually_exclusive_symbols_list_of_states(sym_states, mx_symbols):
    out = []
    for sym_state in sym_states:
        tmp = remove_mutually_exclusive_symbols_one_state(sym_state, mx_symbols)
        out.append(tmp)

    return out


def remove_mutually_exclusive_symbols_intermediate_states(sym_states, mx_symbols):
    out = []
    for pre, posts in sym_states:
        tmp_pre = remove_mutually_exclusive_symbols_one_state(pre, mx_symbols)
        post_list = []
        for post in posts:
            tmp_post = remove_mutually_exclusive_symbols_one_state(post, mx_symbols)
            post_list.append(tmp_post)
        out.append([tmp_pre, post_list])

    return out

def find_one_skill_pre_or_post(arg_folder_traj, arg_symbols, find_pre):

    files_folder = [f for f in os.listdir(arg_folder_traj) if os.path.isfile(os.path.join(arg_folder_traj, f))]
    files_traj = [f for f in files_folder if 'rollout' in f]

    syms_lists = []

    if find_pre:
        idx = 0
    else:
        idx = -1

    for f in files_traj:
        data = np.loadtxt(arg_folder_traj + "/" + f, delimiter=" ", dtype=float)
        traj_syms = find_traj_in_syms(data, arg_symbols)
        if traj_syms[idx] not in syms_lists:
            syms_lists.append(traj_syms[idx])

    return syms_lists


def load_skills_from_trajectories(folder_trajectory_base, skill_names, sym_defs):
    skills = dict()
    for skill_name in skill_names:
        skill = dict()
        skill['folder_train'] = folder_trajectory_base + '/' + skill_name + "/train/"
        skill['folder_val'] = folder_trajectory_base + '/' + skill_name + "/val/"
        folder_trajectories = skill['folder_train']
        skill['name'] = skill_name
        skill['intermediate_states'] = find_one_skill_intermediate_states(folder_trajectories, sym_defs)
        skill['initial_preconditions'] = find_one_skill_pre_or_post(folder_trajectories, sym_defs, True)
        skill['final_postconditions'] = find_one_skill_pre_or_post(folder_trajectories, sym_defs, False)
        for final_post in skill['final_postconditions']:
            for ii, (pre, posts) in enumerate(skill['intermediate_states']):
                if final_post == pre:
                    skill['intermediate_states'][ii][1].append(final_post)

        skills[skill_name] = Skill(skill)

    return skills


def load_skills_from_json(file_json):
    fid = open(file_json, 'r')
    data = json.load(fid)
    fid.close()

    skills =  dict()
    for skill_name, skill in data.items():
        skills[skill_name] = Skill(skill)

    return skills


if __name__ == "__main__":
    folder_trajectories = '../data/nine_squares/trajectories/'
    skill_names = ["skill0", 'skill1', 'skill2', 'skill3', 'skill4', 'skill5', 'skill6', 'skill7', 'skill8', 'skill9', 'skill10']
    f_symbols = "../data/nine_squares/nine_squares_symbols.json"
    f_plot = "../data/nine_squares/plots/"
    f_skills = "../data/nine_squares/nine_squares_skills.json"

    # folder_trajectories = '../data/stretch/trajectories/'
    # skill_names = ["skillStretch0", 'skillStretch1']
    # f_symbols = "../data/stretch/stretch_symbols.json"
    # f_plot = "../data/stretch/plots/"
    # f_skills = "../data/stretch/stretch_skills.json"

    symbols = load_symbols(f_symbols)
    skills = load_skills_from_trajectories(folder_trajectories, skill_names, symbols)
    for skill_name, skill in skills.items():
        fig, ax = plt.subplots()
        skill.plot_original(ax)
        ax.set_xlim([-0.5, 3.5])
        ax.set_ylim([-0.5, 3.5])
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2, 3])
        plt.savefig(f_plot + skill_name + ".png")
        plt.close()
    write_skills_json(skills, f_skills)
