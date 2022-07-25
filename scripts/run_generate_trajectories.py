#!/usr/bin/env python
import os
import numpy as np
from synthesis_based_repair.physical_implementation import run_elaborateDMP
from synthesis_based_repair.symbols import load_symbols
import argparse
from synthesis_based_repair.tools import json_load_wrapper
from synthesis_based_repair.skills import load_skills_from_trajectories, write_skills_json
import matplotlib.pyplot as plt
from dl2_lfd.dmps.dmp import load_dmp_demos, DMP



def generate_trajectories_nine_squares(folder_demo_trajectories, n_train_trajs, n_val_trajs, n_start_rows, skill_names):
    make_folders(folder_demo_trajectories, skill_names)

    # bottom left to top right via bottom right
    skill_name = "skill0"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        left = np.array([0.2, 0.2]) + np.random.random(2) * .6
        top_right = np.array([2.2, 2.2]) + np.random.random(2) * .6
        bottom_right = np.array([2.2, 0.2]) + np.random.random(2) * .6
        data = np.hstack([np.vstack([np.linspace(left[0], top_right[0], 50), np.repeat(left[1], 50)]),
                          np.vstack([np.repeat(top_right[0], 50), np.linspace(left[1], top_right[1], 50)])]).transpose()
        save_data(folder_demo_skill, data, left, top_right, ii, n_start_rows, n_train_trajs)

    # top right to bottom left via top left
    skill_name = "skill1"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        left = np.array([0.2, 0.2]) + np.random.random(2) * .6
        top_right = np.array([2.2, 2.2]) + np.random.random(2) * .6
        bottom_right = np.array([2.2, 0.2]) + np.random.random(2) * .6
        data = np.hstack([np.vstack([np.linspace(top_right[0], left[0], 50), np.repeat(top_right[1], 50)]),
                          np.vstack([np.repeat(left[0], 50), np.linspace(top_right[1], left[1], 50)])]).transpose()
        save_data(folder_demo_skill, data, top_right, left, ii, n_start_rows, n_train_trajs)

    # Skill 2: bottom left to bottom right
    skill_name = "skill2"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        left = np.array([0.2, 0.2]) + np.random.random(2) * .6
        top_right = np.array([2.2, 2.2]) + np.random.random(2) * .6
        bottom_right = np.array([2.2, 0.2]) + np.random.random(2) * .6
        data = np.hstack([np.vstack([np.linspace(left[0], bottom_right[0], 100), np.repeat(left[1], 100)])]).transpose()
        save_data(folder_demo_skill, data, left, bottom_right, ii, n_start_rows, n_train_trajs)

    # Skill 3: bottom right to top right
    skill_name = "skill3"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        left = np.array([0.2, 0.2]) + np.random.random(2) * .6
        top_right = np.array([2.2, 2.2]) + np.random.random(2) * .6
        bottom_right = np.array([2.2, 0.2]) + np.random.random(2) * .6
        data = np.hstack([np.vstack([np.repeat(top_right[0], 100), np.linspace(bottom_right[1], top_right[1], 100)])]).transpose()
        save_data(folder_demo_skill, data, bottom_right, top_right, ii, n_start_rows, n_train_trajs)

    # Skill 4: bottom left to (bottom right or middle)
    skill_name = "skill4"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        left_top = np.array([0.2, 0.5]) + np.multiply(np.random.random(2), np.array([0.6, 0.3]))
        left_bottom = np.array([0.2, 0.2]) + np.multiply(np.random.random(2), np.array([0.6, 0.3]))
        middle = np.array([1.2, 1.2]) + np.random.random(2) * .6
        top_right = np.array([2.2, 2.2]) + np.random.random(2) * .6
        bottom_right = np.array([2.2, 0.2]) + np.random.random(2) * .6
        if ii % 2 == 0:
            data = np.hstack([np.vstack([np.linspace(left_top[0], middle[0], 50), np.repeat(left_top[1], 50)]),
                              np.vstack(
                                  [np.repeat(middle[0], 50), np.linspace(left_top[1], middle[1], 50)])]).transpose()
            save_data(folder_demo_skill, data, left_bottom, middle, ii, n_start_rows, n_train_trajs)
        else:
            data = np.hstack(
                [np.vstack([np.linspace(left_bottom[0], bottom_right[0], 100), np.repeat(left_bottom[1], 100)])]).transpose()
            save_data(folder_demo_skill, data, left_bottom, bottom_right, ii, n_start_rows, n_train_trajs)

    # Skill 5: (middle or bottom right) to top_right
    skill_name = "skill5"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        left = np.array([0.2, 0.2]) + np.random.random(2) * .6
        middle = np.array([1.2, 1.2]) + np.random.random(2) * .6
        top_right_left = np.array([2.2, 2.2]) + np.multiply(np.random.random(2), np.array([0.3, 0.6]))
        top_right_right = np.array([2.5, 2.2]) + np.multiply(np.random.random(2), np.array([0.3, 0.6]))
        middle_right = np.array([2.2, 1.2]) + np.random.random(2) * .6
        bottom_right = np.array([2.2, 0.2]) + np.random.random(2) * .6
        if ii % 2 == 0:
            data = np.hstack([np.vstack([np.linspace(middle[0], top_right_left[0], 50), np.repeat(middle[1], 50)]),
                              np.vstack(
                                  [np.repeat(top_right_left[0], 50), np.linspace(middle[1], top_right_left[1], 50)])]).transpose()
            save_data(folder_demo_skill, data, middle, top_right, ii, n_start_rows, n_train_trajs)
        else:
            data = np.hstack(
                [np.vstack([np.repeat(top_right_right[0], 100), np.linspace(bottom_right[1], top_right_right[1], 100)])]).transpose()
            save_data(folder_demo_skill, data, bottom_right, top_right, ii, n_start_rows, n_train_trajs)

    # Skill 6: bottom_left to (middle or bottom right) to top_right
    skill_name = "skill6"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        left_top = np.array([0.2, 0.5]) + np.multiply(np.random.random(2), np.array([0.6, 0.3]))
        left_bottom = np.array([0.2, 0.2]) + np.multiply(np.random.random(2), np.array([0.6, 0.3]))
        middle = np.array([1.2, 1.2]) + np.random.random(2) * .6
        top_right_left = np.array([2.2, 2.2]) + np.multiply(np.random.random(2), np.array([0.3, 0.6]))
        top_right_right = np.array([2.5, 2.2]) + np.multiply(np.random.random(2), np.array([0.3, 0.6]))
        middle_right = np.array([2.2, 1.2]) + np.random.random(2) * .6
        bottom_right = np.array([2.2, 0.2]) + np.random.random(2) * .6
        if ii % 2 == 0:
            data = np.hstack([np.vstack([np.linspace(left_top[0], middle[0], 25), np.repeat(left_top[1], 25)]),
                              np.vstack(
                                  [np.repeat(middle[0], 25), np.linspace(left_top[1], middle[1], 25)]),
                              np.vstack([np.linspace(middle[0], top_right_left[0], 25), np.repeat(middle[1], 25)]),
                              np.vstack(
                                  [np.repeat(top_right_left[0], 25),
                                   np.linspace(middle[1], top_right_left[1], 25)])]).transpose()
            save_data(folder_demo_skill, data, left_bottom, top_right_left, ii, n_start_rows, n_train_trajs)
        else:
            data = np.hstack([np.vstack([np.linspace(left_bottom[0], top_right_right[0], 50), np.repeat(left_bottom[1], 50)]),
                              np.vstack(
                                  [np.repeat(top_right_right[0], 50), np.linspace(left_bottom[1], top_right_right[1], 50)])]).transpose()
            save_data(folder_demo_skill, data, left_bottom, top_right_right, ii, n_start_rows, n_train_trajs)

    # Skill 7: top_right to (middle or top_left) to bottom_right
    skill_name = "skill7"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        left_left = np.array([0.2, 0.2]) + np.multiply(np.random.random(2), np.array([0.3, 0.6]))
        left_right = np.array([0.5, 0.2]) + np.multiply(np.random.random(2), np.array([0.3, 0.6]))
        middle = np.array([1.2, 1.2]) + np.random.random(2) * .6
        top_right_bottom = np.array([2.2, 2.2]) + np.multiply(np.random.random(2), np.array([0.6, 0.3]))
        top_right_top = np.array([2.2, 2.5]) + np.multiply(np.random.random(2), np.array([0.6, 0.3]))
        middle_right = np.array([2.2, 1.2]) + np.random.random(2) * .6
        bottom_right = np.array([2.2, 0.2]) + np.random.random(2) * .6
        if ii % 2 == 0:
            data = np.hstack([np.vstack([np.linspace(top_right_bottom[0], middle[0], 25), np.repeat(top_right_bottom[1], 25)]),
                              np.vstack(
                                  [np.repeat(middle[0], 25), np.linspace(top_right_bottom[1], middle[1], 25)]),
                              np.vstack([np.linspace(middle[0], left_right[0], 25), np.repeat(middle[1], 25)]),
                              np.vstack(
                                  [np.repeat(left_right[0], 25),
                                   np.linspace(middle[1], left_right[1], 25)])]).transpose()
            save_data(folder_demo_skill, data, top_right_bottom, left_right, ii, n_start_rows, n_train_trajs)
        else:
            data = np.hstack([np.vstack([np.linspace(top_right_top[0], left_left[0], 50), np.repeat(top_right_top[1], 50)]),
                              np.vstack([np.repeat(left_left[0], 50), np.linspace(top_right_top[1], left_left[1], 50)])]).transpose()
            save_data(folder_demo_skill, data, top_right_top, left_left, ii, n_start_rows, n_train_trajs)

    # Skill 8: bottom left corner to middle right
    skill_name = "skill8"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        left = np.array([0.2, 0.2]) + np.random.random(2) * .6
        top_right = np.array([2.2, 2.2]) + np.random.random(2) * .6
        middle_right = np.array([2.2, 1.2]) + np.random.random(2) * .6
        data = np.hstack([np.vstack([np.linspace(left[0], middle_right[0], 70), np.repeat(left[1], 70)]),
                          np.vstack([np.repeat(middle_right[0], 30), np.linspace(left[1], middle_right[1], 30)])]).transpose()
        save_data(folder_demo_skill, data, left, middle_right, ii, n_start_rows, n_train_trajs)

    # Skill 9: Top right corrner to bottom middle corner
    skill_name = "skill9"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        left = np.array([0.2, 0.2]) + np.random.random(2) * .6
        top_right = np.array([2.2, 2.2]) + np.random.random(2) * .6
        bottom_middle = np.array([1.2, 0.2]) + np.random.random(2) * .6
        data = np.hstack([np.vstack([np.linspace(top_right[0], bottom_middle[0], 50), np.repeat(top_right[1], 50)]),
                          np.vstack([np.repeat(bottom_middle[0], 50), np.linspace(top_right[1], bottom_middle[1], 50)])]).transpose()
        save_data(folder_demo_skill, data, top_right, bottom_middle, ii, n_start_rows, n_train_trajs)

    # Skill 10: bottom_middle to (bottom_left or bottom_right)
    skill_name = "skill10"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        left_bottom = np.array([0.2, 0.2]) + np.random.random(2) * .6
        right_bottom = np.array([2.2, 0.2]) + np.random.random(2) * .6
        middle_bottom_top = np.array([1.2, 0.5]) + np.multiply(np.random.random(2), np.array([0.6, 0.3]))
        middle_bottom_bottom = np.array([1.2, 0.2]) + np.multiply(np.random.random(2), np.array([0.6, 0.3]))
        if ii % 2 == 0:
            data = np.vstack([np.linspace(middle_bottom_top[0], left_bottom[0], 100), np.repeat(middle_bottom_top[1], 100)]).transpose()
            save_data(folder_demo_skill, data, middle_bottom_top, left_bottom, ii, n_start_rows, n_train_trajs)
        else:
            data = np.vstack([np.linspace(middle_bottom_bottom[0], right_bottom[0], 100),
                              np.repeat(middle_bottom_bottom[1], 100)]).transpose()
            save_data(folder_demo_skill, data, middle_bottom_bottom, right_bottom, ii, n_start_rows, n_train_trajs)


def save_data(folder_demo_skill, data, start_data, end_data, idx, n_start_rows, n_train_trajs, dim=2):
    start_state = np.zeros([n_start_rows, dim])
    start_state[0, :] = start_data
    start_state[-1, :] = end_data
    if idx < n_train_trajs:
        folder_train_val = folder_demo_skill + "/train"
    else:
        folder_train_val = folder_demo_skill + "/val"
    np.savetxt(folder_train_val + "/rollout-" + str(idx) + ".txt", data)
    np.savetxt(folder_train_val + "/start-state-" + str(idx) + ".txt", start_state)


def generate_trajectories_baxter():
    pass


def generate_trajectories_jackal():
    pass


def generate_trajectories_stretch(folder_demo_trajectories, n_train_trajs, n_val_trajs, n_start_rows, skill_names):
    make_folders(folder_demo_trajectories, skill_names)

    # skill_name = "skillStretch1to2"
    # folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    # for ii in range(n_train_trajs + n_val_trajs):
    #     data = np.zeros([20, 6])
    #
    #     l_start = np.array([0.5])
    #     z_start = 0.7 + 0.2 * np.random.random(1)
    #     t_wrist_start = np.array([np.pi]) + 2 * np.pi * np.random.random(1)
    #
    #     x_start = 0.45 + 0.1 * np.random.random(1)
    #     y_start = 0.45 + 0.1 * np.random.random(1)
    #
    #     x_end = -1.45 - 0.1 * np.random.random(1)
    #     y_end = 0.05 - 0.1 * np.random.random(1)
    #
    #     data[:10, 0] = np.linspace(x_start[0], x_end[0] + 0.5, 10)
    #     data[:10, 1] = y_start
    #     data[10:, 0] = (x_end + 0.5) + 0.5 * np.cos(np.linspace(-np.pi/2, -np.pi, 10))
    #     data[10:, 1] = y_end - 0.5 * np.sin(np.linspace(-np.pi/2, -np.pi, 10))
    #     data[:10, 2] = np.pi
    #     data[10:, 2] = np.linspace(np.pi, 3*np.pi/2, 10)
    #     data[:, 3] = l_start
    #     data[:, 4] = z_start
    #     data[:, 5] = t_wrist_start
    #
    #     start_state = data[0, :]
    #     end_state = data[-1, :]
    #
    #     save_data(folder_demo_skill, data, start_state, end_state, ii, n_start_rows, n_train_trajs, dim=6)
    #
    # # bottom left to top right via bottom right
    # skill_name = "skillStretch2to3"
    # folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    # for ii in range(n_train_trajs + n_val_trajs):
    #     data = np.zeros([20, 6])
    #
    #     l_start = np.array([0.5])
    #     z_start = 0.7 + 0.2 * np.random.random(1)
    #     t_wrist_start = np.array([np.pi]) + 2 * np.pi * np.random.random(1)
    #
    #     x_end = 0.45 + 0.1 * np.random.random(1)
    #     y_end = -0.45 - 0.1 * np.random.random(1)
    #
    #     x_start = -1.45 - 0.1 * np.random.random(1)
    #     y_start = 0.05 - 0.1 * np.random.random(1)
    #
    #     data[:10, 0] = (x_start + 0.5) + 0.5 * np.cos(np.linspace(np.pi, 3 * np.pi/2, 10))
    #     data[:10, 1] = y_start + 0.5 * np.sin(np.linspace(np.pi, 3 * np.pi/2, 10))
    #     data[10:, 0] = np.linspace(data[19, 0], x_end[0], 10)
    #     data[10:, 1] = data[9, 1]
    #
    #     data[:10, 2] = np.linspace(3 * np.pi/2, 2 * np.pi, 10)
    #     data[10:, 2] = 2 * np.pi
    #     data[:, 3] = l_start
    #     data[:, 4] = z_start
    #     data[:, 5] = t_wrist_start
    #
    #     start_state = data[0, :]
    #     end_state = data[-1, :]
    #
    #     save_data(folder_demo_skill, data, start_state, end_state, ii, n_start_rows, n_train_trajs, dim=6)
    #
    # skill_name = "skillStretch3to1"
    # folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    # for ii in range(n_train_trajs + n_val_trajs):
    #     data = np.zeros([20, 6])
    #
    #     l_start = np.array([0.5])
    #     z_start = 0.7 + 0.2 * np.random.random(1)
    #     t_wrist_start = np.array([np.pi]) + 2 * np.pi * np.random.random(1)
    #
    #     x_start = 0.45 + 0.1 * np.random.random(1)
    #     y_start = -0.45 - 0.1 * np.random.random(1)
    #
    #     data[:, 0] = x_start + np.abs(y_start) * np.cos(np.linspace(- np.pi/2, np.pi/2, 20))
    #     data[:, 1] = 0 + np.abs(y_start) * np.sin(np.linspace(-np.pi/2, np.pi/2, 20))
    #
    #     data[:, 2] = np.linspace(0, np.pi, 20)
    #     data[:, 3] = l_start
    #     data[:, 4] = z_start
    #     data[:, 5] = t_wrist_start
    #
    #     start_state = data[0, :]
    #     end_state = data[-1, :]
    #
    #     save_data(folder_demo_skill, data, start_state, end_state, ii, n_start_rows, n_train_trajs, dim=6)

    skill_name = "skillStretch1to2"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        data = np.zeros([100, 6])

        l_start = np.array([0.57])
        z_start = 0.8 + 0.2 * np.random.random(1)
        t_wrist_start = 0 #2 * np.pi * np.random.random(1)

        x_start = 0.45 + 0.1 * np.random.random(1)
        y_start = 0.45 + 0.1 * np.random.random(1)

        x_end = -1.45 - 0.1 * np.random.random(1)
        y_end = 0.05 - 0.1 * np.random.random(1)

        # data[:15, 0] = np.linspace(x_start[0], x_end[0], 15)
        # data[:15, 1] = y_start
        # data[15:, 0] = x_end
        # data[15:, 1] = np.linspace(y_start[0], y_end[0], 5)
        # data[:15, 2] = np.pi
        # data[15:, 2] = 3*np.pi/2
        # data[:, 3] = l_start
        # data[:, 4] = z_start
        # data[:, 5] = t_wrist_start

        data[:50, 0] = np.linspace(x_start[0], x_end[0] + 0.5, 50)
        data[:50, 1] = y_start
        data[50:, 0] = (x_end + 0.5) + 0.5 * np.cos(np.linspace(-np.pi/2, -np.pi, 50))
        data[50:, 1] = y_end - 0.5 * np.sin(np.linspace(-np.pi/2, -np.pi, 50))
        data[:50, 2] = np.pi
        data[50:, 2] = np.linspace(np.pi, 3*np.pi/2, 50)
        data[:, 3] = l_start
        data[:, 4] = z_start
        data[:, 5] = t_wrist_start

        start_state = data[0, :]
        end_state = data[-1, :]

        save_data(folder_demo_skill, data, start_state, end_state, ii, n_start_rows, n_train_trajs, dim=6)

    # bottom left to top right via bottom right
    skill_name = "skillStretch2to3"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        data = np.zeros([20, 6])

        l_start = np.array([0.57])
        z_start = 0.8 + 0.2 * np.random.random(1)
        t_wrist_start = 0#2 * np.pi * np.random.random(1)

        x_end = 0.45 + 0.1 * np.random.random(1)
        y_end = -0.45 - 0.1 * np.random.random(1)

        x_start = -1.45 - 0.1 * np.random.random(1)
        y_start = 0.05 - 0.1 * np.random.random(1)

        # data[:5, 0] = x_start
        # data[:5, 1] = np.linspace(y_start[0], y_end[0], 5)
        # data[5:, 0] = np.linspace(x_start[0], x_end[0], 15)
        # data[5:, 1] = y_end
        # data[:5, 2] = 3 * np.pi / 2
        # data[5:, 2] = 0
        # data[:, 3] = l_start
        # data[:, 4] = z_start
        # data[:, 5] = t_wrist_start

        data[:10, 0] = (x_start + 0.5) + 0.5 * np.cos(np.linspace(np.pi, 3 * np.pi/2, 10))
        data[:10, 1] = y_start + 0.5 * np.sin(np.linspace(np.pi, 3 * np.pi/2, 10))
        data[10:, 0] = np.linspace(data[19, 0], x_end[0], 10)
        data[10:, 1] = data[9, 1]

        data[:10, 2] = np.linspace(3 * np.pi/2, 2 * np.pi, 10)
        data[10:, 2] = 2 * np.pi
        data[:, 3] = l_start
        data[:, 4] = z_start
        data[:, 5] = t_wrist_start

        start_state = data[0, :]
        end_state = data[-1, :]

        save_data(folder_demo_skill, data, start_state, end_state, ii, n_start_rows, n_train_trajs, dim=6)

    skill_name = "skillStretch3to1"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        data = np.zeros([20, 6])

        l_start = np.array([0.57])
        z_start = 0.8 + 0.2 * np.random.random(1)
        t_wrist_start = 0#2 * np.pi * np.random.random(1)

        x_start = 0.45 + 0.1 * np.random.random(1)
        x_mid = x_start + 0.5
        y_start = -0.45 - 0.1 * np.random.random(1)
        y_end = 0.45 + 0.1 * np.random.random(1)

        # data[:7, 0] = np.linspace(x_start[0], x_mid[0], 7)
        # data[:7, 1] = y_start
        # data[7:13, 0] = x_mid
        # data[7:13, 1] = np.linspace(y_start[0], y_end[0], 6)
        # data[13:, 0] = np.linspace(x_mid[0], x_end[0], 7)
        # data[13:, 1] = y_end
        # data[:7, 2] = 0
        # data[7:13, 2] = np.pi / 2
        # data[13:, 2] = np.pi
        # data[:, 3] = l_start
        # data[:, 4] = z_start
        # data[:, 5] = t_wrist_start

        data[:, 0] = x_start + np.abs(y_start) * np.cos(np.linspace(- np.pi/2, np.pi/2, 20))
        data[:, 1] = 0 + np.abs(y_start) * np.sin(np.linspace(-np.pi/2, np.pi/2, 20))

        data[:, 2] = np.linspace(0, np.pi, 20)
        data[:, 3] = l_start
        data[:, 4] = z_start
        data[:, 5] = t_wrist_start

        start_state = data[0, :]
        end_state = data[-1, :]

        save_data(folder_demo_skill, data, start_state, end_state, ii, n_start_rows, n_train_trajs, dim=6)

    skill_name = "skillStretchDownUp1"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        data = np.zeros([20, 6])

        l_start = np.array([0.57])
        z_start = 0.8 + 0.2 * np.random.random(1)
        z_mid = [0.67]
        z_end = z_start
        t_wrist_start = 0  # 2 * np.pi * np.random.random(1)

        x_start = 0.45 + 0.1 * np.random.random(1)
        y_start = 0.45 + 0.1 * np.random.random(1)

        x_end = x_start
        y_end = y_start

        data[:15, 0] = np.linspace(x_start[0], x_end[0], 15)
        data[:15, 1] = y_start
        data[15:, 0] = x_end
        data[15:, 1] = np.linspace(y_start[0], y_end[0], 5)
        data[:, 2] = np.pi
        data[:, 3] = l_start
        data[:10, 4] = np.linspace(z_start[0], z_mid[0], 10)
        data[10:, 4] = np.linspace(z_mid[0], z_end[0], 10)
        data[:, 5] = t_wrist_start

        start_state = data[0, :]
        end_state = data[-1, :]

        save_data(folder_demo_skill, data, start_state, end_state, ii, n_start_rows, n_train_trajs, dim=6)

    # bottom left to top right via bottom right
    skill_name = "skillStretchDownUp2"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        data = np.zeros([20, 6])

        l_start = np.array([0.57])
        z_start = 0.8 + 0.2 * np.random.random(1)
        z_mid = [0.67]
        z_end = z_start
        t_wrist_start = 0#2 * np.pi * np.random.random(1)

        x_start = -1.45 - 0.1 * np.random.random(1)
        y_start = 0.05 - 0.1 * np.random.random(1)

        x_end = x_start
        y_end = y_start

        data[:5, 0] = x_start
        data[:5, 1] = np.linspace(y_start[0], y_end[0], 5)
        data[5:, 0] = np.linspace(x_start[0], x_end[0], 15)
        data[5:, 1] = y_end
        data[:, 2] = 3 * np.pi / 2
        data[:, 3] = l_start
        data[:10, 4] = np.linspace(z_start[0], z_mid[0], 10)
        data[10:, 4] = np.linspace(z_mid[0], z_end[0], 10)
        data[:, 5] = t_wrist_start

        start_state = data[0, :]
        end_state = data[-1, :]

        save_data(folder_demo_skill, data, start_state, end_state, ii, n_start_rows, n_train_trajs, dim=6)

    skill_name = "skillStretchDownUp3"
    folder_demo_skill = folder_demo_trajectories + "/" + skill_name
    for ii in range(n_train_trajs + n_val_trajs):
        data = np.zeros([20, 6])

        l_start = np.array([0.57])
        z_start = 0.8 + 0.2 * np.random.random(1)
        z_mid = [0.67]
        z_end = z_start
        t_wrist_start = 0#2 * np.pi * np.random.random(1)

        x_start = 0.45 + 0.1 * np.random.random(1)
        x_mid = x_start
        x_end = x_start
        y_start = -0.45 - 0.1 * np.random.random(1)
        y_end = y_start

        data[:7, 0] = np.linspace(x_start[0], x_mid[0], 7)
        data[:7, 1] = y_start
        data[7:13, 0] = x_mid
        data[7:13, 1] = np.linspace(y_start[0], y_end[0], 6)
        data[13:, 0] = np.linspace(x_mid[0], x_end[0], 7)
        data[13:, 1] = y_end
        data[:, 2] = 0
        data[:, 3] = l_start
        data[:10, 4] = np.linspace(z_start[0], z_mid[0], 10)
        data[10:, 4] = np.linspace(z_mid[0], z_end[0], 10)
        data[:, 5] = t_wrist_start

        start_state = data[0, :]
        end_state = data[-1, :]

        save_data(folder_demo_skill, data, start_state, end_state, ii, n_start_rows, n_train_trajs, dim=6)


def transformStretchSkillsToEntireSpace(folder_trajectories, skill_names, symbols):
    """Finds the trajectory achieved by the stretch in the state space the symbols care about

    Takes in the folder containing the joint states of the robot, then finds the trajectory of the robot in the state
    space the symbols need. Finds the end effector position. Creates duplicate skills where the stretch moves the duck
    and does not move the duck for each skill/duck combo.

    Args:
        folder_trajectories: path to where the joint space trajectories are (str)
        skill_names:
        symbols:

    Returns:
        skill_names: new skill names
    """

    new_skills = set()
    for skill_name in skill_names:
        if "DownUp" in skill_name:
            new_skills_tmp = transformStretchSkillToEntireSpaceBaseStationary(folder_trajectories, skill_name, symbols)
        else:
            new_skills_tmp = transformStretchSkillToEntireSpaceBaseMoves(folder_trajectories, skill_name, symbols)
        new_skills = new_skills.union(new_skills_tmp)

    return new_skills


def transformStretchSkillToEntireSpaceBaseMoves(folder_trajectories, skill_name, symbols):
    folder_demo_skill = folder_trajectories + "/" + skill_name + "/train/"
    start_states, pose_hists = load_dmp_demos(folder_demo_skill)

    # Loop through each joint space trajectory
    # Find fk for the trajectory
    # Both train and val
    state_space_trajs = np.zeros([pose_hists.shape[0], pose_hists.shape[1], 6])
    state_space_trajs[:, :, 0:3] = pose_hists[:, :, 0:3]
    state_space_trajs[:, :, 3:6] = stretch_fk(pose_hists)

    n_start_rows = 2
    n_train_trajs = 32

    new_skills = set()

    for ii, state_space_traj in enumerate(state_space_trajs):
        # Find the position of the ducks
        # Each duck needs to be moved and the other duck needs to stay in all regions
        # All combinations of ducks need to be on the table
        duck_a_locs = ['1', '2', '3']
        duck_b_locs = ['1', '2', '3']
        ducks_in_ee = [None, 'a', 'b']
        table_ducks = dict()
        for duck_in_ee in ducks_in_ee:
            if duck_in_ee is None:
                for duck_a_loc in duck_a_locs:
                    table_ducks["a"] = duck_a_loc
                    for duck_b_loc in duck_b_locs:
                        if duck_a_loc != duck_b_loc:
                            table_ducks["b"] = duck_b_loc
                            duck_traj = addDuck(duck_in_ee, table_ducks, state_space_traj, symbols)
                            data = np.hstack([state_space_traj, duck_traj])
                            new_skill_name = skill_name + "_a_" + duck_a_loc + "_b_" + duck_b_loc
                            new_skills.add(new_skill_name)
                            folder_demo_skill_new = folder_trajectories + "/" + new_skill_name + "/"
                            make_folders(folder_trajectories, [new_skill_name])
                            save_data(folder_demo_skill_new, data, data[0, :], data[-1, :], ii, n_start_rows, n_train_trajs,
                                      dim=12)
            if duck_in_ee == 'a':
                table_ducks = dict()
                for duck_b_loc in duck_b_locs:
                    table_ducks["b"] = duck_b_loc
                    duck_traj = addDuck(duck_in_ee, table_ducks, state_space_traj, symbols)
                    data = np.hstack([state_space_traj, duck_traj])
                    new_skill_name = skill_name + "_a_" + "hand" + "_b_" + duck_b_loc
                    new_skills.add(new_skill_name)
                    folder_demo_skill_new = folder_trajectories + "/" + new_skill_name + "/"
                    make_folders(folder_trajectories, [new_skill_name])
                    save_data(folder_demo_skill_new, data, data[0, :], data[-1, :], ii, n_start_rows, n_train_trajs,
                              dim=12)
            if duck_in_ee == 'b':
                table_ducks = dict()
                for duck_a_loc in duck_a_locs:
                    table_ducks["a"] = duck_a_loc
                    duck_traj = addDuck(duck_in_ee, table_ducks, state_space_traj, symbols)
                    data = np.hstack([state_space_traj, duck_traj])
                    new_skill_name = skill_name + "_a_" + duck_a_loc + "_b_" + "hand"
                    new_skills.add(new_skill_name)
                    folder_demo_skill_new = folder_trajectories + "/" + new_skill_name + "/"
                    make_folders(folder_trajectories, [new_skill_name])
                    save_data(folder_demo_skill_new, data, data[0, :], data[-1, :], ii, n_start_rows, n_train_trajs,
                              dim=12)

    return new_skills


def transformStretchSkillToEntireSpaceBaseStationary(folder_trajectories, skill_name, symbols):
    folder_demo_skill = folder_trajectories + "/" + skill_name + "/train/"
    start_states, pose_hists = load_dmp_demos(folder_demo_skill)

    # Loop through each joint space trajectory
    # Find fk for the trajectory
    # Both train and val
    state_space_trajs = np.zeros([pose_hists.shape[0], pose_hists.shape[1], 6])
    state_space_trajs[:, :, 0:3] = pose_hists[:, :, 0:3]
    state_space_trajs[:, :, 3:6] = stretch_fk(pose_hists)

    n_start_rows = 2
    n_train_trajs = 32

    new_skills = set()

    for ii, state_space_traj in enumerate(state_space_trajs):
        # Find the position of the ducks
        # Each duck needs to be moved and the other duck needs to stay in all regions
        # All combinations of ducks need to be on the table
        duck_other_locs = ['1', '2', '3']
        ducks_in_ee = ['a', 'b']
        table_ducks = dict()

        for duck_in_ee in ducks_in_ee:
            for duck_other_loc in duck_other_locs:
                table_ducks = {}
                if skill_name[-1] == duck_other_loc:
                    continue
                if duck_in_ee == "a":
                    table_ducks["b"] = duck_other_loc
                if duck_in_ee == "b":
                    table_ducks["a"] = duck_other_loc
                # Pickup
                duck_traj = addDuck(duck_in_ee, table_ducks, state_space_traj, symbols, pickup=True)
                data = np.hstack([state_space_traj, duck_traj])
                if duck_in_ee == "a":
                    new_skill_name = skill_name + "_a_" + "pickup" + "_b_" + duck_other_loc
                else:
                    new_skill_name = skill_name + "_a_" + duck_other_loc + "_b_" + "pickup"
                new_skills.add(new_skill_name)
                folder_demo_skill_new = folder_trajectories + "/" + new_skill_name + "/"
                make_folders(folder_trajectories, [new_skill_name])
                save_data(folder_demo_skill_new, data, data[0, :], data[-1, :], ii, n_start_rows, n_train_trajs,
                          dim=12)
                # place
                duck_traj = addDuck(duck_in_ee, table_ducks, state_space_traj, symbols, place=True)
                data = np.hstack([state_space_traj, duck_traj])
                if duck_in_ee == "a":
                    new_skill_name = skill_name + "_a_" + "place" + "_b_" + duck_other_loc
                else:
                    new_skill_name = skill_name + "_a_" + duck_other_loc + "_b_" + "place"
                new_skills.add(new_skill_name)
                folder_demo_skill_new = folder_trajectories + "/" + new_skill_name + "/"
                make_folders(folder_trajectories, [new_skill_name])
                save_data(folder_demo_skill_new, data, data[0, :], data[-1, :], ii, n_start_rows, n_train_trajs,
                          dim=12)

    return new_skills


def addDuck(duck_in_ee, table_duck_locs, stretch_traj, symbols, pickup=False, place=False):
    """ Adds ducks to the state space

    Given a duck that is being carried, and where the ducks are on the table, returns the position of the ducks through
    the trajectory. Assumes symbols are labeled: duck_LETTER_LOC

    Args:
        duck_in_ee
        table_duck_locs
        stretch_traj
        symbols

    Returns:
         duck_trajs: np.array

    """

    duck_trajs = np.zeros([stretch_traj.shape[0], 6])
    if duck_in_ee is not None:
        duck_move = np.copy(stretch_traj[:, 3:])
        if pickup or place:
            table = symbols['duck_' + duck_in_ee + "_table"].sample_from()
            if pickup:
                duck_move[:10, 2] = table
            if place:
                duck_move[10:, 2] = table
        if duck_in_ee == "a":
            duck_trajs[:, 0:3] = duck_move
        elif duck_in_ee == "b":
            duck_trajs[:, 3:6] = duck_move
    for table_duck, loc in table_duck_locs.items():
        loc_traj = symbols['duck_' + table_duck + "_" + loc].sample_from()
        table = symbols['duck_' + table_duck + "_table"].sample_from()
        if table_duck == "a":
            duck_trajs[:, 0:2] = loc_traj
            duck_trajs[:, 2] = table
        elif table_duck == "b":
            duck_trajs[:, 3:5] = loc_traj
            duck_trajs[:, 5] = table

    return duck_trajs


def stretch_fk(stretch_pose):
    """ Finds the forward kinematics of the stretch

    Given the joint angles of the stretch (x_r, y_r, theta_r, extension, lift, theta_wrist), finds the forward
    kinematics of the end effector

    Args:
        stretch_pose

    Returns:
        forward_kinematics

    """
    ee = np.zeros([stretch_pose.shape[0], stretch_pose.shape[1], 3])
    l_ee = 0.23
    # dim == 0 -> x
    t_robot = stretch_pose[:, :, 2]
    t_wrist = stretch_pose[:, :, 5]
    x_robot = stretch_pose[:, :, 0]
    y_robot = stretch_pose[:, :, 1]
    l_arm = stretch_pose[:, :, 3]
    lift = stretch_pose[:, :, 4]
    x_ee = l_ee * np.cos(t_robot + t_wrist - np.pi/2) + l_arm * np.cos(t_robot - np.pi/2) + x_robot
    y_ee = l_ee * np.sin(t_robot + t_wrist - np.pi/2) + l_arm * np.sin(t_robot - np.pi/2) + y_robot
    z_ee = lift
    ee[:, :, 0] = x_ee
    ee[:, :, 1] = y_ee
    ee[:, :, 2] = z_ee

    return ee


def make_folders(folder_demo_trajectories, skill_names):
    for skill_name in skill_names:
        os.makedirs(folder_demo_trajectories + "/" + skill_name + "/train", exist_ok=True)
        os.makedirs(folder_demo_trajectories + "/" + skill_name + "/val", exist_ok=True)


if __name__ == "__main__":

    #########################################
    # Parse arguments and unpack parameters #
    #########################################

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_names", help="File names", required=True)
    parser.add_argument("--sym_opts", help="Opts involving spec writing and repair", required=True)
    parser.add_argument("--dmp_opts", help="Opts involving plotting, repair, dmps", required=True)
    parser.add_argument("--do_plot", help="Plot the skills?", action='store_true', default=False)
    args = parser.parse_args()

    file_names = json_load_wrapper(args.file_names)
    sym_opts = json_load_wrapper(args.sym_opts)
    dmp_opts = json_load_wrapper(args.dmp_opts)
    folder_trajectories = file_names["folder_trajectories"]

    # skill_names = file_names["skill_names"]
    skill_names = ["skillStretch1to2", "skillStretch2to3", "skillStretch3to1", "skillStretchDownUp1", "skillStretchDownUp2", "skillStretchDownUp3"]
    symbols = load_symbols(file_names["file_symbols"])
    workspace_bnds = np.array(dmp_opts["workspace_bnds"])

    ##########################################################################
    # Generate trajectories and save the raw trajectories in the demo_folder #
    ##########################################################################

    if file_names['file_symbols'].split('/')[2] == 'nine_squares':
        generate_trajectories_nine_squares(folder_trajectories, dmp_opts["n_train_trajs"], dmp_opts["n_val_trajs"], dmp_opts["n_states"], skill_names)
    elif file_names['file_symbols'].split('/')[2] == 'stretch':
        generate_trajectories_stretch(folder_trajectories, dmp_opts["n_train_trajs"], dmp_opts["n_val_trajs"], dmp_opts["n_states"], skill_names)

    ##############################################################################
    # Create DMP from the generated trajectories and save them in the dmp folder #
    ##############################################################################

    dmp_opts['enforce_type'] = 'unconstrained'
    dmp_opts['symbols'] = symbols
    dmp_opts['plot_limits'] = np.array(dmp_opts['plot_limits'])
    dmp_opts['use_previous'] = False
    for skill in skill_names:
        dmp_opts['skill_name'] = skill
        dmp_opts['demo_folder'] = folder_trajectories + '/' + skill + '/'
        _, _, _ = run_elaborateDMP(None, skill, None, None, symbols, workspace_bnds, dmp_opts)

    ################################################################
    # Generate the symbolic/abstract representation of the skills  #
    # for use when writing the specification and save it in a json #
    # file                                                         #
    ################################################################
    if file_names['file_symbols'].split('/')[2] == 'stretch':
        new_skill_names = transformStretchSkillsToEntireSpace(folder_trajectories, skill_names, symbols)
        print(new_skill_names)
    skills = load_skills_from_trajectories(folder_trajectories, new_skill_names, symbols)
    write_skills_json(skills, file_names['file_skills'])

    ##############################
    # Plot skills if desired  ####
    ##############################

    if args.do_plot:
        os.makedirs(file_names["folder_plot"], exist_ok=True)
        dim = len(dmp_opts["plot_limits"])
        for skill_name, skill in skills.items():
            fig = plt.figure()
            if dim == 3:
                ax = plt.axes(projection="3d")
            elif dim == 2:
                ax = plt.axes()
            skill.plot_original(ax)
            ax.set_xlim(dmp_opts["plot_limits"][0])
            ax.set_ylim(dmp_opts["plot_limits"][1])
            if dim == 3:
                ax.set_zlim(dmp_opts["plot_limits"][2])
            plt.savefig(file_names["folder_plot"] + skill_name + ".png")
            plt.close()
