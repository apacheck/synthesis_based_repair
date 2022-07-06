#!/usr/bin/env python
import os
import numpy as np
from synthesis_based_repair.physical_implementation import run_elaborateDMP
from synthesis_based_repair.symbols import load_symbols
import argparse
from synthesis_based_repair.tools import json_load_wrapper
from synthesis_based_repair.skills import load_skills_from_trajectories, write_skills_json
import matplotlib.pyplot as plt


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
        data = np.zeros([20, 6])

        l_start = np.array([0.5])
        z_start = 0.7 + 0.2 * np.random.random(1)
        t_wrist_start = np.array([np.pi]) + 2 * np.pi * np.random.random(1)

        x_start = 0.45 + 0.1 * np.random.random(1)
        y_start = 0.45 + 0.1 * np.random.random(1)

        x_end = -1.45 - 0.1 * np.random.random(1)
        y_end = 0.05 - 0.1 * np.random.random(1)

        data[:15, 0] = np.linspace(x_start[0], x_end[0], 15)
        data[:15, 1] = y_start
        data[15:, 0] = x_end
        data[15:, 1] = np.linspace(y_start[0], y_end[0], 5)
        data[:15, 2] = np.pi
        data[15:, 2] = 3*np.pi/2
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

        l_start = np.array([0.5])
        z_start = 0.7 + 0.2 * np.random.random(1)
        t_wrist_start = np.array([np.pi]) + 2 * np.pi * np.random.random(1)

        x_end = 0.45 + 0.1 * np.random.random(1)
        y_end = -0.45 - 0.1 * np.random.random(1)

        x_start = -1.45 - 0.1 * np.random.random(1)
        y_start = 0.05 - 0.1 * np.random.random(1)

        data[:5, 0] = x_start
        data[:5, 1] = np.linspace(y_start[0], y_end[0], 5)
        data[5:, 0] = np.linspace(x_start[0], x_end[0], 15)
        data[5:, 1] = y_end
        data[:5, 2] = 3 * np.pi / 2
        data[5:, 2] = 0
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

        l_start = np.array([0.5])
        z_start = 0.7 + 0.2 * np.random.random(1)
        t_wrist_start = np.array([np.pi]) + 2 * np.pi * np.random.random(1)

        x_start = 0.45 + 0.1 * np.random.random(1)
        x_mid = x_start + 0.5
        y_start = -0.45 - 0.1 * np.random.random(1)
        y_end = 0.45 + 0.1 * np.random.random(1)

        data[:7, 0] = np.linspace(x_start[0], x_mid[0], 7)
        data[:7, 1] = y_start
        data[7:13, 0] = x_mid
        data[7:13, 1] = np.linspace(y_start[0], y_end[0], 6)
        data[13:, 0] = np.linspace(x_mid[0], x_end[0], 7)
        data[13:, 1] = y_end
        data[:7, 2] = 0
        data[7:13, 2] = np.pi / 2
        data[13:, 2] = 3 * np.pi / 2
        data[:, 3] = l_start
        data[:, 4] = z_start
        data[:, 5] = t_wrist_start

        start_state = data[0, :]
        end_state = data[-1, :]

        save_data(folder_demo_skill, data, start_state, end_state, ii, n_start_rows, n_train_trajs, dim=6)


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

    skill_names = file_names["skill_names"]
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

    skills = load_skills_from_trajectories(folder_trajectories, skill_names, symbols)
    write_skills_json(skills, file_names['file_skills'])

    ##############################
    # Plot if skills if desired  #
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
