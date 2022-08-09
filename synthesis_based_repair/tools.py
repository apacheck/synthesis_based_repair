#!/usr/bin/env
from synthesis_based_repair.symbols import find_symbols_by_var, symbols_intersect
import json
import numpy as np


def json_load_wrapper(arg_file):
    fid = open(arg_file, "r")
    d = json.load(fid)
    for key, val in d.items():
        if val == 'none':
            d[key] = None
    return d


def dict_to_formula(sym_dict, prime=False, include_false=True):
    str_list = []
    for sym, val in sym_dict.items():
        if val:
            str_list.append(sym)
        elif include_false:
            str_list.append("!" + sym)
    if prime:
        out = "' & ".join(str_list) + "'"
    else:
        out = " & ".join(str_list)

    return out


def pre_posts_to_env_formula(skill_name, pre_syms_dict, post_syms_dict_list):
    pre = dict_to_formula(pre_syms_dict)
    post_list = []
    for one_post in post_syms_dict_list:
        post_list.append("(" + dict_to_formula(one_post, prime=True) + ")")
    post = "(" + " | ".join(post_list) + ")"

    return "{} & {} -> {}".format(skill_name, pre, post)


def sym_list_to_dict(true_syms, all_syms):
    dict_out = dict()
    for symbol in all_syms:
        if symbol in true_syms:
            dict_out[symbol] = True
        else:
            dict_out[symbol] = False

    return dict_out


def write_symbols(file_spec, heading, symbols, symbols_reactive):
    if heading != 'INPUT' and heading != 'OUTPUT':
        raise Exception('heading should be INPUT or OUTPUT')
    fid = open(file_spec, 'a')
    fid.write('[{}]\n\n'.format(heading))
    for sym in symbols.keys():
        fid.write('{}\n'.format(sym))
    for sym in symbols_reactive:
        fid.write("{}\n".format(sym))
    fid.write('\n')
    fid.close()


def write_init(file_spec, heading, symbols_true, symbols_false):

    if heading != 'ENV_INIT' and heading != 'SYS_INIT':
        raise Exception('heading should be ENV_INIT or SYS_INIT')
    fid = open(file_spec, 'a')
    fid.write('[{}]\n\n'.format(heading))

    # Writes the true and false symbols. All other symbols can be anything
    if len(symbols_false) > 0:
        fid.write("!" + "\n!".join(symbols_false) + "\n")
    if len(symbols_true) > 0:
        fid.write("\n".join(symbols_true) + "\n")
    fid.write("\n")
    fid.close()


def write_env_trans(file_spec, symbols, skills, opts):
    fid = open(file_spec, 'a')
    fid.write('[%s]\n\n' % "ENV_TRANS")

    # Write as skill + pre -> post and skill + post -> post
    # Ignore overlap and hope it is taken care of by mutual exclusion at the end
    for skill_name, skill in skills.items():
        for pre_syms_dict, post_syms_dict_list in skill.get_intermediate_states():
            tmp_formula = pre_posts_to_env_formula(skill_name, pre_syms_dict, post_syms_dict_list)
            fid.write("{}\n".format(tmp_formula))

    fid.write("\n[ENV_TRANS_HARD]\n")
    # Mutual exclusion. Writes the spec that says only one symbol for each
    # state variable can be true at a time
    syms_by_var = find_symbols_by_var(symbols, opts['n_factors'])
    for syms in syms_by_var:
        for ii, sym1 in enumerate(syms):
            s1_sym = symbols[sym1]
            for jj, sym2 in enumerate(syms):
                s2_sym = symbols[sym2]
                inter = symbols_intersect(s1_sym, s2_sym)
                if not inter and s1_sym.get_index() < s2_sym.get_index():
                    fid.write('!({}\' & {}\')\n'.format(sym1, sym2))
    for syms in syms_by_var:
        sym_str = '(' + '\' | '.join(syms) + '\')\n'
        fid.write(sym_str)

    for syms in syms_by_var:
        for ii, sym1 in enumerate(syms):
            s1_sym = symbols[sym1]
            for jj, sym2 in enumerate(syms):
                s2_sym = symbols[sym2]
                inter = symbols_intersect(s1_sym, s2_sym)
                if not inter and s1_sym.get_index() < s2_sym.get_index():
                    fid.write('!({} & {})\n'.format(sym1, sym2))
    for syms in syms_by_var:
        sym_str = '(' + ' | '.join(syms) + ')\n'
        fid.write(sym_str)

    # If no actions are taken, no symbols change
    # First writes the formula saying no actions are taken
    # then appends that to each symbol
    not_action = "!" + " & !".join(skills.keys())
    sym_stay_list = []
    for sym in symbols.keys():
        sym_stay_list.append("({} <-> {}')".format(sym, sym))
    sym_stay = " & ".join(sym_stay_list)
    fid.write('({}) -> ({})\n'.format(not_action, sym_stay))

    # Mutual exclusion. Writes the spec that says only one action is true at
    # a time
    u_acts = list(skills.keys())
    for ii in range(len(u_acts)):
        s_write = ''
        for jj in range(len(u_acts)):
            if ii == jj:
                s_write = s_write + " & " + u_acts[jj]
            else:
                s_write = s_write + " & !" + u_acts[jj]
        s_out = '(' + s_write[3:] + ')'
        fid.write("%s" % s_out)
        if ii < len(u_acts) - 1:
            fid.write(" | ")
        else:
            fid.write(" | ")
    # If no actions are taken
    fid.write('(%s)\n' % not_action)

    fid.close()


def write_sys_trans(file_spec, symbols, skills, user_spec, opts):
    sys_false_trans = True
    fid = open(file_spec, 'a')
    fid.write('[%s]\n\n' % "SYS_TRANS")

    # What states need to be true for a skill to be performed
    allowable_skill_list = []
    skill_continue_list = []
    for skill_name, skill in skills.items():
        init_pres = skill.get_initial_pres()
        final_post = skill.get_final_posts()
        allowable_skill = '!('
        all_pres = [pre for pre, _ in skill.get_intermediate_states()]
        for pre_dict, post_dict_list in skill.get_intermediate_states():
            for post_dict in post_dict_list:
                if post_dict not in all_pres or pre_dict == post_dict:
                    continue
                allowable_skill += "({} & {} & {}) | ".format(dict_to_formula(pre_dict, include_false=sys_false_trans), skill_name, dict_to_formula(post_dict, prime=True, include_false=sys_false_trans))
                skill_continue = "({} & {} & {}) -> {}'".format(dict_to_formula(pre_dict, include_false=sys_false_trans), skill_name, dict_to_formula(post_dict, prime=True, include_false=sys_false_trans), skill_name)
                skill_continue_list.append(skill_continue)

        p_sym_list = []
        for pre_dict in init_pres:
            p_sym_list.append("(" + dict_to_formula(pre_dict, prime=True, include_false=sys_false_trans) + ")")
        allowable_skill += "{}) -> !{}\'".format(" | ".join(p_sym_list), skill_name)
        allowable_skill_list.append(allowable_skill)

    fid.write("{}\n".format("\n".join(allowable_skill_list)))
    fid.write("{}\n".format("\n".join(skill_continue_list)))

    fid.write('\n[SYS_TRANS_HARD]\n')

    u_acts = list(skills.keys())
    for ii in range(len(u_acts)):
        s_write = ''
        for jj in range(len(u_acts)):
            if ii == jj:
                s_write = s_write + " & " + u_acts[jj] + "\'"
            else:
                s_write = s_write + " & !" + u_acts[jj] + "\'"
        s_out = '(' + s_write[3:] + ')'
        fid.write("%s" % s_out)
        if ii < len(u_acts) - 1:
            fid.write(" | ")
        else:
            fid.write(" | ")

    # If no actions are taken
    not_action = ''
    for ii, action in enumerate(u_acts):
        if ii > 0:
            not_action = not_action + ' & '
        not_action = not_action + '!' + action + '\''
    fid.write('(%s)\n' % not_action)

    fid.write("\n# User SYS_TRANS_HARD\n")
    fid.write("{}\n\n".format(user_spec['sys_trans_hard']))

    fid.close()


def write_section(file_spec, section_name, section_spec):
    fid = open(file_spec, 'a')
    fid.write('[{}]\n\n'.format(section_name))
    fid.write(section_spec)
    fid.write("\n")
    fid.close()


def clear_file(f):
    """
        Clears the file
    """
    fid = open(f, 'w')
    fid.write('')
    fid.close()


def write_spec(file_spec, symbols, skills, user_spec, change_cons, not_allowed_repair, opts):

    clear_file(file_spec)

    # INPUT
    write_symbols(file_spec, 'INPUT', symbols, user_spec['reactive_variables'])

    # OUTPUT
    write_symbols(file_spec, 'OUTPUT', skills, [])

    # ENV_INIT
    write_init(file_spec, 'ENV_INIT', user_spec['env_init_true'], user_spec['env_init_false'])

    # SYS_INIT
    write_init(file_spec, 'SYS_INIT', user_spec['sys_init_true'], user_spec['sys_init_false'])

    # ENV_TRANS
    write_env_trans(file_spec, symbols, skills, opts)

    # SYS_TRANS
    write_sys_trans(file_spec, symbols, skills, user_spec, opts)

    # SYS_LIVENESS
    write_section(file_spec, "SYS_LIVENESS", user_spec['sys_live'])

    write_section(file_spec, "CHANGE_CONS", change_cons)

    write_section(file_spec, "NOT_ALLOWED_REPAIR", not_allowed_repair)


def fk_stretch(rollouts):
    """
    Calculates the forward kinematics for a set of rollouts. Each rollout has [n different trajectories, m points in
    a trajectory, and 6 dimensions (for x_robot, y_robot, theta_robot, arm_extension, lift, theta_wrist]
    :param rollouts:
    :return:
    """
    wrist_x_in_base_frame = 0.14
    wrist_y_in_base_frame = -0.16
    gripper_length_xy = 0.23
    gripper_offset_z = -0.1
    base_height_z = 0.05

    xyz_ee_in_global_frame = np.zeros([rollouts.shape[0], rollouts.shape[1], 3]
                                      )
    for rr, rollout in enumerate(rollouts):
        for pp, point in enumerate(rollout):
            x_robot = point[0]
            y_robot = point[1]
            t_robot = point[2]
            arm_extension = point[3]
            lift = point[4]
            t_wrist = point[5]

            # Transform to robot frame by taking into account offsets and extension
            xy_ee_in_robot_frame = np.zeros([2])
            xy_ee_in_robot_frame[0] = wrist_x_in_base_frame + gripper_length_xy * np.sin(t_wrist)
            xy_ee_in_robot_frame[1] = wrist_y_in_base_frame - arm_extension - gripper_length_xy * np.cos(t_wrist)

            xy_ee_in_global_frame = robot_to_global(point[:3], xy_ee_in_robot_frame)
            xyz_ee_in_global_frame[rr, pp, :2] = xy_ee_in_global_frame
            xyz_ee_in_global_frame[rr, pp, 2] = lift + gripper_offset_z + base_height_z

    return xyz_ee_in_global_frame


def robot_to_global(robot_xytheta, point_in_robot_frame):
    """
    Transforms a point in the robot's frame to be in the global frame
    :param robot_xytheta:
    :param point_in_robot_frame:
    :return:
    """
    R = np.zeros([3, 3])
    R[0, 0] = np.cos(robot_xytheta[2])
    R[0, 1] = -np.sin(robot_xytheta[2])
    R[0, 2] = robot_xytheta[0]
    R[1, 0] = np.sin(robot_xytheta[2])
    R[1, 1] = np.cos(robot_xytheta[2])
    R[1, 2] = robot_xytheta[1]
    R[2, 2] = 1

    point_in_robot_frame_with_one = np.ones([3, 1])
    point_in_robot_frame_with_one[:2, 0] = point_in_robot_frame
    xy_out = np.matmul(R, point_in_robot_frame_with_one)

    return xy_out[:2].squeeze()
