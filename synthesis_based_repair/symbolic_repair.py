#!/usr/bin/env python

import argparse
from typing import List

# import dd.autoref as _bdd
import dd.cudd as _bdd
import copy
import math
import sys
import time
import os
import numpy as np

DEBUG_THRESH = 121

PRINT_EXPR = True
DEBUG_PRE = False
DEBUG = False
DEBUG_WS_COMPUTE = False
DEBUG_DETERMINIZE = False
DEBUG_REVISION = True

# PRINT_EXPR = False
# DEBUG_PRE = False
# DEBUG = False
# DEBUG_WS_COMPUTE = False
# DEBUG_DETERMINIZE = False
# DEBUG_REVISION = False

class Specification:
    def __init__(self, file_in_internal, bdd_internal):
        """

        :param file_in_internal:
        :param bdd_internal:
        """
        self.file_in = file_in_internal
        self.bdd = bdd_internal

        self.sections = {"input": "[INPUT]", "output": "[OUTPUT]", "envinit": "[ENV_INIT]", "sysinit": "[SYS_INIT]",
                         "envtrans": "[ENV_TRANS]", "envtranshard": "[ENV_TRANS_HARD]", "systrans": "[SYS_TRANS]",
                         "systranshard": "[SYS_TRANS_HARD]", "syslive": "[SYS_LIVENESS]", "envlive": "[ENV_LIVENESS]",
                         "changecons": "[CHANGE_CONS]", "notallowedrepair": "[NOT_ALLOWED_REPAIR]"}
        self.sections_inv = {v: k for k, v in self.sections.items()}

        self.input_vars = []
        self.input_vars_prime = []
        self.input_vars_double_prime = []
        self.output_vars = []
        self.output_vars_prime = []
        self.output_vars_double_prime = []
        self.env_init = []
        self.sys_init = []
        self.env_trans = []
        self.env_trans_hard = []
        self.env_trans_not_hard = []
        self.sys_trans = []
        self.sys_trans_hard = []
        self.sys_trans_not_hard = []
        self.env_live_assumptions = []
        self.sys_live_guarantees = []
        self.change_cons = []
        self.not_allowed_repair = []

        self.vars = []
        self.vars_prime = []
        self.input_vars = []
        self.input_vars_prime = []
        self.output_vars = []
        self.output_vars_prime = []
        self.output_vars_double_prime = []

        self.vars_and_vars_prime = []
        self.vars_and_input_prime = []

        self.T_env_init = None
        self.T_sys_init = None
        self.T_init = None
        self.T_env = None
        self.T_env_hard = None
        self.T_sys = None
        self.T_sys_hard = None
        self.T_env_live = None
        self.T_sys_live = None
        self.T_change_cons = None
        self.T_not_allowed_repair = None

        self.T_sys_not_hard = None
        self.T_env_not_hard = None

        self.gs = None

        self.read_in_spec()

    def read_in_spec(self):
        current_section = ""
        fid = open(self.file_in, "r")
        lines = fid.readlines()
        fid.close()

        # Read in and store formulas
        for line in lines:
            if line == "\n" or line[0] == "#":
                continue
            if list(self.sections.values()).count(line[:-1]):
                current_section = self.sections_inv[line[:-1]]
                continue
            line = line[:-1]
            if current_section == "input":
                self.input_vars.append(line)
            elif current_section == "output":
                self.output_vars.append(line)
            elif current_section == "envinit":
                self.env_init.append(line)
            elif current_section == "sysinit":
                self.sys_init.append(line)
            elif current_section == "envtrans":
                self.env_trans.append(line)
                self.env_trans_not_hard.append(line)
            elif current_section == "envtranshard":
                self.env_trans.append(line)
                self.env_trans_hard.append(line)
            elif current_section == "systrans":
                self.sys_trans.append(line)
                self.sys_trans_not_hard.append(line)
            elif current_section == "systranshard":
                self.sys_trans.append(line)
                self.sys_trans_hard.append(line)
            elif current_section == "envlive":
                self.env_live_assumptions.append(line)
            elif current_section == "syslive":
                self.sys_live_guarantees.append(line)
            elif current_section == "changecons":
                self.change_cons.append(line)
            elif current_section == "notallowedrepair":
                self.not_allowed_repair.append(line)
            else:
                print("current_section '{}' not recognized".format(current_section))

        # Create variables and add them to a list
        for input_var in self.input_vars:
            self.bdd.add_var(input_var)
            self.bdd.add_var(input_var + "'")
            self.input_vars_prime.append(input_var + "'")
            self.bdd.add_var(input_var + "''")
            self.input_vars_double_prime.append(input_var + "''")

        for output_var in self.output_vars:
            self.bdd.add_var(output_var)
            self.bdd.add_var(output_var + "'")
            self.output_vars_prime.append(output_var + "'")
            # output_vars_double_prime.append(output_var + "''")

        self.vars = copy.deepcopy(self.input_vars)
        self.vars.extend(self.output_vars)
        self.vars_and_input_prime = copy.copy(self.vars)
        self.vars_and_input_prime.extend(self.input_vars_prime)

        self.vars_prime = copy.deepcopy(self.input_vars_prime)
        self.vars_prime.extend(self.output_vars_prime)

        self.vars_and_vars_prime = copy.deepcopy(self.vars)
        self.vars_and_vars_prime.extend(self.vars_prime)

        # Create environment and system init
        self.T_env_init = create_expr(self.bdd, self.env_init)
        self.T_sys_init = create_expr(self.bdd, self.sys_init)
        self.T_init = self.T_env_init & self.T_sys_init

        # Create environment and system transitions
        self.T_env = create_expr(self.bdd, self.env_trans)
        self.T_env_hard = create_expr(self.bdd, self.env_trans_hard)
        self.T_sys = create_expr(self.bdd, self.sys_trans)
        self.T_sys_hard = create_expr(self.bdd, self.sys_trans_hard)

        self.T_sys_not_hard = create_expr(self.bdd, self.sys_trans_not_hard)
        self.T_env_not_hard = create_expr(self.bdd, self.env_trans_not_hard)

        # Create environment and system liveness guarantees
        self.T_sys_live = create_expr(self.bdd, self.sys_live_guarantees)
        self.T_env_live = create_expr(self.bdd, self.env_live_assumptions)

        # Create change constraints
        self.T_change_cons = create_expr(self.bdd, self.change_cons)
        self.T_not_allowed_repair = create_expr(self.bdd, self.not_allowed_repair)

    def get_game_structure(self):
        if self.gs is None:
            self.gs = GameStructure(self.bdd, self.vars, self.vars_prime,
                                    self.input_vars, self.input_vars_prime, self.input_vars_double_prime,
                                    self.output_vars, self.output_vars_prime, self.output_vars_double_prime,
                                    self.T_env, self.T_env_hard, self.T_env_not_hard,
                                    self.T_sys, self.T_sys_hard, self.T_sys_not_hard,
                                    self.env_live_assumptions, self.sys_live_guarantees,
                                    self.T_env_init, self.T_sys_init,
                                    arg_change_cons=self.T_change_cons,
                                    arg_not_allowed_repair=self.T_not_allowed_repair)
        return self.gs

    def get_init(self):
        return self.T_init


class GameStructure:
    def __init__(self, bdd_internal, vars_internal, vars_prime_internal,
                 input_vars_internal, input_vars_prime_internal, input_vars_double_prime_internal,
                 output_vars_internal, output_vars_prime_internal, output_vars_double_prime_internal,
                 T_env_internal, T_env_hard_internal, arg_T_env_not_hard,
                 T_sys_internal, T_sys_hard_internal, arg_T_sys_not_hard,
                 env_live_assumptions_internal, sys_live_guarantees_internal,
                 arg_T_env_init, arg_T_sys_init,
                 arg_cntr_vars=None, arg_cntr_vars_prime=None,
                 arg_change_cons=None, arg_not_allowed_repair=None):
        """

        :type bdd_internal: dd.autoref.BDD
        """
        self.bdd = bdd_internal
        self.vars = vars_internal
        self.vars_prime = vars_prime_internal
        self.input_vars = input_vars_internal
        self.input_vars_prime = input_vars_prime_internal
        self.input_vars_double_prime = input_vars_double_prime_internal
        self.output_vars = output_vars_internal
        self.output_vars_prime = output_vars_prime_internal
        self.output_vars_double_prime = output_vars_double_prime_internal
        self.T_env = T_env_internal
        self.T_env_hard = T_env_hard_internal
        self.T_sys = T_sys_internal
        self.T_sys_hard = T_sys_hard_internal
        self.env_live_assumptions = env_live_assumptions_internal
        if len(self.env_live_assumptions) == 0:
            self.env_live_assumptions = ["True"]
        self.sys_live_guarantees = sys_live_guarantees_internal
        self.T_env_init = arg_T_env_init
        self.T_sys_init = arg_T_sys_init
        self.T_sys_not_hard = arg_T_sys_not_hard
        self.T_env_not_hard = arg_T_env_not_hard
        self.cntr_vars = arg_cntr_vars
        self.cntr_vars_prime = arg_cntr_vars_prime
        self.change_cons = arg_change_cons
        self.not_allowed_repair = arg_not_allowed_repair

        self.v_to_v_prime = self.create_mapping(self.vars, self.vars_prime)
        self.v_prime_to_v = self.create_mapping(self.vars_prime, self.vars)
        self.outputprime_to_outputdoubleprime = self.create_mapping(self.output_vars_prime,
                                                                    self.output_vars_double_prime)
        self.outputdoubleprime_to_outputprime = self.create_mapping(self.output_vars_double_prime,
                                                                    self.output_vars_prime)
        self.output_to_outputprime = self.create_mapping(self.output_vars, self.output_vars_prime)
        self.inputprime_to_inputdoubleprime = self.create_mapping(self.input_vars_prime,
                                                                  self.input_vars_double_prime)
        self.inputdoubleprime_to_inputprime = self.create_mapping(self.input_vars_double_prime,
                                                                  self.input_vars_prime)
        self.inputdoubleprime_to_input = self.create_mapping(self.input_vars_double_prime,
                                                             self.input_vars)
        self.input_to_inputprime = self.create_mapping(self.input_vars, self.input_vars_prime)
        self.input_to_inputdoubleprime = self.create_mapping(self.input_vars, self.input_vars_double_prime)

        self.inputprime_to_input = self.create_mapping(self.input_vars_prime, self.input_vars)
        self.outputprime_to_output = self.create_mapping(self.output_vars_prime, self.output_vars)

        self.vars_and_vars_prime = copy.copy(self.vars)
        self.vars_and_vars_prime.extend(self.vars_prime)
        self.vars_and_input_prime = copy.copy(self.vars)
        self.vars_and_input_prime.extend(self.input_vars_prime)
        self.vars_and_output_prime = copy.copy(self.vars)
        self.vars_and_output_prime.extend(self.output_vars_prime)

    def create_mapping(self, orig, new):
        d = dict()
        for o, n in zip(orig, new):
            d[o] = n
        return d

    def get_v_to_v_prime(self):
        return self.v_to_v_prime

    def get_v_prime_to_v(self):
        return self.v_prime_to_v

    def get_outputprime_to_outputdoubleprime(self):
        return self.outputprime_to_outputdoubleprime

    def get_outputdoubleprime_to_outputprime(self):
        return self.outputdoubleprime_to_outputprime

    def get_output_to_outputprime(self):
        return self.output_to_outputprime

    def get_inputprime_to_inputdoubleprime(self):
        return self.inputprime_to_inputdoubleprime

    def get_inputdoubleprime_to_inputprime(self):
        return self.inputdoubleprime_to_inputprime

    def get_inputdoubleprime_to_input(self):
        return self.inputdoubleprime_to_input

    def get_input_to_inputdoubleprime(self):
        return self.input_to_inputdoubleprime

    def get_input_to_inputprime(self):
        return self.input_to_inputprime

    def get_inputprime_to_input(self):
        return self.inputprime_to_input

    def get_outputprime_to_output(self):
        return self.outputprime_to_output

    def get_change_cons(self):
        return self.change_cons

    def get_change_cons_p_and_dp(self):
        tmp = self.bdd.let(self.inputprime_to_inputdoubleprime, self.change_cons)
        tmp = self.bdd.let(self.input_to_inputprime, tmp)
        return tmp

    def get_not_allowed_repair(self):
        return self.not_allowed_repair

    def get_not_allowed_repair_v_and_dp(self):
        return self.bdd.let(self.inputprime_to_inputdoubleprime, self.not_allowed_repair)

    def get_not_allowed_repair_p_and_dp(self):
        tmp = self.bdd.let(self.inputprime_to_inputdoubleprime, self.not_allowed_repair)
        return self.bdd.let(self.input_to_inputprime, tmp)

    def get_env_live_assumptions(self):
        return self.env_live_assumptions

    def get_sys_live_guarantees(self):
        return self.sys_live_guarantees

    # def get_sys_trans(self):
    #     return self.T_sys
    #
    # def get_env_trans(self):
    #     return self.T_env

    def get_t_sys_not_hard(self):
        return self.T_sys_not_hard

    def get_t_env_not_hard(self):
        return self.T_env_not_hard

    def get_output_vars(self):
        return self.output_vars

    def get_output_vars_prime(self):
        return self.output_vars_prime

    def get_output_vars_double_prime(self):
        return self.output_vars_double_prime

    def get_output_vars_and_prime(self):
        x = copy.copy(self.output_vars)
        x.extend(self.output_vars_prime)
        return x

    def get_input_vars(self):
        return self.input_vars

    def get_input_vars_prime(self):
        return self.input_vars_prime

    def get_input_vars_double_prime(self):
        return self.input_vars_double_prime

    def get_input_vars_and_prime(self):
        x = copy.copy(self.input_vars)
        x.extend(self.input_vars_prime)
        return x

    def get_vars(self):
        return self.vars

    def get_vars_prime(self):
        return self.vars_prime

    def get_vars_and_vars_prime(self):
        return self.vars_and_vars_prime

    def get_vars_and_input_prime(self):
        return self.vars_and_input_prime

    def get_vars_and_output_prime(self):
        return self.vars_and_output_prime

    def get_vars_and_prime_and_dp(self):
        x = copy.copy(self.vars_and_vars_prime)
        x.extend(self.input_vars_double_prime)
        x.extend(self.output_vars_double_prime)
        return x

    def get_t_sys(self):
        return self.T_sys

    def get_t_sys_hard(self):
        return self.T_sys_hard

    def get_t_env(self):
        return self.T_env

    def get_t_env_hard(self):
        return self.T_env_hard

    def get_t_init(self):
        return self.T_env_init & self.T_sys_init

    def get_t_env_init(self):
        return self.T_env_init

    def get_t_sys_init(self):
        return self.T_sys_init

    def get_cntr_vars(self):
        return self.cntr_vars

    def get_cntr_vars_prime(self):
        return self.cntr_vars_prime

    def get_cntr_vars_and_prime(self):
        x = copy.copy(self.cntr_vars)
        x.extend(self.cntr_vars_prime)
        return x

    def cox(self, x_set):
        x_set_prime = self.bdd.let(self.get_v_to_v_prime(), x_set)
        tmp_sys_and_x_set_prime = x_set_prime & self.get_t_sys()
        tmp_exists = self.bdd.exist(self.get_output_vars_prime(), tmp_sys_and_x_set_prime)
        tmp_implies = (~ self.get_t_env()) | tmp_exists
        tmp_all = self.bdd.forall(self.get_input_vars_prime(), tmp_implies)
        # Addition to make sure the sys does not try to violate T_env
        tmp_exists_env = self.bdd.exist(self.get_input_vars_prime(), self.get_t_env())
        out = tmp_all & tmp_exists_env

        return out

    def reachable(self, init_states):
        Q = self.bdd.false
        Q_prime = init_states
        while Q != Q_prime:
            Q = Q_prime
            post = self.game_one_step(Q_prime)
            Q_prime = Q_prime | post
        return Q

    def game_one_step(self, set_of_states):
        return self.exists_post_image(set_of_states)

    def exists_post_image(self, set_of_states):
        tmp = set_of_states & self.T_env & self.T_sys
        tmp_image = self.bdd.exist(self.vars, tmp)
        post_image = self.bdd.let(self.get_v_prime_to_v(), tmp_image)
        return post_image

    def update_change_cons(self, arg_skills_changed):
        for new_skill, old_skill in arg_skills_changed:
            old_cons = self.bdd.exist(self.get_output_vars(), self.change_cons & old_skill)
            new_cons = old_cons & new_skill
            self.change_cons = (self.change_cons & ~new_skill) | new_cons

    def update_not_allowed_repair(self, arg_skills_changed):
        for new_skill, old_skill in arg_skills_changed:
            old_cons = self.bdd.exist(self.get_output_vars(), self.not_allowed_repair & old_skill)
            new_cons = old_cons & new_skill
            self.not_allowed_repair = (self.not_allowed_repair & ~new_skill) | new_cons


class WinningStates:
    def __init__(self, arg_Z, arg_mY, arg_mX, arg_need_repair, arg_target):
        self.Z = arg_Z
        self.mY = arg_mY
        self.mX = arg_mX
        self.need_repair = arg_need_repair
        self.target = arg_target

    def get_z(self):
        return self.Z

    def get_my(self):
        return self.mY

    def get_mx(self):
        return self.mX

    def does_need_repair(self):
        return self.need_repair

    def get_target_states(self):
        return self.target


def print_debug(msg):
    # if DEBUG:
    if False:
        print(msg)


def compute_winning_states(arg_bdd, gs_internal, arg_opts):
    # Compute winning states
    Z_internal = arg_bdd.true
    Z_prime_internal = arg_bdd.false
    z_cnt = 0
    while Z_prime_internal != Z_internal:  # fixed point of Z
        print_debug("Starting another Z fixed point")
        print_expr(arg_bdd, "iter_" + str(z_cnt) + "_Z", Z_internal,
                   vars_ordering=gs_internal.get_vars_and_vars_prime(), do_print=DEBUG & DEBUG_WS_COMPUTE,
                   do_names=arg_opts['do_names'], arg_opts=arg_opts, to_file=arg_opts['to_file'])

        Z_prime_internal = Z_internal
        # memory
        mY = []
        mX = []
        for ii, sys_live_internal in enumerate(
                gs_internal.get_sys_live_guarantees()):  # loop through liveness guarantees
            # print_debug("Liveness guarantee: {}".format(ii))

            print_expr(arg_bdd, "iter_{}_Liveness_guarantee_{}".format(z_cnt, ii), arg_bdd.add_expr(sys_live_internal),
                       vars_ordering=gs_internal.get_vars_and_vars_prime(),
                       do_print=DEBUG & DEBUG_WS_COMPUTE, do_names=arg_opts['do_names'], arg_opts=arg_opts,
                       to_file=arg_opts['to_file'])

            Y = arg_bdd.false
            Y_prime = arg_bdd.true
            fpY_cnt = 0

            # if bdd_internal.add_expr(sys_live_internal) & gs_internal.cox(Z_internal) == bdd_internal.false: # or (z_cnt == 1 and ii == 0):
            #     print_expr(bdd_internal, "Cannot win. Z is:", Z_internal, vars_ordering=gs_internal.get_vars_and_vars_prime(),
            #                do_print=DEBUG & DEBUG_WS_COMPUTE)
            #     return WinningStates(Z_internal, bdd.false, bdd.false, True,
            #                          bdd_internal.add_expr(sys_live_internal))

            while Y_prime != Y:
                print_debug("Fixed point y cnt: {}".format(fpY_cnt))
                print_expr(arg_bdd, "iter_{}_live_{}_Y_{}".format(z_cnt, ii, fpY_cnt), Y,
                           vars_ordering=gs_internal.get_vars_and_vars_prime(), do_print=DEBUG & DEBUG_WS_COMPUTE,
                           do_names=arg_opts['do_names'], arg_opts=arg_opts, to_file=arg_opts['to_file'])
                Y_prime = Y

                print_expr(arg_bdd, "iter_{}_live_{}_y_{}_gs_internal.cox(Z_internal)".format(z_cnt, ii, fpY_cnt),
                           gs_internal.cox(Z_internal), vars_ordering=gs_internal.get_vars_and_vars_prime(),
                           do_print=DEBUG & DEBUG_WS_COMPUTE, do_names=arg_opts['do_names'], arg_opts=arg_opts,
                           to_file=arg_opts['to_file'])

                start = arg_bdd.add_expr(sys_live_internal) & gs_internal.cox(Z_internal)
                print_expr(arg_bdd, "iter_{}_live_{}_y_{}_start".format(z_cnt, ii, fpY_cnt), start,
                           vars_ordering=gs_internal.get_vars_and_vars_prime(),
                           do_print=DEBUG & DEBUG_WS_COMPUTE, do_names=arg_opts['do_names'],
                           arg_opts=arg_opts, to_file=arg_opts['to_file'])

                if start == arg_bdd.false:  # or (z_cnt == 1 and ii == 0):
                    print_expr(arg_bdd, "Cannot win. Z is:", Z_internal,
                               vars_ordering=gs_internal.get_vars_and_vars_prime(),
                               do_print=DEBUG & DEBUG_WS_COMPUTE)
                    return WinningStates(Z_internal, arg_bdd.false, arg_bdd.false, True,
                                         arg_bdd.add_expr(sys_live_internal))
                    # print("This will break")

                start = start | gs_internal.cox(Y)
                print_expr(arg_bdd, "iter_{}_live_{}_y_{}_start_with_coxY", start,
                           vars_ordering=gs_internal.get_vars_and_vars_prime(), do_print=DEBUG & DEBUG_WS_COMPUTE,
                           do_names=arg_opts['do_names'], arg_opts=arg_opts, to_file=arg_opts['to_file'])

                Y = arg_bdd.false
                for jj, env_live_internal in enumerate(gs_internal.get_env_live_assumptions()):
                    print_debug("Liveness assumption: {}".format(jj))
                    X = Z_internal
                    X_prime = arg_bdd.false
                    print_debug("Starting X fixed point")
                    fpX_cnt = 0
                    while X_prime != X:
                        print_expr(arg_bdd,
                                   "iter_{}_live_{}_y_{}_x_{}_x_begin".format(z_cnt, ii, fpY_cnt, fpX_cnt),
                                   X, vars_ordering=gs_internal.get_vars_and_vars_prime(),
                                   do_print=DEBUG & DEBUG_WS_COMPUTE, do_names=arg_opts['do_names'],
                                   arg_opts=arg_opts, to_file=arg_opts['to_file'])

                        print_expr(arg_bdd, "iter_{}_live_{}_y_{}_x_{}_cox_x".format(z_cnt, ii, fpY_cnt, fpX_cnt),
                                   gs_internal.cox(X), vars_ordering=gs_internal.get_vars_and_vars_prime(),
                                   do_print=DEBUG & DEBUG_WS_COMPUTE, do_names=arg_opts['do_names'],
                                   arg_opts=arg_opts, to_file=arg_opts['to_file'])

                        X_prime = X
                        X = start | ((~ arg_bdd.add_expr(env_live_internal)) & gs_internal.cox(X))

                        print_expr(arg_bdd, "iter_{}_live_{}_y_{}_x_{}_x_end".format(z_cnt, ii, fpY_cnt, fpX_cnt),
                                   X, vars_ordering=gs_internal.get_vars_and_vars_prime(),
                                   do_print=DEBUG & DEBUG_WS_COMPUTE, do_names=arg_opts['do_names'],
                                   arg_opts=arg_opts, to_file=arg_opts['to_file'])
                        fpX_cnt += 1

                    print_debug("Finished X fixed point")
                    Y = Y | X

                    print_expr(arg_bdd, "iter_{}_live_{}_y_{}_Y_after_X".format(z_cnt, ii, fpY_cnt), Y,
                               vars_ordering=gs_internal.get_vars_and_vars_prime(), do_print=DEBUG & DEBUG_WS_COMPUTE,
                               do_names=arg_opts['do_names'], arg_opts=arg_opts, to_file=arg_opts['to_file'])
                    # Add to mX
                    if ii >= len(mX):
                        mX.append([[]])
                    if fpY_cnt >= len(mX[ii]):
                        mX[ii].append([])
                    mX[ii][fpY_cnt].append(X)
                if ii >= len(mY):
                    mY.append([])
                mY[ii].append(Y)
                fpY_cnt += 1
            Z_internal = Y

            # if bdd_internal.add_expr(sys_live_internal) & gs_internal.cox(Z_internal) == bdd_internal.false: # or (z_cnt == 1 and ii == 0):
            #     print_expr(bdd_internal, "Cannot win. Z is:", Z_internal, vars_ordering=gs_internal.get_vars_and_vars_prime(),
            #                do_print=DEBUG & DEBUG_WS_COMPUTE)
            #     return WinningStates(Z_internal, bdd.false, bdd.false, True,
            #                          bdd_internal.add_expr(sys_live_internal))

            print_expr(arg_bdd, "iter_{}_live_{}_Z_after_Y".format(z_cnt, ii), Z_internal,
                       vars_ordering=gs_internal.get_vars_and_vars_prime(), do_print=DEBUG & DEBUG_WS_COMPUTE,
                       do_names=arg_opts['do_names'], arg_opts=arg_opts, to_file=arg_opts['to_file'])
        z_cnt += 1

    return WinningStates(Z_internal, mY, mX, False, arg_bdd.false)


def synthesize(arg_bdd, arg_gs, winning_states_internal):
    # Synthesize a strategy

    # number of bits needed for counter
    if len(arg_gs.get_sys_live_guarantees()) == 1:
        num_cntr_vars = 1
    else:
        num_cntr_vars = int(math.log(len(arg_gs.get_sys_live_guarantees()) - 1) / math.log(2) + 1)
    cntr_vars = []
    cntr_vars_prime = []
    for ii in range(num_cntr_vars):
        cntr_var_name = "cntr_" + str(ii)
        cntr_var_prime_name = cntr_var_name + "'"
        cntr_vars.append(cntr_var_name)
        cntr_vars_prime.append(cntr_var_prime_name)
        if not (cntr_var_name in arg_bdd.vars):
            arg_bdd.add_var(cntr_var_name)
            arg_bdd.add_var(cntr_var_prime_name)
    cntr_vars_and_prime = copy.copy(cntr_vars)
    cntr_vars_and_prime.extend(copy.copy(cntr_vars_prime))
    vars_internal = arg_gs.get_input_vars() + arg_gs.get_output_vars()
    vars_internal.extend(cntr_vars)
    vars_prime_internal = arg_gs.get_input_vars_prime() + arg_gs.get_output_vars_prime()
    vars_prime_internal.extend(cntr_vars_prime)

    zero_cntr = assign(arg_bdd, 0, cntr_vars_and_prime)
    strategy_init = arg_gs.get_t_init() & zero_cntr

    strategy_bdd = arg_bdd.false
    n_sys_live_guarantees = len(arg_gs.get_sys_live_guarantees())

    Z_internal = winning_states_internal.get_z()
    Z_prime_internal = arg_bdd.let(arg_gs.get_v_to_v_prime(), Z_internal)

    # check = arg_bdd.add_expr("symbol_0 & symbol_1 & symbol_2 & symbol_3 & symbol_6 & "
    #                          "!blue_person & !green_person & !blue_set_to_dirty_partition_0_5 & "
    #                          "!blue_clean_to_set_partition_0_4 & !blue_dirty_to_clean_partition_0_3 & "
    #                          "!green_set_to_dirty_partition_0_2 & !green_clean_to_set_partition_0_0 & "
    #                          "!green_clean_to_set_partition_0_1 & !green_dirty_to_clean_partition_0_6 & "
    #                          "!extra3 & !extra1 & !extra2 & symbol_0' & symbol_1' & symbol_2' & symbol_3' & symbol_6'"
    #                          "& !blue_person' & !green_person'"
    #                          "& !cntr_0 & !cntr_1")

    # Loop for rho1. Moves to the next goal once this goal is satisfied
    print_debug("Starting rho_1")
    for jj, sys_live in enumerate(arg_gs.get_sys_live_guarantees()):
        pre = arg_bdd.add_expr(sys_live) & Z_internal
        post_j = pre & arg_gs.get_t_env() & arg_gs.get_t_sys()

        rho_1 = assign(arg_bdd, jj, cntr_vars)
        rho_1 = rho_1 & post_j
        rho_1 = rho_1 & Z_prime_internal

        nxt_cntr_val = (jj + 1) % n_sys_live_guarantees
        nxt_cntr = assign(arg_bdd, nxt_cntr_val, cntr_vars_prime)

        rho_1 = rho_1 & nxt_cntr

        strategy_bdd = strategy_bdd | rho_1

        # print_expr(arg_bdd, "Check for init", check & strategy_bdd, vars_ordering=vars_internal + vars_prime_internal, do_print=True)

    # rho_2. attempt to satisfy liveness guarantee j
    print_debug("Starting rho_2")
    # same_cntr =  arg_bdd.add_expr("!cntr_0 & !cntr_1 & !cntr_0' & !cntr_1'")
    mY_internal = winning_states_internal.get_my()
    for jj, sys_live in enumerate(arg_gs.get_sys_live_guarantees()):
        mY_j = mY_internal[jj]
        low = mY_j[0]
        cntr_j = assign(arg_bdd, jj, cntr_vars)
        cntr_j_prime = assign(arg_bdd, jj, cntr_vars_prime)
        cntr_part = cntr_j & cntr_j_prime

        for rr, mY_j_r in enumerate(mY_j):
            not_low = ~ low
            pre = not_low & mY_j_r
            post = pre & arg_gs.get_t_env() & arg_gs.get_t_sys()
            nxt_low = arg_bdd.let(arg_gs.get_v_to_v_prime(), low)
            rho_2 = cntr_part & post
            rho_2 = rho_2 & nxt_low
            low = low | mY_j_r
            # strat_2 = strat_2 | rho_2
            strategy_bdd = strategy_bdd | rho_2

            # print_expr(arg_bdd, "Check for init", check & strategy_bdd, vars_ordering=vars_internal + vars_prime_internal, do_print=True)
            # print_expr(arg_bdd, "same cntr", same_cntr & strategy_bdd, vars_ordering=vars_internal + vars_prime_internal,
            #             do_print=True)
    mX_internal = winning_states_internal.get_mx()
    # rho 3
    print_debug("Starting rho_3")
    for ii, mX_ii in enumerate(mX_internal):
        low = arg_bdd.false
        cntr_ii = assign(arg_bdd, ii, cntr_vars)
        cntr_ii_prime = assign(arg_bdd, ii, cntr_vars_prime)
        cntr_part = cntr_ii & cntr_ii_prime

        for rr, mX_ii_rr in enumerate(mX_ii):
            for jj, env_live in enumerate(arg_gs.get_env_live_assumptions()):
                not_low = ~ low
                not_env_live = ~ arg_bdd.add_expr(env_live)
                pre = not_low & not_env_live
                pre = pre & mX_ii_rr[jj]
                post = pre & arg_gs.get_t_env() & arg_gs.get_t_sys()

                rho_3 = post & cntr_part
                nxt_mem_iirrjj = arg_bdd.let(arg_gs.get_v_to_v_prime(), mX_ii_rr[jj])
                rho_3 = rho_3 & nxt_mem_iirrjj

                low = low | mX_ii_rr[jj]

                strategy_bdd = strategy_bdd | rho_3

    # print_expr(bdd, "Strategy", strategy_bdd, arg_bdd.support(strategy_bdd))
    zero_cntr_not_prime = assign(arg_bdd, 0, cntr_vars)
    strategy_out = GameStructure(arg_bdd, vars_internal, vars_prime_internal,
                                 arg_gs.get_input_vars(), arg_gs.get_input_vars_prime(),
                                 arg_gs.get_input_vars_double_prime(),
                                 arg_gs.get_output_vars(), arg_gs.get_output_vars_prime(),
                                 arg_gs.get_output_vars_double_prime(),
                                 arg_gs.get_t_env(), arg_gs.get_t_env_hard(), arg_gs.get_t_env_not_hard(),
                                 strategy_bdd, arg_gs.get_t_sys_hard(), arg_gs.get_t_sys_not_hard(),
                                 arg_gs.get_env_live_assumptions(), arg_gs.get_sys_live_guarantees(),
                                 arg_gs.get_t_env_init() & zero_cntr_not_prime,
                                 arg_gs.get_t_sys_init() & zero_cntr_not_prime,
                                 cntr_vars, cntr_vars_prime,
                                 arg_gs.get_change_cons(), arg_gs.get_not_allowed_repair())

    return strategy_out


def determinize_strategy(arg_bdd, arg_strategy):
    deterministic_strat_internal = arg_bdd.false
    cntr_vars_internal = arg_strategy.get_cntr_vars()
    vars_internal = arg_strategy.get_vars()

    init_states = list(
        arg_bdd.pick_iter(arg_strategy.get_t_init() & assign(arg_bdd, 0, cntr_vars_internal), vars_internal))
    states_to_process = []
    for init_state in init_states:
        states_to_process.append(arg_bdd.cube(init_state))
    states_visited = copy.copy(states_to_process)

    T_sys_strat = arg_strategy.get_t_sys()

    while not (len(states_to_process) == 0):
        current_state = states_to_process.pop(0)
        print_expr(arg_bdd, "Current State", current_state, vars_ordering=arg_strategy.get_vars_and_vars_prime(),
                   do_print=DEBUG_DETERMINIZE, do_names=False)
        next_env = current_state & arg_strategy.get_t_env()
        print_expr(arg_bdd, "Next States Env Chooses", next_env, vars_ordering=arg_strategy.get_vars_and_vars_prime(),
                   do_print=DEBUG_DETERMINIZE, do_names=False)

        if next_env == arg_bdd.false:
            print("This shouldn't happen bc we are not violating the enviornment spec")

        # Enumerate the environment moves and decide what the system is doing from each of these
        states_system_makes_choice_from = list(
            arg_bdd.pick_iter(next_env, arg_strategy.get_vars_and_input_prime()))

        for after_env_choice in states_system_makes_choice_from:
            # All possible valid choices of states by the system
            after_env_choice_bdd = arg_bdd.cube(after_env_choice)
            print_expr(arg_bdd, "System chooses what to do with this state", after_env_choice_bdd,
                       vars_ordering=arg_strategy.get_vars_and_vars_prime(), do_print=DEBUG_DETERMINIZE, do_names=False)
            next_states_prime = arg_bdd.exist(vars_internal, after_env_choice_bdd & T_sys_strat)
            print_expr(arg_bdd, "Possible System Choices", next_states_prime,
                       vars_ordering=arg_strategy.get_vars_and_vars_prime(), do_print=DEBUG_DETERMINIZE, do_names=False)

            next_states = arg_bdd.let(arg_strategy.get_v_prime_to_v(), next_states_prime)
            # Pick one possible action out of the many possible actions for the result of the action
            # choice = next(arg_bdd.pick_iter(next_states))
            all_choices = list(arg_bdd.pick_iter(next_states))
            choice = all_choices[np.random.randint(len(all_choices))]
            choice_bdd = arg_bdd.cube(choice)
            print_expr(arg_bdd, "System Choice State", choice_bdd, vars_ordering=arg_strategy.get_vars_and_vars_prime(),
                       do_print=DEBUG_DETERMINIZE, do_names=False)

            # Add the chosen transition to the deterministic strategy
            choice_prime = arg_bdd.let(arg_strategy.get_v_to_v_prime(), choice_bdd)
            chosen_transition = after_env_choice_bdd & choice_prime
            deterministic_strat_internal = deterministic_strat_internal | chosen_transition

            # Add the next state to the set of states to process if necessary
            if states_visited.count(choice_bdd) == 0:
                states_to_process.append(choice_bdd)
                states_visited.append(choice_bdd)

    deterministic_strat_out = GameStructure(arg_bdd, vars_internal, arg_strategy.get_vars_prime(),
                                            arg_strategy.get_input_vars(), arg_strategy.get_input_vars_prime(),
                                            arg_strategy.get_input_vars_double_prime(),
                                            arg_strategy.get_output_vars(), arg_strategy.get_output_vars_prime(),
                                            arg_strategy.get_output_vars_double_prime(),
                                            arg_strategy.get_t_env(), arg_strategy.get_t_env_hard(),
                                            arg_strategy.get_t_env_hard(),
                                            deterministic_strat_internal, arg_strategy.get_t_sys_hard(),
                                            arg_strategy.get_t_sys_not_hard(),
                                            arg_strategy.get_env_live_assumptions(),
                                            arg_strategy.get_sys_live_guarantees(),
                                            arg_strategy.get_t_env_init(),
                                            arg_strategy.get_t_sys_init(),
                                            arg_strategy.get_cntr_vars(),
                                            arg_strategy.get_cntr_vars_prime(),
                                            arg_strategy.get_change_cons(),
                                            arg_strategy.get_not_allowed_repair())
    return deterministic_strat_out


def perform_repair(arg_bdd, arg_gs, arg_winning_states, arg_target_states, arg_T_previously_changed, arg_opts):
    print("perform_repair call number: {}".format(arg_opts["post_repair_cnt"]))
    repaired_T_env = arg_gs.get_t_env()
    repaired_T_sys = arg_gs.get_t_sys_not_hard()

    current_winning_states = arg_winning_states.get_z()
    gs_internal = arg_gs

    while True:

        # Finds the set of winning states
        tmp = arg_bdd.false

        pre_Z = arg_bdd.false
        while tmp != current_winning_states:
            pre_Z = gs_internal.cox(current_winning_states)
            tmp = current_winning_states
            current_winning_states = current_winning_states | pre_Z

        # Computes the intersection of the controllable predecssor and the target states
        # Returns if the intersection and cover are appropriate
        intersect = arg_target_states & pre_Z
        if arg_opts['cover'] and intersect == arg_target_states:
            return gs_internal
        elif not arg_opts['cover'] and intersect != arg_bdd.false:
            return gs_internal

        # Modify the postconditions and preconditions
        # TODO: Option to change order of modify post / pre
        tmp_T_env = repaired_T_env
        tmp_T_sys = repaired_T_sys

        acts_changed = []
        T_swapped_pre = arg_bdd.false
        T_swapped_post = arg_bdd.false
        if arg_opts['post_first']:
            tmp_T_env, tmp_T_sys, acts_changed, arg_T_previously_changed, T_swapped_post = modify_postconditions(
                arg_bdd,
                repaired_T_env,
                repaired_T_sys,
                current_winning_states,
                gs_internal,
                arg_T_previously_changed,
                arg_opts)

            gs_internal.update_change_cons(acts_changed)
            gs_internal.update_not_allowed_repair(acts_changed)

            if tmp_T_env == repaired_T_env and tmp_T_sys == repaired_T_sys:
                tmp_T_env, tmp_T_sys, T_swapped_pre = modify_preconditions(arg_bdd, tmp_T_env,
                                                                           tmp_T_sys,
                                                                           current_winning_states, gs_internal,
                                                                           arg_opts)
            else:
                arg_opts['post_first'] = False
        else:
            tmp_T_env, tmp_T_sys, T_swapped_pre = modify_preconditions(arg_bdd, repaired_T_env, repaired_T_sys,
                                                                       current_winning_states,
                                                                       gs_internal,
                                                                       arg_opts)
            if tmp_T_env == repaired_T_env and tmp_T_sys == repaired_T_sys:

                gs_internal.update_change_cons(acts_changed)
                gs_internal.update_not_allowed_repair(acts_changed)

                tmp_T_env, tmp_T_sys, acts_changed, arg_T_previously_changed, T_swapped_post = modify_postconditions(
                    arg_bdd,
                    tmp_T_env,
                    tmp_T_sys,
                    current_winning_states,
                    gs_internal,
                    arg_T_previously_changed,
                    arg_opts)

                gs_internal.update_change_cons(acts_changed)
                gs_internal.update_not_allowed_repair(acts_changed)
            else:
                arg_opts['post_first'] = True

        # Determine if modifications have occurred
        if (tmp_T_env & arg_gs.get_t_env_hard() != repaired_T_env & arg_gs.get_t_env_hard()) or (
                tmp_T_sys & arg_gs.get_t_sys_hard() != repaired_T_sys & arg_gs.get_t_sys_hard()):
        # if (tmp_T_env & arg_gs.get_t_env_hard() != repaired_T_env & arg_gs.get_t_env_hard()) or (
        #         tmp_T_sys & arg_gs.get_t_sys_hard() != repaired_T_sys & arg_gs.get_t_sys_hard()):
            arg_opts['post_repair_cnt'] += 1
            if tmp_T_env != repaired_T_env:
                repaired_T_env = tmp_T_env
            if tmp_T_sys != repaired_T_sys:
                repaired_T_sys = tmp_T_sys

            # Update with repaired T_env and T_sys
            # TODO:Update T_{env,sys}_not_hard
            gs_internal.T_env = repaired_T_env & gs_internal.get_t_env_hard()
            gs_internal.T_sys = repaired_T_sys & gs_internal.get_t_sys_hard()
            gs_internal.T_sys_not_hard = repaired_T_sys
            gs_internal.T_env_not_hard = repaired_T_env

            # if 'skill2_new' in gs_internal.get_vars() and gs_internal.get_t_sys() & arg_bdd.add_expr(
            #     "skill1 & !skill2 & !skill3 & !skill1b & !skill2b & !skill3b & !skill2_new & !skill1' & !skill2' & !skill3' & skill1b' & !skill2b' & !skill3b' & !skill2_new' & x0' & !x1' & !x2' & !y0' & y1' & !y2'") != arg_bdd.false:
            #     print_expr(arg_bdd, "Winning_states", gs_internal.get_t_sys() & arg_bdd.add_expr(
            #         "skill1 & !skill2 & !skill3 & !skill1b & !skill2b & !skill3b & !skill2_new & !skill1' & !skill2' & !skill3' & skill1b' & !skill2b' & !skill3b' & !skill2_new' & x0' & !x1' & !x2' & !y0' & y1' & !y2'"),
            #                vars_ordering=gs_internal.get_vars_and_vars_prime(),
            #                do_print=True)
            if arg_opts['return_with_one_repair']:
                return gs_internal, acts_changed, arg_T_previously_changed, T_swapped_pre, T_swapped_post
        else:
            arg_opts['post_repair_cnt'] += 1
            raise Exception("No repair found")


def find_encoded_skills(arg_bdd, arg_gs, arg_T_env):
    return arg_bdd.exist(arg_gs.get_input_vars_prime(), arg_bdd.exist(arg_gs.get_vars(),
                                                                      arg_gs.get_t_env_hard()) & ~arg_T_env) & arg_T_env


def modify_postconditions(arg_bdd, arg_T_env, arg_T_sys, arg_winning_states, arg_gs, arg_T_previously_changed,
                          arg_opts):
    """
    Modifies the postconditions by changing a single postcondition such that it increases the winning states

    Parameters
    ----------
    arg_bdd
    arg_T_env
    arg_T_sys
    arg_winning_states
    arg_gs
    arg_T_previously_changed
    arg_opts

    Returns
    -------

    """
    print("modify postconditions")
    # These are the states that are winning at the next time that the system must get to
    winning_prime = arg_bdd.let(arg_gs.get_v_to_v_prime(), arg_winning_states)
    winning_p_dp = arg_bdd.let(arg_gs.get_inputprime_to_inputdoubleprime(), winning_prime)
    T_sys_mutable = arg_gs.get_t_sys_not_hard()

    print_expr(arg_bdd, "Winning_states", arg_winning_states, vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
               do_print=DEBUG)

    # We want to change the postconditions of the skills, but we don't want to have to change them to a specific skill
    # that already exists, because that skill may not exist and we are changing other things at the same time.
    # Instead, we find all the current skills, combine that with possible changes, and find which of these changes are
    # winning. We then choose just one of these changes (we can later select multiple changes if they all start at the
    # same precondition)
    # Finds the skills that are actually encoded and not the result of LTL just trying to do things
    # Line 1
    T_reachable = arg_bdd.let(arg_gs.get_v_prime_to_v(),
                              arg_bdd.exist(arg_gs.get_vars(), T_sys_mutable & arg_T_env)) & arg_T_env
    # T_reachable = arg_bdd.let(arg_gs.get_v_to_v_prime(),
    #                           arg_bdd.let(arg_gs.get_inputprime_to_inputdoubleprime(), arg_T_env)) & T_sys_mutable
    print_expr(arg_bdd, "Only actually known skills", T_reachable,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG)

    # Line 2
    # The full skills that are encoded
    no_op = "!" + " & !".join(arg_opts['existing_skills'])
    T_full_skills = arg_gs.get_t_sys_not_hard() & T_reachable & ~arg_bdd.add_expr(no_op)
    print_expr(arg_bdd, "T_full_skills", T_full_skills, vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
               do_print=DEBUG)

    # Line 3
    # The full skills that aren't currently winning
    T_full_skills_not_winning = T_full_skills & ~winning_prime & ~arg_winning_states
    print_expr(arg_bdd, "T_full_skills_not_winning", T_full_skills_not_winning,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG)

    # The possible changes that can be applied to the skills (minus the ones we don't allow because they are no-op or
    # were found not to work).
    # Also don't allow postconditions to go back to the initial preconditions.
    # Also do not allow the postcondition to be changed to a state that is already in the skill unless it has the same
    # precondition as the state it is changing to
    # Line 5
    T_pres = arg_bdd.exist(arg_gs.get_input_vars_prime(), T_reachable)
    T_skill_with_pre_dp = arg_bdd.let(arg_gs.get_input_to_inputdoubleprime(), T_pres)
    T_skill_with_post_dp = arg_bdd.let(arg_gs.get_inputprime_to_inputdoubleprime(), T_reachable)
    T_no_effect = find_skill_has_no_effect(arg_bdd, arg_gs, "v_and_dp")
    T_possible_changes = T_full_skills_not_winning & \
                         arg_gs.get_change_cons_p_and_dp() & \
                         arg_gs.get_not_allowed_repair_v_and_dp() & \
                         ~T_no_effect & \
                         (~T_skill_with_pre_dp | T_skill_with_post_dp) & \
                         arg_bdd.let(arg_gs.get_inputprime_to_inputdoubleprime(), arg_T_sys)# & \
                         # arg_bdd.let(arg_gs.get_inputprime_to_inputdoubleprime(), arg_T_sys & arg_gs.get_t_sys_hard()) & \
                         # arg_bdd.let(arg_gs.get_inputprime_to_inputdoubleprime(), arg_gs.get_t_env_hard())

    print_expr(arg_bdd, "T_possible_changes", T_possible_changes,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG)

    # These are the changes that are winning
    T_winning_changes = T_possible_changes & winning_p_dp
    print_expr(arg_bdd, "T_winning_changes", T_winning_changes,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG)

    # if arg_opts['enforce_reactive_variables']:
    #     T_winning_changes = arg_bdd.exist(arg_opts['reactive_variables'], T_winning_changes)
    #     print_expr(arg_bdd, "T_winning_changes (after enforcing reactive variables)", T_winning_changes,
    #                vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG)

    # Lines 6-11
    if T_winning_changes != arg_bdd.false:
        # non_rv = copy.deepcopy(arg_gs.get_vars_and_prime_and_dp())
        # if arg_opts['enforce_reactive_variables']:
        #     for rv in arg_opts['reactive_variables']:
        #         non_rv.remove(rv)
        T_winning_changes_list = list(arg_bdd.pick_iter(T_winning_changes, care_vars=arg_gs.get_vars_and_prime_and_dp()))
        sel_idx = np.random.randint(len(T_winning_changes_list))
        # sel_idx = 11
        # if arg_opts['post_repair_cnt'] == 1 and arg_opts['generate_figure'] == 'symbolic':
        #     sel_idx = 0
        # print("Post_repair_cnt: {}, sel_idx: {}".format(arg_opts['post_repair_cnt'], sel_idx))
        T_winning_change_sel = arg_bdd.cube(T_winning_changes_list[sel_idx])
    else:
        T_winning_change_sel = arg_bdd.false
    print_expr(arg_bdd, "T_winning_change_sel", T_winning_change_sel,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=True)
    if arg_opts['enforce_reactive_variables']:
        T_winning_change_sel = arg_bdd.exist(arg_opts['reactive_variables'], T_winning_change_sel)
        print_expr(arg_bdd, "T_winning_change_sel (after enforcing reactive variables)", T_winning_change_sel,
                   vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG)

    T_winning_change_new = arg_bdd.exist(arg_gs.get_output_vars_prime(),
                                         arg_bdd.let(arg_gs.get_inputdoubleprime_to_inputprime(),
                                                     arg_bdd.exist(arg_gs.get_input_vars_prime(),
                                                                   T_winning_change_sel)))
    print_expr(arg_bdd, "T_winning_change_new", T_winning_change_new,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG)

    # Lines 12-15
    # For the environment transition, we just need to remove the old postcondition(s) and add the new postcondition(s)
    T_env_out = arg_T_env & ~arg_bdd.exist(arg_gs.get_output_vars_prime() + arg_gs.get_input_vars_double_prime(),
                                           T_winning_change_sel)
    T_env_out = T_env_out | arg_bdd.exist(arg_gs.get_output_vars_prime(), T_winning_change_new)
    print_expr(arg_bdd, "T_env removed",
               arg_bdd.exist(arg_gs.get_output_vars_prime() + arg_gs.get_input_vars_double_prime(),
                             T_winning_change_sel),
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=True)

    # # Line 16-21
    # # Modify the preconditions
    T_current_skill = arg_bdd.exist(arg_gs.get_input_vars() + arg_gs.get_vars_prime(), T_winning_change_new)
    # T_next_skill = arg_bdd.exist(arg_gs.get_vars_and_input_prime(), T_winning_change_new & arg_gs.get_t_sys_not_hard() & arg_gs.get_t_sys_hard())
    # print_expr(arg_bdd, "T_next_skill", T_next_skill,
    #            vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=True)
    # T_next_skill_in_current = arg_bdd.let(arg_gs.get_v_prime_to_v(), T_next_skill)
    # if T_current_skill & T_next_skill_in_current == arg_bdd.false:
    #     T_sys_add = T_winning_change_new & arg_bdd.let(arg_gs.get_output_to_outputprime(), T_current_skill)
    # else:
    #     T_sys_add = arg_bdd.false
    #
    # print_expr(arg_bdd, "T_sys_add", T_sys_add,
    #            vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=True)
    # T_sys_out = T_sys_mutable | T_sys_add
    T_sys_out = T_sys_mutable

    # if T_sys_out != arg_T_sys:
    #     print("System transitions changed!")

    acts_changed = [[T_current_skill, T_current_skill]]
    T_swapped = arg_bdd.false

    return T_env_out, T_sys_out, acts_changed, arg_T_previously_changed, T_swapped


def find_final_post(arg_bdd, arg_gs, arg_skills, arg_T_env, arg_T_sys_nh):
    T_out = arg_bdd.false
    for skill in arg_bdd.pick_iter(arg_skills, care_vars=arg_gs.get_output_vars()):
        T_skill = arg_bdd.cube(skill)
        T_env_skill = arg_T_env & T_skill
        T_pres = arg_bdd.exist(arg_gs.get_input_vars_prime(), T_env_skill)
        T_pres_as_posts = arg_bdd.let(arg_gs.get_input_to_inputprime(), T_pres)
        T_final_env_trans = T_env_skill & ~T_pres_as_posts
        T_final_post = arg_bdd.exist(arg_gs.get_input_vars(), T_final_env_trans)
        T_out = T_out | T_final_post

    return T_out


def modify_preconditions(arg_bdd, arg_T_env, arg_T_sys, arg_winning_states, arg_gs, arg_opts):
    print("modify preconditions")
    # np.random.seed(42)
    print_expr(arg_bdd, "winning states", arg_winning_states, vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
               do_print=DEBUG_PRE)

    winning_prime = arg_bdd.let(arg_gs.get_v_to_v_prime(), arg_winning_states)
    T_sys_mutable = arg_gs.get_t_sys_not_hard()

    # Line 1
    # All skills that could win if not for the hard constraints.
    winning_sys_trans = winning_prime & T_sys_mutable
    T_sys_can_win = arg_bdd.exist(arg_gs.get_output_vars_prime(), winning_sys_trans)
    # print_expr(arg_bdd, "T_sys_can_win (E action respecting T_sys_not_hard to get to winning states)", T_sys_can_win,
    #            vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG_PRE)

    # Line 2
    # For any outcome of the environment, the system is able to win
    # Only checks the outcomes of the grounded variables, because the user controlled variables can change at any time
    # If we require them to exist to win, then we will never find modifications for livenesses with user controlled
    # propositions as part of them because the environment can just change the propositions so not **all** of them are
    # winning. This has no effect on reacting to user controlled variables during tasks (which should really just not
    # happen because we assume the system can't change anything during skill execution so they just unnecessarily make
    # the spec more complicated)
    no_skills = "!" + " & !".join(arg_opts['existing_skills'])
    input_vars_prime_minus_reactive = copy.deepcopy(arg_gs.get_input_vars_prime())
    if arg_opts['enforce_reactive_variables']:
        for rv in arg_opts['reactive_variables_current']:
            input_vars_prime_minus_reactive.remove(rv + "'")
    # T_sys_always_wins = arg_bdd.forall(input_vars_prime_minus_reactive, (~arg_T_env) | T_sys_can_win) & ~arg_bdd.add_expr(no_skills)
    T_sys_always_wins = (arg_bdd.forall(input_vars_prime_minus_reactive,
                                        arg_bdd.add_expr(arg_gs.get_env_live_assumptions()[0]) | T_sys_can_win) |
                         arg_bdd.forall(input_vars_prime_minus_reactive, (~arg_T_env) | T_sys_can_win)) \
                        & ~arg_bdd.add_expr(no_skills)
    # print_expr(arg_bdd, "A x get to Y", T_sys_always_wins,
    #            vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG)

    # Line 3
    # Find the skills that are actually possible to execute (like T_encoded, but hopefully making more sense to the reader)
    T_reachable = arg_bdd.let(arg_gs.get_v_prime_to_v(),
                              arg_bdd.exist(arg_gs.get_vars(), T_sys_mutable & arg_T_env)) & arg_T_env
    print_expr(arg_bdd, "T_reachable", T_reachable, vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
               do_print=DEBUG_PRE)

    # Line 4
    # The full state from which the system will always win and the skills are real
    T_sys_always_wins_and_reachable = T_sys_always_wins & T_reachable & winning_prime & T_sys_mutable
    print_expr(arg_bdd, "T_sys_always_wins_and_reachable", T_sys_always_wins_and_reachable,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
               do_print=DEBUG_PRE)

    # We enforce that suggestions can't include existing states in skills (the preconditions and final postcondition)
    # Line 5
    # These are the skills that might be changed
    T_possible_skills_changing = arg_bdd.exist(arg_gs.get_input_vars_and_prime() + arg_gs.get_output_vars_prime(),
                                               T_sys_always_wins_and_reachable)
    print_expr(arg_bdd, "T_possible_skills_changing", T_possible_skills_changing,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG_PRE)

    # Line 6
    # These are the preconditions of skills that might be changed
    T_pres_in_skill = arg_bdd.let(arg_gs.get_input_to_inputdoubleprime(), arg_bdd.exist(arg_gs.get_input_vars_prime(),
                                                                                        T_reachable & T_possible_skills_changing))
    print_expr(arg_bdd, "T_pres_in_skill", T_pres_in_skill, vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
               do_print=DEBUG_PRE)

    # Line 7
    # These are the final postconditions of skills that might be changed
    T_final_post = arg_bdd.let(arg_gs.get_inputprime_to_inputdoubleprime(),
                               find_final_post(arg_bdd, arg_gs, T_possible_skills_changing, T_reachable, T_sys_mutable))
    print_expr(arg_bdd, "T_final_post", T_final_post, vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
               do_print=DEBUG_PRE)

    # Line 8
    # These are the possible changes to the preconditions while not allowing changes to preconditions that are already in the skill
    # Also restricts that the change must obey hard system constraints
    skill_has_no_effect = find_skill_has_no_effect(arg_bdd, arg_gs, 'p_and_dp')
    T_possible_changes_in_dp_all = T_sys_always_wins_and_reachable & \
                                   arg_bdd.let(arg_gs.get_v_prime_to_v(), arg_gs.get_change_cons_p_and_dp()) & \
                                   arg_bdd.let(arg_gs.get_input_to_inputdoubleprime(),
                                               arg_gs.get_not_allowed_repair()) & \
                                   ~T_pres_in_skill & \
                                   arg_bdd.let(arg_gs.get_input_to_inputdoubleprime(), arg_gs.get_t_sys_hard()) & \
                                   ~arg_bdd.let(arg_gs.get_input_to_inputdoubleprime(),
                                                T_sys_always_wins_and_reachable) & \
                                   ~skill_has_no_effect & \
                                   ~T_final_post
    print_expr(arg_bdd, "T_possible_changes_in_dp_all", T_possible_changes_in_dp_all,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG_PRE)

    # Line 10
    # Only select one precondition to be added
    all_possible_changes = list(arg_bdd.pick_iter(T_possible_changes_in_dp_all,
                                                  care_vars=arg_gs.get_vars_and_prime_and_dp()))
    if len(all_possible_changes) == 0:
        return arg_T_env, arg_gs.get_t_sys_not_hard(), arg_bdd.false
    sel_idx = np.random.randint(len(all_possible_changes))
    # if arg_opts['post_repair_cnt'] == 0 and arg_opts['generate_figure'] == 'symbolic':
    #     sel_idx = 7
    # if arg_opts['post_repair_cnt'] == 0 and arg_opts['generate_figure'] == 'integrated':
    #     sel_idx = 5
    # print("Post_repair_cnt: {}, sel_idx: {}".format(arg_opts['post_repair_cnt'], sel_idx))
    T_selected_change = arg_bdd.cube(all_possible_changes[sel_idx])
    print_expr(arg_bdd, "T_selected_change", T_selected_change,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG_PRE)

    # Line 11
    # Select other valid changes if this is the precondition to the last transition
    T_selected_change = T_sys_mutable & arg_bdd.exist(arg_gs.get_output_vars_prime(), T_selected_change)
    print_expr(arg_bdd, "T_selected_change expanded", T_selected_change,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG_PRE)

    # Line 12
    if arg_opts['enforce_reactive_variables']:
        T_selected_change = arg_bdd.exist(arg_opts['reactive_variables'], T_selected_change)
        print_expr(arg_bdd, "T_possible_changes_in_dp (enforce reactive variables)", T_selected_change,
                   vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG_PRE)

    # Line 16
    # This is the precondition that is added
    T_new_full_skill = arg_bdd.let(arg_gs.get_inputdoubleprime_to_input(),
                                   arg_bdd.exist(arg_gs.get_input_vars(), T_selected_change))
    print_expr(arg_bdd, "T_new_full_skill (new precondition)", T_new_full_skill,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
               do_print=DEBUG_PRE)

    # We now need to add the transitions to the system transitions
    # line 17
    # This is the old skill
    T_old_full_skill = arg_bdd.exist(arg_gs.get_input_vars_double_prime(), T_selected_change)
    print_expr(arg_bdd, "T_old_full_skill (old precondition)", T_old_full_skill,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
               do_print=DEBUG_PRE)

    # Find the pre-preconditions
    # Lines 17 and 18
    # Add the new precondition to the system transition where the old precondition was
    T_old_pre_primed = T_sys_mutable & arg_bdd.let(arg_gs.get_v_to_v_prime(),
                                                   arg_bdd.exist(arg_gs.get_vars_prime(), T_old_full_skill))
    T_old_pre_pre = arg_bdd.exist(arg_gs.get_vars_prime(), T_old_pre_primed)
    T_new_pre_and_skill = arg_bdd.exist(arg_gs.get_vars_prime(), T_new_full_skill)
    T_new_pre_primed = T_old_pre_pre & arg_bdd.let(arg_gs.get_v_to_v_prime(), T_new_pre_and_skill)
    print_expr(arg_bdd, "T_new_pre_primed (new precondition)", T_new_pre_primed,
               vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
               do_print=DEBUG_PRE)

    # Check if this is the initial precondition. If so, don't let skills switch to this one in the middle
    # Line 20
    if arg_bdd.forall(arg_gs.get_output_vars(), T_new_pre_primed) == T_new_pre_primed:
        T_new_pre_input_primed = arg_bdd.exist(arg_gs.get_output_vars_prime(), T_new_pre_primed)

        print_expr(arg_bdd, "Skills that are available from the new precondition",
                   T_new_pre_input_primed & arg_gs.get_t_sys_not_hard(),
                   vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG_PRE)

        # Line 21
        T_all_skills_can_be_applied = arg_bdd.forall(arg_gs.get_output_vars(), T_new_pre_input_primed & T_sys_mutable)
        print_expr(arg_bdd, "All skills can be applied", T_all_skills_can_be_applied,
                   vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG_PRE)
        # print_expr(arg_bdd, "All skills cannot be applied", ~T_all_skills_can_be_applied, vars_ordering=arg_gs.get_vars_and_prime_and_dp())

        T_remove_from_pre = arg_bdd.exist(arg_gs.get_output_vars_prime(), T_new_pre_input_primed &
                                          T_sys_mutable & ~T_all_skills_can_be_applied)
        print_expr(arg_bdd, "T_remove_from_pre", T_remove_from_pre,
                   vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG_PRE)

        T_new_pre_primed = T_new_pre_primed & ~T_remove_from_pre
        print_expr(arg_bdd, "T_new_pre_primed (new precondition) (revised)", T_new_pre_primed,
                   vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=DEBUG_PRE)

    # Line 26
    T_sys_new = T_sys_mutable | T_new_pre_primed | T_new_full_skill

    # Adds the new preconditions to the environment transitions.
    # First, removes all transitions associated with the new precondition and skill
    # We don't have to worry about messing up a skill, because we specifically dissallowed that above
    # Also adds the condition that the old precondition of the precondition goes to the new precondition if it is an
    # intermediate state
    # Line 27
    # Finds the precondition to the precondition with respect to the environment transitions (that are reachable)
    T_cur_skill = arg_bdd.exist(arg_gs.get_input_vars() + arg_gs.get_vars_prime(), T_new_full_skill)
    T_old_pre_primed_env = arg_bdd.let(arg_gs.get_v_to_v_prime(),
                                       arg_bdd.exist(arg_gs.get_vars_prime() + arg_gs.get_output_vars(),
                                                     T_old_full_skill)) & T_cur_skill & T_reachable
    T_new_pre_pre = arg_bdd.exist(arg_gs.get_input_vars_prime(), T_old_pre_primed_env)

    # Line 28
    T_new_pre_pre_env = T_new_pre_pre & arg_bdd.let(arg_gs.get_v_to_v_prime(),
                                                    arg_bdd.exist(arg_gs.get_output_vars(), T_new_pre_and_skill))
    print_expr(arg_bdd, "T_new_pre_pre_env", T_new_pre_pre_env, vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
               do_print=DEBUG_PRE)

    # Line 29 and 30
    T_env_new = ((arg_T_env & ~T_new_pre_and_skill) | arg_bdd.exist(arg_gs.get_output_vars_prime(),
                                                                    T_new_full_skill) | T_new_pre_pre_env) & arg_gs.get_t_env_hard()

    return T_env_new, T_sys_new, arg_bdd.false


def find_skill_has_no_effect(arg_bdd, arg_gs, arg_which_vars):
    no_change_list = []
    for var in arg_gs.get_input_vars():
        if arg_which_vars == 'p_and_dp':
            no_change_list.append("(" + var + "' <-> " + var + "'')")
        elif arg_which_vars == "v_and_dp":
            no_change_list.append("(" + var + " <-> " + var + "'')")
        else:
            raise Exception("arg_which_vars must be p_and_dp or v_and_dp")
    return arg_bdd.add_expr(" & ".join(no_change_list))


def find_same_skill(arg_bdd, arg_gs, arg_which_vars):
    no_change_list = []
    for var in arg_gs.get_output_vars():
        if arg_which_vars == 'v_and_p':
            no_change_list.append("(" + var + " <-> " + var + "')")
        else:
            raise Exception("arg_which_vars must be v_and_p")
    return arg_bdd.add_expr(" & ".join(no_change_list))


def report_skill_revision(arg_bdd, arg_original_gs, arg_deterministic_strat, arg_opts):
    winning_states_internal = arg_deterministic_strat.reachable(arg_deterministic_strat.get_t_init())

    winning_states_system_internal = winning_states_internal & arg_deterministic_strat.get_t_env()

    # For this we don't care about the cntr variables, just what skills are possible
    if arg_opts['enforce_reactive_variables']:
        relaxed_T_sys_internal = arg_bdd.exist(arg_deterministic_strat.get_cntr_vars() + arg_opts['reactive_variables'],
                                               arg_deterministic_strat.get_t_sys())
    else:
        relaxed_T_sys_internal = arg_bdd.exist(arg_deterministic_strat.get_cntr_vars(),
                                               arg_deterministic_strat.get_t_sys())

    T_added_sys = (relaxed_T_sys_internal & ~arg_original_gs.get_t_sys()) & winning_states_system_internal
    T_removed_env = arg_bdd.exist(arg_opts['reactive_variables'] + arg_deterministic_strat.get_cntr_vars_and_prime(), (
            arg_original_gs.get_t_env() & ~arg_deterministic_strat.get_t_env()) & winning_states_internal)
    modifications = []
    if T_removed_env != arg_bdd.false:
        # if T_added_sys != arg_bdd.false or T_removed_env != arg_bdd.false:
        T_skills_modified_env = arg_bdd.exist(arg_deterministic_strat.get_input_vars_and_prime(), T_removed_env)
        # T_skills_modified_sys = arg_bdd.exist(arg_deterministic_strat.get_vars_and_input_prime(), T_added_sys)
        # T_skills_modified_sys_as_current = arg_bdd.let(arg_original_gs.get_outputprime_to_output(), T_skills_modified_sys)
        # skills_modified = arg_bdd.pick_iter(arg_bdd.exist(arg_deterministic_strat.get_cntr_vars_prime(), T_skills_modified_env | T_skills_modified_sys_as_current), care_vars=arg_deterministic_strat.get_output_vars())
        skills_modified = arg_bdd.pick_iter(
            arg_bdd.exist(arg_deterministic_strat.get_cntr_vars_prime(), T_skills_modified_env),
            care_vars=arg_deterministic_strat.get_output_vars())
        for skill_modified in skills_modified:
            one_mod = []
            skill_bdd = arg_bdd.cube(skill_modified)
            print_expr(arg_bdd, "Skill postconditions modified:", skill_bdd,
                       vars_ordering=arg_deterministic_strat.get_vars_and_vars_prime(), do_print=DEBUG_REVISION)
            print_expr(arg_bdd, "New skill:", skill_bdd & arg_deterministic_strat.get_t_env() & winning_states_internal,
                       vars_ordering=arg_deterministic_strat.get_vars_and_vars_prime(), do_print=DEBUG_REVISION)
            one_mod.append(skill_bdd)
            if arg_opts['enforce_reactive_variables']:
                one_mod.append(
                    arg_bdd.exist(arg_opts['reactive_variables'] + arg_deterministic_strat.get_cntr_vars_and_prime(),
                                  skill_bdd & T_removed_env))
                one_mod.append(
                    arg_bdd.exist(arg_opts['reactive_variables'] + arg_deterministic_strat.get_cntr_vars_and_prime(),
                                  skill_bdd & arg_deterministic_strat.get_t_env() & winning_states_internal))
            else:
                one_mod.append(
                    arg_bdd.exist(arg_deterministic_strat.get_cntr_vars_and_prime(), skill_bdd & T_removed_env))
                one_mod.append(arg_bdd.exist(arg_deterministic_strat.get_cntr_vars_and_prime(),
                                             skill_bdd & arg_deterministic_strat.get_t_env() & winning_states_internal))
            modifications.append(one_mod)

    return None, None, modifications, None


def create_expr(arg_bdd, exprs):
    t = arg_bdd.true
    for e in exprs:
        try:
            t = arg_bdd.add_expr(e) & t
        except Exception as err:
            raise err

    return t


def print_expr(bdd_internal, txt, expr, vars_internal=None, vars_ordering=None, do_print=True, do_names=False, fid=None,
               to_file=False, arg_opts=dict()):
    if to_file:
        fid = open(arg_opts['fid_base'] + "/" + txt + ".txt", "w")
    if PRINT_EXPR and (do_print or fid):
        if vars_internal is None:
            vars_internal = bdd_internal.support(expr)
        expr_txt = bdd_internal.pick_iter(expr, vars_internal)
        if vars_ordering is not None:
            if do_print:
                print("{}:".format(txt))
            if fid and txt:
                fid.write("{}:\n".format(txt))
            # for var in vars_ordering:
            #     print("{}".format(var), end=",")
            # print("")
            for ex in expr_txt:
                for var in vars_ordering:
                    if do_names:
                        if var in ex and ex[var]:
                            if do_print:
                                print(var, end=" & ")
                            if fid:
                                fid.write("{} & ".format(var))
                        elif var in ex and not ex[var]:
                            if do_print:
                                print("!" + var, end=" & ")
                            if fid:
                                fid.write("!{} & ".format(var))
                    else:
                        if var in ex and ex[var]:
                            if do_print:
                                print("1", end="")
                            if fid:
                                fid.write("1")
                        elif var in ex and not ex[var]:
                            if do_print:
                                print("0", end="")
                            if fid:
                                fid.write("0")
                        else:
                            if do_print:
                                print("-", end="")
                            if fid:
                                fid.write("-")
                if do_print:
                    print("")
                if fid:
                    fid.write("\n")
        else:
            print("{}: {}".format(txt, expr_txt))
    if to_file:
        fid.close()


def print_expr_as_env_trans(bdd_internal, txt, expr, fid=None, vars_internal=None, vars_current=None, vars_primed=None,
                            do_print=True, do_names=False):
    if PRINT_EXPR and do_print:
        if vars_internal is None:
            vars_internal = bdd_internal.support(expr)
        # find the preconditions and then the postconditions from there
        only_pres = bdd_internal.exist(vars_primed, expr)
        pre_list = list(bdd_internal.pick_iter(only_pres, vars_current))
        # expr_txt = list(bdd_internal.pick_iter(expr, vars_internal))
        if vars_current is not None and vars_primed is not None:
            print("{}:".format(txt))
            if fid and txt:
                fid.write("{}:\n".format(txt))
            # for var in vars_ordering:
            #     print("{}".format(var), end=",")
            # print("")
            for pre in pre_list:
                print("(", end='')
                if fid:
                    fid.write("(")
                var_written = False
                for jj, var in enumerate(vars_current):
                    if do_names:
                        if var in pre and pre[var]:
                            if var_written:
                                print(" & ", end="")
                                if fid:
                                    fid.write(" & ")
                            else:
                                var_written = True
                            print(var, end="")
                            if fid:
                                fid.write("{}".format(var))
                        elif var in pre and not pre[var]:
                            if var_written:
                                print(" & ", end="")
                                if fid:
                                    fid.write(" & ")
                            else:
                                var_written = True
                            print("!" + var, end="")
                            if fid:
                                fid.write("!{}".format(var))
                    else:
                        if var in pre and pre[var]:
                            print("1", end="")
                            if fid:
                                fid.write("1")
                        elif var in pre and not pre[var]:
                            print("0", end="")
                            if fid:
                                fid.write("0")
                        else:
                            print("-", end="")
                            if fid:
                                fid.write("-")
                print(")", end="")
                print(" -> ", end='')
                if fid:
                    fid.write(") -> ")

                post_expr = bdd_internal.cube(pre) & expr
                post_only = bdd_internal.exist(vars_current, post_expr)
                post_list = list(bdd_internal.pick_iter(post_only, vars_primed))
                print("(", end="")
                if fid:
                    fid.write("(")
                for pp, post in enumerate(post_list):
                    print("(", end='')
                    if fid:
                        fid.write("(")
                    var_written = False
                    for jj, var in enumerate(vars_primed):
                        if do_names:
                            if var in post and post[var]:
                                if var_written:
                                    print(" & ", end="")
                                    if fid:
                                        fid.write(" & ")
                                else:
                                    var_written = True
                                print(var, end="")
                                if fid:
                                    fid.write("{}".format(var))
                            elif var in post and not post[var]:
                                if var_written:
                                    print(" & ", end="")
                                    if fid:
                                        fid.write(" & ")
                                else:
                                    var_written = True
                                print("!" + var, end="")
                                if fid:
                                    fid.write("!{}".format(var))
                        else:
                            if var in post and post[var]:
                                print("1", end="")
                                if fid:
                                    fid.write("1")
                            elif var in post and not post[var]:
                                print("0", end="")
                                if fid:
                                    fid.write("0")
                            else:
                                print("-", end="")
                                if fid:
                                    fid.write("-")
                    print(")", end="")
                    if fid:
                        fid.write(")")
                    if pp < len(post_list) - 1:
                        print(" | ", end="")
                        if fid:
                            fid.write(" | ")

                print(")")
                if fid:
                    fid.write(")\n")

                # # Prints which variables are the same pre/post
                # print(" Same: ", end="")
                # for vc in vars_current:
                #     vp = vc + "'"
                #     if vc in ex and vp in ex and ex[vc] == ex[vp]:
                #         print("{} ".format(vc), end="")

            print("")
            # if fid:
            #     fid.write("\n")
        else:
            print("{}: {}".format(txt, expr_txt))


def print_expr_as_sys_trans(bdd_internal, txt, expr, fid=None, vars_internal=None, vars_input_primed=None,
                            vars_output_primed=None, do_print=True, do_names=False):
    if PRINT_EXPR and do_print:
        if vars_internal is None:
            vars_internal = bdd_internal.support(expr)
        expr_txt = list(bdd_internal.pick_iter(expr, vars_internal))
        if vars_input_primed is not None and vars_output_primed is not None:
            print("{}:".format(txt))
            if fid and txt:
                fid.write("{}:\n".format(txt))
            # for var in vars_ordering:
            #     print("{}".format(var), end=",")
            # print("")
            print("!(", end='')
            if fid:
                fid.write("!(")
            for ii, ex in enumerate(expr_txt):
                print("(", end="")
                if fid:
                    fid.write("(")
                var_written = False
                for jj, var in enumerate(vars_input_primed):
                    if var in ex and ex[var]:
                        if var_written:
                            print(" & ", end="")
                            if fid:
                                fid.write(" & ")
                        else:
                            var_written = True
                        print(var, end="")
                        if fid:
                            fid.write("{}".format(var))
                    elif var in ex and not ex[var]:
                        if var_written:
                            print(" & ", end="")
                            if fid:
                                fid.write(" & ")
                        else:
                            var_written = True
                        print("!" + var, end="")
                        if fid:
                            fid.write("!{}".format(var))
                print(")", end="")
                if fid:
                    fid.write(")")
                if ii < len(expr_txt) - 1:
                    print(" | ", end="")
                    if fid:
                        fid.write(" | ")
            print(") -> ", end='')
            if fid:
                fid.write(") -> ")
            for ii, ex in enumerate(expr_txt):
                if ii == 0:
                    print("!(", end="")
                    if fid:
                        fid.write("!(")
                    var_written = False
                    for jj, var in enumerate(vars_output_primed):
                        if var in ex and ex[var]:
                            if var_written:
                                print(" & ", end="")
                                if fid:
                                    fid.write(" & ")
                            else:
                                var_written = True
                            print(var, end="")
                            if fid:
                                fid.write("{}".format(var))
                        elif var in ex and not ex[var] and False:
                            if var_written:
                                print(" & ", end="")
                                if fid:
                                    fid.write(" & ")
                            else:
                                var_written = True
                            print("!" + var, end="")
                            if fid:
                                fid.write("!{}".format(var))
                    print(")", end="")
                    if fid:
                        fid.write(")\n")
            print("")
            # if fid:
            #     fid.write("\n")
        else:
            print("{}: {}".format(txt, expr_txt))


def assign(bdd_internal, nmbr, vars_internal):
    """
    Assigns the number nmbr to the variables in vars by converting to base 2
    :param bdd_internal:
    :param nmbr:
    :param vars_internal:
    :return:
    """

    digits = []
    while nmbr:
        digits.append(int(nmbr % 2))
        nmbr //= 2

    bin_str = ""
    for ii, v in enumerate(vars_internal):
        if ii < len(digits) and digits[-(ii + 1)] == 1:
            bin_str += v
        else:
            bin_str += "!" + v
        if ii < len(vars_internal) - 1:
            bin_str += " & "

    return bdd_internal.add_expr(bin_str)


def print_suggestions(arg_bdd, arg_mod_pre, arg_mod_post, arg_opts, arg_acts_changed, vars_ordering=None,
                      vars_current=None, vars_input_primed=None, vars_output_primed=None, do_names=False):
    int_acts_changed = arg_acts_changed

    if arg_opts['suggestions']:
        fid = open(arg_opts['suggestions'], 'w')

    if arg_opts['enforce_reactive_variables']:
        for rv in arg_opts['reactive_variables_current']:
            vars_current.remove(rv)

    for ii, (one_mod_pre, one_mod_post) in enumerate(zip(arg_mod_pre, arg_mod_post)):
        print("Suggestion {}".format(ii))
        if arg_opts['suggestions']:
            fid.write("Suggestion {}\n".format(ii))

        # # Print new preconditions
        # for ompre in one_mod_pre:
        #     if ompre is not None:
        #         print_expr(arg_bdd, "Action preconditions modified:", ompre[0], vars_ordering=vars_ordering,
        #                    do_names=do_names)
        #         print_expr(arg_bdd, "Preconditions added:", ompre[1], vars_ordering=vars_ordering, do_names=do_names)
        #         # print_expr(arg_bdd, "New action:", ompre[2], vars_ordering=vars_ordering, do_names=do_names)
        #         if arg_opts['suggestions']:
        #             if do_names:
        #                 # Printing the added preconditions
        #                 print_expr_as_sys_trans(arg_bdd, None, ompre[1], vars_input_primed=vars_input_primed,
        #                                         vars_output_primed=vars_output_primed, do_names=do_names, fid=fid)
        #             else:
        #                 print_expr(arg_bdd, None, ompre[1], vars_ordering=vars_ordering, do_names=do_names, fid=fid)
        #         else:
        #             print_expr_as_sys_trans(arg_bdd, "New action: ", ompre[2], vars_input_primed=vars_input_primed,
        #                                     vars_output_primed=vars_output_primed, do_names=do_names)

        # Print new postconditions
        for ompost in one_mod_post:
            if ompost is not None:
                print_expr(arg_bdd, "Action postconditions modified:", ompost[0], vars_ordering=vars_ordering,
                           do_names=do_names)
                for (new_skill, old_skill) in int_acts_changed:
                    if new_skill & ompost[0] != arg_bdd.false:
                        print_expr(arg_bdd, "Original skill", old_skill, vars_ordering=vars_ordering, do_names=do_names)
                # try:
                #     if arg_bdd.pick(ompost[1])['extra_action'] == False:
                #         print_expr(arg_bdd, "Postconditions removed:", ompost[1], vars_ordering=vars_ordering, do_names=do_names)
                # except:
                #     print_expr(arg_bdd, "Postconditions removed:", ompost[1], vars_ordering=vars_ordering,
                #                do_names=do_names)
                # print_expr(arg_bdd, "New action:", ompost[2], vars_ordering=vars_ordering, do_names=do_names)
                if arg_opts['suggestions']:
                    if do_names:
                        print_expr_as_env_trans(arg_bdd, None, ompost[2], vars_current=vars_current,
                                                vars_primed=vars_input_primed, do_names=do_names, fid=fid)
                    else:
                        print_expr(arg_bdd, None, ompost[2], vars_ordering=vars_ordering, do_names=do_names, fid=fid)
                else:
                    vars_input_primed_copy = copy.deepcopy(vars_input_primed)
                    if arg_opts['enforce_reactive_variables']:
                        for rv in arg_opts['reactive_variables']:
                            if vars_input_primed_copy.count(rv) > 0:
                                vars_input_primed_copy.remove(rv)
                    print_expr_as_env_trans(arg_bdd, "New action:", ompost[2], vars_current=vars_current,
                                            vars_primed=vars_input_primed_copy, do_names=do_names)

    if arg_opts['suggestions']:
        fid.close()


def string_escape(s, encoding='utf-8'):
    return (s.encode('latin1')  # To bytes, required by 'unicode-escape'
            .decode('unicode-escape')  # Perform the actual octal-escaping decode
            .encode('latin1')  # 1:1 mapping back to bytes
            .decode(encoding))  # Decode original encoding


def string_escape_list(s_list):
    out = []
    for s in s_list:
        out.append(string_escape(s))

    return out


def bdd_to_suggestions(arg_bdd, arg_mod_pre, arg_mod_post, arg_opts, arg_acts_changed, arg_gs, arg_T_swapped_pre,
                       arg_T_swapped_post):
    int_acts_changed = arg_acts_changed

    if arg_opts['suggestions']:
        fid = open(arg_opts['suggestions'], 'w')

    # if arg_opts['enforce_reactive_variables']:
    #     for rv in arg_opts['reactive_variables_current']:
    #         vars_current.remove(rv)

    all_suggestions = []

    for ii, one_mod in enumerate(arg_mod_post):
        print("Suggestion {}".format(ii))
        skill_pre_posts = dict()

        mod_cnt = 0
        # Print new postconditions
        for ompost in one_mod:
            # for ompost in one_mod_pre + one_mod_post:
            if ompost is not None:
                new_skill = ompost[0]
                # print_expr(arg_bdd, "Original skill", old_skill,
                #            vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_names=True)

                # Finds the intermediate states, first preconditions, final postconditions
                all_pres_for_a_post = []
                all_posts_for_a_pre = []
                final_post = []
                first_pre = []

                # For each postcondition, finds all possible preconditions that could exist to allow it to happen
                posts = arg_bdd.exist(arg_gs.get_vars(), ompost[2])
                inp_vars = copy.deepcopy(arg_gs.get_input_vars())
                inp_vars_prime = copy.deepcopy(arg_gs.get_input_vars_prime())
                if arg_opts['enforce_reactive_variables']:
                    for rv in arg_opts['reactive_variables_current']:
                        inp_vars_prime.remove(rv + "'")
                        inp_vars.remove(rv)
                pres = arg_bdd.exist(arg_gs.get_output_vars() + inp_vars_prime,
                                     ompost[2] & posts)
                init_pres = arg_bdd.let(arg_gs.get_inputprime_to_input(),
                                        arg_bdd.exist(arg_gs.get_vars_and_output_prime(),
                                                      arg_bdd.let(arg_gs.get_v_to_v_prime(), pres & new_skill) & \
                                                      arg_bdd.add_expr("!" + " & !".join(arg_opts['existing_skills'])) & \
                                                      arg_gs.get_t_sys_not_hard()))
                print_expr(arg_bdd, "new skill", new_skill, vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
                           do_print=True)
                print_expr(arg_bdd, "Init pres", init_pres, vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
                           do_print=True)

                final_posts = arg_bdd.false
                for post_dict in arg_bdd.pick_iter(posts, care_vars=inp_vars_prime):
                    post = arg_bdd.cube(post_dict)
                    if arg_bdd.let(arg_gs.get_inputprime_to_input(), post) & ompost[2] == arg_bdd.false:
                        final_posts = final_posts | post
                    if arg_bdd.let(arg_gs.get_inputprime_to_input(), post) & ompost[2] & post != arg_bdd.false:
                        final_posts = final_posts | post
                final_posts = arg_bdd.let(arg_gs.get_inputprime_to_input(), final_posts)
                # final_posts = changing_final_posts | same_final_posts
                print_expr(arg_bdd, "posts", posts, vars_ordering=arg_gs.get_vars_and_prime_and_dp(), do_print=True)
                print_expr(arg_bdd, "Final posts", final_posts, vars_ordering=arg_gs.get_vars_and_prime_and_dp(),
                           do_print=True)

                # Finds all possible postconditions for a precondition
                pres = arg_bdd.exist(arg_gs.get_output_vars() + arg_gs.get_input_vars_prime(), ompost[2])
                for pre in arg_bdd.pick_iter(pres, care_vars=inp_vars):
                    posts = arg_bdd.exist(arg_gs.get_vars(), arg_bdd.cube(pre) & ompost[2])
                    all_posts_for_a_pre.append(
                        [pre,
                         list(arg_bdd.pick_iter(arg_bdd.let(arg_gs.get_v_prime_to_v(), posts),
                                                care_vars=inp_vars))])

                pres = arg_bdd.exist(arg_gs.get_output_vars() + arg_gs.get_input_vars_prime(), ompost[2])
                posts = arg_bdd.exist(arg_gs.get_vars(), ompost[2])
                T_unique = arg_bdd.let(arg_gs.get_v_prime_to_v(), posts) | pres
                unique_states = list(
                    arg_bdd.pick_iter(T_unique, care_vars=inp_vars))

                skill_pre_posts[str(mod_cnt)] = {
                    'name': 'skill' + str(len(arg_gs.get_output_vars()) + mod_cnt),
                    'new_skill':
                        list(arg_bdd.pick_iter(new_skill, care_vars=arg_gs.get_output_vars()))[
                            0],
                    'original_skill':
                        list(arg_bdd.pick_iter(new_skill, care_vars=arg_gs.get_output_vars()))[
                            0],
                    'intermediate_states_all_pres': all_pres_for_a_post,
                    'intermediate_states': all_posts_for_a_pre,
                    'final_postconditions': list(arg_bdd.pick_iter(final_posts, care_vars=inp_vars)),
                    'initial_preconditions': list(arg_bdd.pick_iter(init_pres, care_vars=inp_vars)),
                    'unique_states': unique_states,
                    'swapped': list(arg_bdd.pick_iter(arg_bdd.false,
                                                      care_vars=arg_gs.get_vars() + arg_gs.get_input_vars_prime())),
                    'avoid_states': [],
                    'folder_train': '',
                    'folder_val': '',
                    'suggestion': True
                }
                mod_cnt += 1

                # print_expr_as_env_trans(arg_bdd, "New action:", ompost[2], vars_current=vars_current,
                #                         vars_primed=vars_input_primed, do_names=do_names)
        all_suggestions.append(skill_pre_posts)
        if ii > 0:
            break
    return all_suggestions


def run_repair(file_in, opts):
    s_time = time.time()
    bdd = _bdd.BDD()
    spec_in = Specification(file_in_internal=file_in, bdd_internal=bdd)
    if opts['suggestions']:
        os.makedirs(opts['fid_base'], exist_ok=True)
    gs = spec_in.get_game_structure()

    do_compute_winning_states = True

    # Make system reach all liveness guarantees from somewhere
    repaired_gs = copy.copy(gs)
    T_swapped_pre_all = bdd.false
    T_swapped_post_all = bdd.false
    # repaired_gs.bdd = gs.bdd
    acts_changed_ext = []
    T_previously_changed = bdd.false
    while do_compute_winning_states:
        # Compute winning states
        winning_states = compute_winning_states(repaired_gs.bdd, repaired_gs, opts)
        do_compute_winning_states = winning_states.does_need_repair()

        if opts['only_synthesis'] and do_compute_winning_states:
            # raise Exception(
            #     "The specification actually needs repair (a liveness cannot be reached) and a strategy cannot just be synthesized")
            return False, dict()

        if winning_states.does_need_repair():
            opts['cover'] = False
            repaired_gs, acts_changed, T_previously_changed, T_swapped_pre, T_swapped_post = perform_repair(
                repaired_gs.bdd, repaired_gs,
                winning_states,
                winning_states.get_target_states(),
                T_previously_changed, opts)
            # opts['post_first'] = not opts['post_first']
            acts_changed_ext.extend(acts_changed)
            T_swapped_pre_all = T_swapped_pre_all | T_swapped_pre
            T_swapped_post_all = T_swapped_post_all | T_swapped_post
            # repaired_gs.update_change_cons(acts_changed)

    print("Computed winning states in: {}".format(time.time() - s_time))
    # Make system reach livesness guarantees from initial states
    is_realizable = repaired_gs.get_t_init() & winning_states.get_z() == repaired_gs.get_t_init()
    print_expr(bdd, "init", repaired_gs.get_t_init(), vars_ordering=repaired_gs.get_vars_and_vars_prime(),
               do_print=False)
    print_expr(bdd, "winning states", winning_states.get_z(), vars_ordering=repaired_gs.get_vars_and_vars_prime(),
               do_print=False)
    if opts['only_synthesis'] and not is_realizable:
        # raise Exception(
        #     "The specification actually needs repair (init states do not overlap winning states) and a strategy cannot just be synthesized")
        return False, dict()
    elif opts['only_synthesis'] and is_realizable:
        return True, dict()
    while not is_realizable:
        opts['cover'] = True
        repaired_gs, acts_changed, T_previously_changed, T_swapped_pre, T_swapped_post = perform_repair(repaired_gs.bdd,
                                                                                                        repaired_gs,
                                                                                                        winning_states,
                                                                                                        repaired_gs.get_t_init(),
                                                                                                        T_previously_changed,
                                                                                                        opts)
        # opts['post_first'] = not opts['post_first']
        winning_states = compute_winning_states(repaired_gs.bdd, repaired_gs, opts)
        acts_changed_ext.extend(acts_changed)
        T_swapped_pre_all = T_swapped_pre_all | T_swapped_pre
        T_swapped_post_all = T_swapped_post_all | T_swapped_post
        # repaired_gs.update_change_cons(acts_changed)

        is_realizable = repaired_gs.get_t_init() & winning_states.get_z() == repaired_gs.get_t_init()
    print("Checked initial states contained in: {}".format(time.time() - s_time))

    # Synthesize a strategy
    all_mod_posts = []
    all_mod_pres = []
    suggestion_cnt = 0
    # while is_realizable and not winning_states.does_need_repair():
    #     # if suggestion_cnt > 1:
    #     #     break
    strategy = synthesize(repaired_gs.bdd, repaired_gs, winning_states)

    # Determinize
    deterministic_strat = determinize_strategy(strategy.bdd, strategy)
    print_expr(deterministic_strat.bdd, "Deterministic Strategy", deterministic_strat.get_t_sys(),
               vars_ordering=deterministic_strat.get_vars_and_vars_prime(), do_names=False, do_print=False)
    # if opts['only_synthesis']:
    #     print("Synthesized a strategy in: {}".format(time.time() - s_time))
    #     # sys.exit("Done with synthesis")
    #     return True, dict()
    T_removed, T_added, one_rep_mod_post, one_rep_mod_pre = \
        report_skill_revision(deterministic_strat.bdd, gs, deterministic_strat, opts)
    all_mod_pres.append(one_rep_mod_pre)
    all_mod_posts.append(one_rep_mod_post)

    # # Disallow previous suggestions and recompute winning states
    # # Note: only works for original algorithm, not tested for changing pre/post instead of simply removing/adding
    # repaired_gs.T_sys = repaired_gs.get_t_sys() & ~T_added
    # repaired_gs.T_env = repaired_gs.get_t_env() | T_removed

    # winning_states = compute_winning_states(repaired_gs.bdd, repaired_gs, opts)
    # is_realizable = repaired_gs.get_t_init() & winning_states.get_z() == repaired_gs.get_t_init()
    # suggestion_cnt += 1

    print_suggestions(repaired_gs.bdd, all_mod_pres, all_mod_posts, opts, acts_changed_ext,
                      vars_ordering=repaired_gs.get_vars_and_vars_prime(),
                      vars_current=repaired_gs.get_input_vars() + repaired_gs.get_output_vars(),
                      vars_input_primed=repaired_gs.get_input_vars_prime(),
                      vars_output_primed=repaired_gs.get_output_vars_prime(), do_names=opts['do_names'])
    print("This took: {}".format(time.time() - s_time))

    # plot_suggestions(repaired_gs.bdd, all_mod_pres, all_mod_posts, vars_ordering=repaired_gs.get_vars_and_vars_prime(), do_names=True)

    suggestions = bdd_to_suggestions(repaired_gs.bdd, all_mod_pres, all_mod_posts, opts, acts_changed_ext, repaired_gs,
                                     T_swapped_pre_all, T_swapped_post_all)

    return False, suggestions[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_in", help="Name of structuredslugs files to repair")
    parser.add_argument("--run_original", help="Run repair as in ICRA2020 paper", default=False, action="store_true",
                        required=False)
    parser.add_argument("--recompute_winning_states_after_one_repair",
                        help="Recompute winning states after each suggestion instead of waiting to cover/include start",
                        action="store_true", default=False, required=False)
    parser.add_argument("--only_synthesis", help="Only perform synthesis and create a strategy, do not run repair",
                        action="store_true", default=False, required=False)
    parser.add_argument("--enforce_reactive_variables",
                        help='Make sure reactive variables are not changed during repair',
                        action='store_true', default=False, required=False)
    parser.add_argument("--reactive_variables", help="Which variable(s) are reactive",
                        required=False, type=str)
    parser.add_argument("--reactive_variables_current", help="Which variable(s) are reactive",
                        required=False, type=str)
    parser.add_argument("--suggestions", help="where to save suggestions",
                        required=False, default="", type=str)
    parser.add_argument("--fid_base", help="where to save intermediate steps", required=False,
                        default="/home/adam/Documents/repair_tmp/", type=str)
    parser.add_argument("--post_first", default=False, required=False, action="store_true",
                        help="Run repair post before pre")
    parser.add_argument('--do_names', required=False, action='store_true', default=False,
                        help="write suggestions as human readable")
    parser.add_argument('--save_intermediate_steps', required=False, action='store_true', default=False,
                        help="save intermediate states of repair")
    parser.add_argument('--extra_skills', required=True, help="which skills are extra")
    parser.add_argument('--existing_skills', required=True, help="which skills are existing")
    args = parser.parse_args()
    file_in = args.file_in
    # "/home/adam/Documents/projects/learning_missing_skills/examples/four_squares/input_multistep.structuredslugs"

    opts = {'return_with_one_repair': args.recompute_winning_states_after_one_repair,
            'run_original': args.run_original,
            'only_synthesis': args.only_synthesis,
            'enforce_reactive_variables': args.enforce_reactive_variables,
            # 'reactive_variables_current': [args.reactive_variables[0]],
            # 'reactive_variables': ['blue_person', 'blue_person\'', 'green_person', 'green_person\''],
            # 'reactive_variables_current': ['blue_person', 'green_person'],
            'suggestions': args.suggestions,
            'do_names': args.do_names,
            'post_first': args.post_first,
            'post_repair_cnt': 0,
            'to_file': args.save_intermediate_steps,
            'fid_base': args.fid_base,
            'extra_skills': string_escape_list(args.extra_skills.split(" & ")),
            'existing_skills': string_escape_list(args.existing_skills.split(" & "))}
    if args.reactive_variables is not None:
        opts['reactive_variables'] = string_escape_list(args.reactive_variables.split(" & "))
        opts['reactive_variables_current'] = string_escape_list(args.reactive_variables_current.split(" & "))
    else:
        opts['reactive_variables'] = []
        opts['reactive_variables_current'] = []

    run_repair(file_in, opts)
