#!/usr/bin/env python

from dl2_lfd.ltl_diff.constraints import fully_global_ins
# import dl2_lfd.ltl_diff.ltldiff as ltd
import synthesis_based_repair.additional_ltldiff as ltd
import torch
import numpy as np


class AutomaticSkill:
    def __init__(self, symbols, suggestions_intermediate_all_pres, suggestions_intermediate_all_posts, suggestion_post, suggestion_unique, hard_constraints,
                 workspace_bnds, epsilon, opts):
        self.symbols = symbols
        self.suggestions_intermediate_all_pres = suggestions_intermediate_all_pres
        self.suggestions_intermediate_all_posts = suggestions_intermediate_all_posts
        self.suggestion_post = suggestion_post
        self.suggestion_unique = suggestion_unique
        self.hard_constraints = hard_constraints
        self.epsilon = epsilon
        self.workspace_bnds = workspace_bnds
        self.opts = opts

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)

        # Loop through all 4 corners of the center plane of the cube
        offset = 0.0
        corners = [[offset, offset]]
        # corners = [[offset, offset], [offset, -offset], [-offset, offset], [-offset, -offset]]
        for corner_x, corner_y in corners:
            rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]
            rollout_traj_corner = rollout_traj
            rollout_traj_corner[:, 0] += corner_x
            rollout_traj_corner[:, 1] += corner_y
            rollout_term = ltd.TermDynamic(rollout_traj_corner)
            sym_ltd = dict()
            for sym_name, sym in self.symbols.items():
                bnds_list = []
                if sym.get_type() == 'rectangle':
                    for dim in sym.get_dims():
                        bnds_list.append(ltd.GEQ2(rollout_term, ltd.TermStatic(sym.bounds[[dim], 0]), dim=np.array([dim])))
                        bnds_list.append(ltd.LEQ2(rollout_term, ltd.TermStatic(sym.bounds[[dim], 1]), dim=np.array([dim])))
                if sym.get_type() == 'circle':
                    bnds_list.append(ltd.LT2(
                        ltd.TermDynamic(torch.norm(rollout_term.xs - sym.get_center(), dim=2, keepdim=True)),
                                       ltd.TermStatic(sym.get_radius()), dim=np.array([0])))
                if sym.get_type() == "rectangle-ee":
                    for dim in sym.get_dims():
                        l_wrist = rollout_term.xs[:, :, 3]
                        t_robot = rollout_term.xs[:, :, 2]
                        t_wrist = rollout_term.xs[:, :, 5]
                        x_robot = rollout_term.xs[:, :, 0]
                        y_robot = rollout_term.xs[:, :, 1]
                        l_ee = 0.1
                        # dim == 0 -> x
                        x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                        y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                        pos_ee = ltd.TermDynamic(torch.stack([x_ee, y_ee], dim=2))
                        bnds_list.append(ltd.GEQ2(pos_ee, ltd.TermStatic(sym.bounds[[dim], 0]), dim=np.array([dim])))
                        bnds_list.append(ltd.LEQ2(pos_ee, ltd.TermStatic(sym.bounds[[dim], 1]), dim=np.array([dim])))
                if sym.get_type() == "circle-ee":
                    l_wrist = rollout_term.xs[:, :, 3]
                    t_robot = rollout_term.xs[:, :, 2]
                    t_wrist = rollout_term.xs[:, :, 5]
                    x_robot = rollout_term.xs[:, :, 0]
                    y_robot = rollout_term.xs[:, :, 1]
                    l_ee = 0.1
                    # dim == 0 -> x
                    x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                    y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                    pos_ee = torch.stack([x_ee, y_ee], dim=2)
                    bnds_list.append(ltd.LT2(
                        ltd.TermDynamic(torch.norm(pos_ee - sym.get_center(), dim=2, keepdim=True)),
                        ltd.TermStatic(sym.get_radius()), dim=np.array([0])))
                sym_ltd[sym_name] = bnds_list
            neg_sym_ltd = dict()
            for sym_name, sym in self.symbols.items():
                bnds_list = []
                if sym.get_type() == 'rectangle':
                    for dim in sym.get_dims():
                        bnds_list.append(ltd.LT2(rollout_term, ltd.TermStatic(sym.bounds[[dim], 0]), dim=np.array([dim])))
                        bnds_list.append(ltd.GT2(rollout_term, ltd.TermStatic(sym.bounds[[dim], 1]), dim=np.array([dim])))
                # neg_sym_ltd[sym_name] = ltd.Or(bnds_list)
                if sym.get_type() == 'circle':
                    bnds_list.append(ltd.GEQ2(
                        ltd.TermDynamic(torch.norm(rollout_term.xs - sym.get_center(), dim=2, keepdim=True)),
                        ltd.TermStatic(sym.get_radius()), dim=np.array([0])))

                if sym.get_type() == "rectangle-ee":
                    for dim in sym.get_dims():
                        l_wrist = rollout_term.xs[:, :, 3]
                        t_robot = rollout_term.xs[:, :, 2]
                        t_wrist = rollout_term.xs[:, :, 5]
                        x_robot = rollout_term.xs[:, :, 0]
                        y_robot = rollout_term.xs[:, :, 1]
                        l_ee = 0.1
                        # dim == 0 -> x
                        x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                        y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                        pos_ee = ltd.TermDynamic(torch.stack([x_ee, y_ee], dim=2))
                        bnds_list.append(ltd.LT2(pos_ee, ltd.TermStatic(sym.bounds[[dim], 0]), dim=np.array([dim])))
                        bnds_list.append(ltd.GT2(pos_ee, ltd.TermStatic(sym.bounds[[dim], 1]), dim=np.array([dim])))
                if sym.get_type() == "circle-ee":
                    l_wrist = rollout_term.xs[:, :, 3]
                    t_robot = rollout_term.xs[:, :, 2]
                    t_wrist = rollout_term.xs[:, :, 5]
                    x_robot = rollout_term.xs[:, :, 0]
                    y_robot = rollout_term.xs[:, :, 1]
                    l_ee = 0.1
                    # dim == 0 -> x
                    x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                    y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                    pos_ee = torch.stack([x_ee, y_ee], dim=2)
                    bnds_list.append(ltd.GEQ2(
                        ltd.TermDynamic(torch.norm(pos_ee - sym.get_center(), dim=2, keepdim=True)),
                        ltd.TermStatic(sym.get_radius()), dim=np.array([0])))
                neg_sym_ltd[sym_name] = bnds_list

            # A postcondition or the same precondition will always come after a given precondition
            # pre -> N (pre | posts) is the same as !pre | N (pre | posts)
            implication_next_list = []
            for suggestion_intermediate in self.suggestions_intermediate_all_posts:
                # A post will come after the pres
                pre_list = []
                for p, val in suggestion_intermediate[0].items():
                    if val:
                        pre_list.extend([ltd.And(sym_ltd[p])])
                    else:
                        pre_list.extend([ltd.Or(neg_sym_ltd[p])])
                pre_ltd = ltd.Next(ltd.And(pre_list))

                neg_pre_list = []
                for p, val in suggestion_intermediate[0].items():
                    if val:
                        neg_pre_list.extend([ltd.Or(neg_sym_ltd[p])])
                    else:
                        neg_pre_list.extend([ltd.And(sym_ltd[p])])
                neg_pre_ltd = ltd.Or(neg_pre_list)

                all_posts_list = []
                for post in suggestion_intermediate[1]:
                    post_list = []
                    for p, val in post.items():
                        if val:
                            post_list.extend([ltd.And(sym_ltd[p])])
                        else:
                            post_list.extend([ltd.Or(neg_sym_ltd[p])])
                    all_posts_list.append(ltd.Next(ltd.And(post_list)))
                if len(all_posts_list) == 1:
                    all_posts_ltd = all_posts_list[0]
                else:
                    all_posts_ltd = ltd.Or(all_posts_list)

                # implication_next_ltd = ltd.Always(ltd.Or([neg_pre_ltd, ltd.Next(ltd.Or(all_posts_list + [pre_ltd]))]), rollout_traj.shape[1]-1)
                # implication_next_ltd = ltd.Always(ltd.Or([neg_pre_ltd, ltd.Or(all_posts_list + [pre_ltd])]), rollout_traj.shape[1]-1)
                implication_next_ltd = ltd.Always(ltd.Or([neg_pre_ltd] + all_posts_list + [pre_ltd]), rollout_traj.shape[1]-1)
                implication_next_list.extend([implication_next_ltd])

        # Always stay in states in the specification
        always_list = []
        for state in self.suggestion_unique:
            state_list = []
            for s, val in state.items():
                if val:
                    state_list.extend([ltd.And(sym_ltd[s])])
                else:
                    state_list.extend([ltd.Or(neg_sym_ltd[s])])
            always_list.append(ltd.And(state_list))

        if len(always_list) == 1:
            always_ltd = ltd.Always(always_list[0], rollout_traj.shape[1])
        else:
            always_ltd = ltd.Always(ltd.Or(always_list), rollout_traj.shape[1])



        final_list = []
        if "implication_next" in self.opts:
            final_list.extend(implication_next_list)
        if "always" in self.opts:
            final_list.append(always_ltd)
        # Stretch LTD constraints:
        # The stretch dmps are in configuration/joint space instead of cartesian space.
        # We therefore need to constrain the end effector cartesian coordinate based on the joint angles
        stretch_limits_list = []
        if "stretch" in self.opts:
            final_list.append(ltd.Always(ltd.GEQ2(rollout_term, ltd.TermStatic(self.workspace_bnds[[2, 3, 4, 5], 0]), dim=np.array([2, 3, 4, 5])), rollout_traj.shape[1]))
            final_list.append(ltd.Always(ltd.LEQ2(rollout_term, ltd.TermStatic(self.workspace_bnds[[2, 3, 4, 5], 1]), dim=np.array([2, 3, 4, 5])), rollout_traj.shape[1]))

        if len(final_list) == 1:
            final_ltd = final_list[0]
        else:
            final_ltd = ltd.And(final_list)

        return final_ltd

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class AutomaticIntermediateSteps:
    def __init__(self, symbols, suggestion_intermediate, suggestion_unique, hard_constraints, workspace_bnds, epsilon):
        self.symbols = symbols
        self.suggestion_intermediate = suggestion_intermediate
        self.suggestion_unique = suggestion_unique
        self.hard_constraints = hard_constraints
        self.epsilon = epsilon
        self.workspace_bnds = workspace_bnds

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]
        rollout_term = ltd.TermDynamic(rollout_traj)

        sym_ltd = dict()
        for sym_name, sym in self.symbols.items():
            bnds_list = []
            if sym.get_type() == 'rectangle':
                for dim in sym.get_dims():
                    bnds_list.append(ltd.GEQ2(rollout_term, ltd.TermStatic(sym.bounds[[dim], 0]), dim=np.array([dim])))
                    bnds_list.append(ltd.LEQ2(rollout_term, ltd.TermStatic(sym.bounds[[dim], 1]), dim=np.array([dim])))
            if sym.get_type() == 'circle':
                bnds_list.append(ltd.LT2(
                    ltd.TermDynamic(torch.norm(rollout_term.xs - sym.get_center(), dim=2, keepdim=True)),
                    ltd.TermStatic(sym.get_radius()), dim=np.array([0])))
            if sym.get_type() == "rectangle-ee":
                for dim in sym.get_dims():
                    l_wrist = rollout_term.xs[:, :, 3]
                    t_robot = rollout_term.xs[:, :, 2]
                    t_wrist = rollout_term.xs[:, :, 5]
                    x_robot = rollout_term.xs[:, :, 0]
                    y_robot = rollout_term.xs[:, :, 1]
                    l_ee = 0.1
                    # dim == 0 -> x
                    x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                    y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                    pos_ee = ltd.TermDynamic(torch.stack([x_ee, y_ee], dim=2))
                    bnds_list.append(ltd.GEQ2(pos_ee, ltd.TermStatic(sym.bounds[[dim], 0]), dim=np.array([dim])))
                    bnds_list.append(ltd.LEQ2(pos_ee, ltd.TermStatic(sym.bounds[[dim], 1]), dim=np.array([dim])))
            if sym.get_type() == "circle-ee":
                l_wrist = rollout_term.xs[:, :, 3]
                t_robot = rollout_term.xs[:, :, 2]
                t_wrist = rollout_term.xs[:, :, 5]
                x_robot = rollout_term.xs[:, :, 0]
                y_robot = rollout_term.xs[:, :, 1]
                l_ee = 0.1
                # dim == 0 -> x
                x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                pos_ee = torch.stack([x_ee, y_ee], dim=2)
                bnds_list.append(ltd.LT2(
                    ltd.TermDynamic(torch.norm(pos_ee - sym.get_center(), dim=2, keepdim=True)),
                    ltd.TermStatic(sym.get_radius()), dim=np.array([0])))
            sym_ltd[sym_name] = ltd.And(bnds_list)
        neg_sym_ltd = dict()
        for sym_name, sym in self.symbols.items():
            bnds_list = []
            if sym.get_type() == 'rectangle':
                for dim in sym.get_dims():
                    multiplier = 1
                    bnds_list.append(ltd.LT2(rollout_term, ltd.TermStatic(sym.bounds[[dim], 0]), dim=np.array([dim])))
                    bnds_list.append(ltd.GT2(rollout_term, ltd.TermStatic(sym.bounds[[dim], 1]), dim=np.array([dim])))
            if sym.get_type() == 'circle':
                bnds_list.append(ltd.GEQ2(
                    ltd.TermDynamic(torch.norm(rollout_term.xs - sym.get_center(), dim=2, keepdim=True)),
                    ltd.TermStatic(sym.get_radius()), dim=np.array([0])))
            if sym.get_type() == "rectangle-ee":
                for dim in sym.get_dims():
                    l_wrist = rollout_term.xs[:, :, 3]
                    t_robot = rollout_term.xs[:, :, 2]
                    t_wrist = rollout_term.xs[:, :, 5]
                    x_robot = rollout_term.xs[:, :, 0]
                    y_robot = rollout_term.xs[:, :, 1]
                    l_ee = 0.1
                    # dim == 0 -> x
                    x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                    y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                    pos_ee = ltd.TermDynamic(torch.stack([x_ee, y_ee], dim=2))
                    bnds_list.append(ltd.LT2(pos_ee, ltd.TermStatic(sym.bounds[[dim], 0]), dim=np.array([dim])))
                    bnds_list.append(ltd.GT2(pos_ee, ltd.TermStatic(sym.bounds[[dim], 1]), dim=np.array([dim])))
            if sym.get_type() == "circle-ee":
                l_wrist = rollout_term.xs[:, :, 3]
                t_robot = rollout_term.xs[:, :, 2]
                t_wrist = rollout_term.xs[:, :, 5]
                x_robot = rollout_term.xs[:, :, 0]
                y_robot = rollout_term.xs[:, :, 1]
                l_ee = 0.1
                # dim == 0 -> x
                x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                pos_ee = torch.stack([x_ee, y_ee], dim=2)
                bnds_list.append(ltd.GEQ2(
                    ltd.TermDynamic(torch.norm(pos_ee - sym.get_center(), dim=2, keepdim=True)),
                    ltd.TermStatic(sym.get_radius()), dim=np.array([0])))
            neg_sym_ltd[sym_name] = ltd.Or(bnds_list)

        # A post will come after the pres
        pre_list = []
        for p, val in self.suggestion_intermediate[0].items():
            if val:
                pre_list.append(sym_ltd[p])
            else:
                pre_list.append(neg_sym_ltd[p])
        pre_ltd = ltd.And(pre_list)

        # A post will come after the pres
        neg_pre_list = []
        for p, val in self.suggestion_intermediate[0].items():
            if val:
                neg_pre_list.append(neg_sym_ltd[p])
            else:
                neg_pre_list.append(sym_ltd[p])
        neg_pre_ltd = ltd.Or(neg_pre_list)

        all_posts_list = []
        for post in self.suggestion_intermediate[1]:
            post_list = []
            for p, val in post.items():
                if val:
                    post_list.append(sym_ltd[p])
                else:
                    post_list.append(neg_sym_ltd[p])
            all_posts_list.append(ltd.And(post_list))
        if len(all_posts_list) == 1:
            all_posts_ltd = all_posts_list[0]
        else:
            all_posts_ltd = ltd.Or(all_posts_list)

        until_ltd = ltd.Until1(pre_ltd, all_posts_ltd, rollout_traj.shape[1])
        implication_ltd = ltd.Always(ltd.Or([neg_pre_ltd, until_ltd]), rollout_traj.shape[1])
        eventually_post_ltd = ltd.Eventually(all_posts_ltd, rollout_traj.shape[1])
        eventually_pre_ltd = ltd.Eventually(pre_ltd, rollout_traj.shape[1])
        final_ltd = ltd.And([implication_ltd, eventually_post_ltd, eventually_pre_ltd])

        return implication_ltd

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class States:
    def __init__(self, symbols, suggestions_pre, epsilon, buffer=0.2):
        self.symbols = symbols
        self.suggestions_pre = suggestions_pre
        self.epsilon = epsilon
        self.buffer = buffer

    def condition(self, point):
        sym_ltd = dict()
        for sym_name, sym in self.symbols.items():
            bnds_dim_list = []
            if sym.get_type() == 'rectangle':
                for dim in sym.get_dims():
                    bnds_dim_list.append(ltd.And([
                        ltd.GEQ2(point, ltd.TermStatic((torch.from_numpy(sym.bounds[:, 0]) + self.buffer)[[dim]]),
                                      dim=np.array([dim])),
                        ltd.LEQ2(point, ltd.TermStatic((torch.from_numpy(sym.bounds[:, 1]) - self.buffer)[[dim]]),
                                      dim=np.array([dim]))]))
            if sym.get_type() == 'circle':
                bnds_dim_list.append(
                    ltd.LT2(
                        ltd.TermStatic(torch.norm(point.x - torch.from_numpy(sym.get_center()), dim=1, keepdim=True)),
                        ltd.TermStatic(torch.from_numpy(sym.get_radius() - self.buffer)), dim=[0])
                )
            if sym.get_type() == "rectangle-ee":
                for dim in sym.get_dims():
                    l_wrist = point.x[0, 3]
                    t_robot = point.x[0, 2]
                    t_wrist = point.x[0, 5]
                    x_robot = point.x[0, 0]
                    y_robot = point.x[0, 1]
                    l_ee = 0.1
                    # dim == 0 -> x
                    x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                    y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                    pos_ee = ltd.TermStatic(torch.stack([x_ee, y_ee])[None, :])
                    bnds_dim_list.append(ltd.GEQ2(pos_ee, ltd.TermStatic(torch.from_numpy(sym.bounds[[dim], 0] + self.buffer)), dim=np.array([dim])))
                    bnds_dim_list.append(ltd.LEQ2(pos_ee, ltd.TermStatic(torch.from_numpy(sym.bounds[[dim], 1] - self.buffer)), dim=np.array([dim])))
            if sym.get_type() == "circle-ee":
                l_wrist = point.x[0, 3]
                t_robot = point.x[0, 2]
                t_wrist = point.x[0, 5]
                x_robot = point.x[0, 0]
                y_robot = point.x[0, 1]
                l_ee = 0.1
                # dim == 0 -> x
                x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                pos_ee = torch.stack([x_ee, y_ee])
                bnds_dim_list.append(ltd.LT2(
                    ltd.TermStatic(torch.norm(pos_ee - torch.from_numpy(sym.get_center()), dim=2, keepdim=True)),
                    ltd.TermStatic(torch.from_numpy(sym.get_radius() - self.buffer)), dim=np.array([0])))
            sym_ltd[sym_name] = ltd.And(bnds_dim_list)
            # sym_ltd[sym_name] = ltd.And([ltd.GEQ2(point, ltd.TermStatic(torch.from_numpy(sym['bnds'][:, 0]) + self.buffer * (torch.from_numpy(sym['bnds'][:, 1]) - torch.from_numpy(sym['bnds'][:, 0]))), dim=sym['dims']),
            #                              ltd.LEQ2(point, ltd.TermStatic(torch.from_numpy(sym['bnds'][:, 1]) - self.buffer * (torch.from_numpy(sym['bnds'][:, 1]) - torch.from_numpy(sym['bnds'][:, 0]))), dim=sym['dims'])])

        all_pres_list = []
        for pre in self.suggestions_pre:
            pre_list = []
            for p, val in pre.items():
                if val:
                    pre_list.append(sym_ltd[p])
                else:
                    pre_list.append(ltd.Negate(sym_ltd[p]))
            all_pres_list.append(ltd.And(pre_list))
        return ltd.Or(all_pres_list)

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)
