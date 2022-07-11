#!/usr/bin/env python

from dl2_lfd.nns.dmp_nn import DMPNN
from dl2_lfd.dmps.dmp import load_dmp_demos, DMP
import torch
from dl2_lfd.helper_funcs.conversions import np_to_pgpu

from synthesis_based_repair.tools import write_spec, clear_file, dict_to_formula, json_load_wrapper

import numpy as np

DEVICE = 'cpu'


def main():
    opts = json_load_wrapper('/home/adam/repos/synthesis_based_repair/data/stretch/stretch_dmp_opts.json')
    dmp_folder = '/home/adam/repos/synthesis_based_repair/data/dmps/'
    skill_name = 'skillStretch3to1'
    model = DMPNN(opts['start_dimension'], 1024, opts['dimension'], opts['basis_fs']).to(DEVICE)
    model.load_state_dict(torch.load(dmp_folder + skill_name + '.pt'))
    starts = np.array([[[0.5, -0.5, 0, 0, 0.7, 0], [0.5, 0.5, 3.14, 0, 0.7, 0]]])
    print(starts)
    learned_weights = model(np_to_pgpu(starts))
    dmp = DMP(opts['basis_fs'], opts['dt'], opts['dimension'])
    learned_rollouts, _, _ = dmp.rollout_torch(torch.tensor(starts[:, 0, :]).to(DEVICE), torch.tensor(starts[:, 1, :]).to(DEVICE), learned_weights)
    print(learned_rollouts)
    print("Rollout size {}".format(learned_rollouts.shape))
    return

if __name__ == "__main__":
    main()