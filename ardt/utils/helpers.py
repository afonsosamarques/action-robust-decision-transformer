import os
import subprocess
import random

import numpy as np
import torch


def set_seed_everywhere(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    if env is not None:
        env.seed = seed
        env.action_space.seed = seed


def find_root_dir():
    try:
        root_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
    except Exception as e:
        root_dir = os.getcwd()[:os.getcwd().find('action-robust-decision-transformer')+len('action-robust-decision-transformer')]
    return root_dir + ('' if root_dir.endswith('action-robust-decision-transformer') else '/action-robust-decision-transformer') + "/ardt"

