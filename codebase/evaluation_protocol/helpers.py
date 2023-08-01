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
    return root_dir + ('' if root_dir.endswith('action-robust-decision-transformer') else '/action-robust-decision-transformer') + "/codebase/evaluation_protocol"


def scrappy_print_eval_dict(model_name, eval_dict, other_model_name=None):
    print(f"\n********** {model_name}{(f'/{other_model_name} ' if other_model_name is not None else ' ')}**********")
    print(f"Initial target returns | Avg: {np.round(np.mean(eval_dict['init_target_return']), 4)} | Std: {np.round(np.std(eval_dict['init_target_return']), 4)} | Min: {np.round(np.min(eval_dict['init_target_return']), 4)} | Median: {np.round(np.median(eval_dict['init_target_return']), 4)} | Max: {np.round(np.max(eval_dict['init_target_return']), 4)}")
    print(f"Episode lengths | Avg: {np.round(np.mean(eval_dict['ep_length']), 4)} | Std: {np.round(np.std(eval_dict['ep_length']), 4)} | Min: {np.round(np.min(eval_dict['ep_length']), 4)} | Median: {np.round(np.median(eval_dict['ep_length']), 4)} | Max: {np.round(np.max(eval_dict['ep_length']), 4)}")
    print(f"Episode returns | Avg: {np.round(np.mean(eval_dict['ep_return']), 4)} | Std: {np.round(np.std(eval_dict['ep_return']), 4)} | Min: {np.round(np.min(eval_dict['ep_return']), 4)} | Median: {np.round(np.median(eval_dict['ep_return']), 4)} | Max: {np.round(np.max(eval_dict['ep_return']), 4)}")


class EvalWrapper:
    def __init__(self, model):
        self.model = model
    
    def new_eval(self, *args, **kwargs):
        pass

    def new_batch_eval(self, *args, **kwargs):
        pass

    def get_actions(self, *args, **kwargs):
        pass
    
    def get_batch_actions(self,  *args, **kwargs):
        pass
    
    def update_history(self, *args, **kwargs):
        pass

    def update_batch_history(self, *args, **kwargs):
        pass

    def to(self, device=torch.device('cpu')):
        pass
