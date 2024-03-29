import gymnasium as gym
import numpy as np

from collections import defaultdict
from datasets import Dataset


class OneStepEnvVOne(gym.Env):
    def __init__(self):
        super().__init__()

    def step(self, action):
        if np.all(action == np.array([0, 0])):
            return np.array([1]), 0.5, True, False, {}
        elif np.all(action == np.array([0, 1])):
            return np.array([2]), 2, True, False, {}
        elif np.all(action == np.array([1, 0])):
            return np.array([3]), 2, True, False, {}
        elif np.all(action == np.array([1, 1])):
            return np.array([4]), 1.5, True, False, {}

    def reset(self, seed=0):
        super().reset(seed=seed)
        return np.array([0]), {}
    
    @classmethod
    def get_returns_for_action(cls, action):
        if np.all(action == np.array([0, 0])):
            return 0.5
        elif np.all(action == np.array([0, 1])):
            return 2
        elif np.all(action == np.array([1, 0])):
            return 2
        elif np.all(action == np.array([1, 1])):
            return 1.5
    
    @classmethod
    def get_wc_returns_for_pr_action(cls, action):
        if np.all(action == np.array([0, 0])):
            return 0.5
        elif np.all(action == np.array([0, 1])):
            return 0.5
        elif np.all(action == np.array([1, 0])):
            return 1.5
        elif np.all(action == np.array([1, 1])):
            return 1.5

    @classmethod
    def get_correct_pr_action(cls, target):
        if target == 0.5:
            return [([0], 1.0)]
        elif target == 2.0:
            return [([0], 0.5), ([1], 0.5)]
        elif target == 1.5:
            return [([1], 1.0)]
        else:
            raise ValueError(f"Invalid target: {target}")
        
    @classmethod
    def get_best_pr_action(cls, target):
        if target == 0.5:
            return [([0], 1.0)]
        elif target == 2.0:
            return [([1], 1.0)]
        elif target == 1.5:
            return [([1], 1.0)]
        else:
            raise ValueError(f"Invalid target: {target}")
        
    @classmethod
    def get_all_possible_pr_actions(cls):
        return [[0], [1]]
    
    @classmethod
    def get_eval_targets(cls):
        return [0.5, 2.0, 1.5]


def create_onestep_vone_toy_dataset(n_trajs=1000):
    traj_per_type = n_trajs // 4
    traj_one = [[[0], 0, False, [0], [0]], [[1], 0.5, True, [0], [0]]]
    traj_two = [[[0], 0, False, [0], [1]], [[2], 2, True, [0], [1]]]
    traj_three = [[[0], 0, False, [1], [0]], [[3], 2, True, [1], [0]]]
    traj_four = [[[0], 0, False, [1], [1]], [[4], 1.5, True, [1], [1]]]
    trajs_dict = defaultdict(list)
    for _ in range(traj_per_type):
        #
        trajs_dict['observations'].append([traj_one[0][0], traj_one[1][0]])
        trajs_dict['rewards'].append([traj_one[0][1], traj_one[1][1]])
        trajs_dict['dones'].append([traj_one[0][2], traj_one[1][2]])
        trajs_dict['pr_actions'].append([traj_one[0][3], traj_one[1][3]])
        trajs_dict['adv_actions'].append([traj_one[0][4], traj_one[1][4]])
        #
        trajs_dict['observations'].append([traj_two[0][0], traj_two[1][0]])
        trajs_dict['rewards'].append([traj_two[0][1], traj_two[1][1]])
        trajs_dict['dones'].append([traj_two[0][2], traj_two[1][2]])
        trajs_dict['pr_actions'].append([traj_two[0][3], traj_two[1][3]])
        trajs_dict['adv_actions'].append([traj_two[0][4], traj_two[1][4]])
        #
        trajs_dict['observations'].append([traj_three[0][0], traj_three[1][0]])
        trajs_dict['rewards'].append([traj_three[0][1], traj_three[1][1]])
        trajs_dict['dones'].append([traj_three[0][2], traj_three[1][2]])
        trajs_dict['pr_actions'].append([traj_three[0][3], traj_three[1][3]])
        trajs_dict['adv_actions'].append([traj_three[0][4], traj_three[1][4]])
        #
        trajs_dict['observations'].append([traj_four[0][0], traj_four[1][0]])
        trajs_dict['rewards'].append([traj_four[0][1], traj_four[1][1]])
        trajs_dict['dones'].append([traj_four[0][2], traj_four[1][2]])
        trajs_dict['pr_actions'].append([traj_four[0][3], traj_four[1][3]])
        trajs_dict['adv_actions'].append([traj_four[0][4], traj_four[1][4]])

    dataset = Dataset.from_dict(trajs_dict)
    return dataset
