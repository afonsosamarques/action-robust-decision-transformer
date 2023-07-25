import numpy as np
import torch

from evaluation_protocol.helpers import EvalWrapper


class RandomAgentWrapper(EvalWrapper):
    def __init__(self, model, *args, **kwargs):
            super().__init__(model)


class RandomAgent:
    def __init__(self, action_space, is_adv=False):
        self.sample_pr_dir = lambda: torch.tensor(np.random.choice([-1, 1], size=action_space))
        self.sample_pr_magn = lambda: torch.distributions.Uniform(0, 1).sample((action_space,))
        self.sample_adv_act = lambda: torch.distributions.Normal(0, 0.15).sample((action_space,))

    def eval(self, *args, **kwargs):
        return RandomAgentWrapper(self)
    
    def to(self, device):
        pass

    def get_action(self, *args, **kwargs):
        return self.sample_pr_dir() * self.sample_pr_magn(), self.sample_adv_act()
