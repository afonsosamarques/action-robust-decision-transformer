import numpy as np
import torch

from evaluation_protocol.helpers import EvalWrapper


class RandomAgentWrapper(EvalWrapper):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def eval(self, *args, **kwargs):
        # necessary to simplify evaluation code
        return self

    def to(self, device=torch.device('cpu')):
        # necessary to simplify evaluation code
        pass

    def get_action(self, *args, **kwargs):
        return self.model.get_action(*args, **kwargs)
    
    def get_batch_actions(self, *args, batch_size=1, **kwargs):
        return self.model.get_batch_actions(*args, batch_size=batch_size, **kwargs)


class RandomAgent:
    def __init__(self, action_space, is_adv=False):
        self.sample_pr_dir = lambda batch_size: torch.tensor(np.random.choice([-1, 1], size=(batch_size, action_space)))
        self.sample_pr_magn = lambda batch_size: torch.distributions.Uniform(0, 1).sample((batch_size, action_space,))
        self.sample_adv_act = lambda batch_size: torch.distributions.Normal(0, 0.2).sample((batch_size, action_space,))

    def eval(self, *args, **kwargs):
        return RandomAgentWrapper(self)
    
    def to(self, device):
        pass

    def get_action(self, *args, **kwargs):
        return (self.sample_pr_dir(1) * self.sample_pr_magn(1)).detach().cpu().numpy(), self.sample_adv_act(1).detach().cpu().numpy()
    
    def get_batch_actions(self, states, *args, batch_size=1, **kwargs):
        batch_size = states.shape[0]
        return (self.sample_pr_dir(batch_size) * self.sample_pr_magn(batch_size)).detach().cpu().numpy(), self.sample_adv_act(batch_size).detach().cpu().numpy()
