import numpy as np
import torch

from evaluation_protocol.helpers import EvalWrapper


class ZeroAgentWrapper(EvalWrapper):
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
        return self.model.get_actions(*args, **kwargs)


class ZeroAgent:
    def __init__(self, action_space):
        self.action_space = action_space 

    def eval(self, *args, **kwargs):
        return ZeroAgentWrapper(self)

    def to(self, device):
        pass

    def get_actions(self, *args, batch_size=1, **kwargs):
        return np.zeros((batch_size, self.action_space)), np.zeros((batch_size, self.action_space))
    