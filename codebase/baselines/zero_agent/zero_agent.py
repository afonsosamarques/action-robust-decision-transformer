import numpy as np
import torch

from evaluation_protocol.helpers import EvalWrapper


class ZeroAgentWrapper(EvalWrapper):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def eval(self, *args, **kwargs):
        return self
    
    def get_action(self, *args, **kwargs):
        return self.model.get_action(*args, **kwargs)
    
    def get_batch_actions(self, *args, batch_size=1, **kwargs):
        return self.model.get_batch_actions(*args, batch_size=batch_size, **kwargs)


class ZeroAgent:
    def __init__(self, action_space):
        self.action_space = action_space 

    def eval(self, *args, **kwargs):
        return ZeroAgentWrapper(self)

    def get_action(self, *args, **kwargs):
        return np.zeros((1, self.action_space)), np.zeros((1, self.action_space))
    
    def get_batch_actions(self, states, *args, **kwargs):
        batch_size = states.shape[0]
        return np.zeros((batch_size, self.action_space)), np.zeros((batch_size, self.action_space))
    
    def to(self, device):
        pass
    