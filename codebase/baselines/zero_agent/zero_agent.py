import numpy as np
import torch

from evaluation_protocol.helpers import EvalWrapper


class ZeroAgentWrapper(EvalWrapper):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def eval(self, *args, **kwargs):
        return self


class ZeroAgent:
    def __init__(self, action_space):
        self.action_space = action_space 

    def eval(self, *args, **kwargs):
        return ZeroAgentWrapper(self)

    def get_actions(self, *args, batch_size=1, **kwargs):
        return np.zeros((batch_size, self.action_space)), np.zeros((batch_size, self.action_space))
    
    def to(self, device):
        pass
    