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
        return np.zeros((1, self.action_space)), np.zeros((1, 1))
    
    def get_batch_actions(self, states, *args, **kwargs):
        batch_size = states.shape[0]
        return np.zeros((batch_size, self.action_space)), np.zeros((batch_size, 1))
    
    def to(self, device):
        pass


class UniformAgentWrapper(EvalWrapper):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def eval(self, *args, **kwargs):
        return self
    
    def get_action(self, *args, **kwargs):
        return self.model.get_action(*args, **kwargs)
    
    def get_batch_actions(self, *args, batch_size=1, **kwargs):
        return self.model.get_batch_actions(*args, batch_size=batch_size, **kwargs)


class UniformAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def eval(self, *args, **kwargs):
        return UniformAgentWrapper(self)

    def get_action(self, pr_action, *args, **kwargs):
        return pr_action, np.random.choice(2).reshape(1, self.action_space), np.random.choice(2).reshape(1, 1)
    
    def get_batch_actions(self, states, pr_actions, *args, **kwargs):
        batch_size = states.shape[0]
        random_mask = np.random.choice([True, False], batch_size)
        random_mask = random_mask[:, np.newaxis]
        adv_actions = np.where(random_mask, np.array([0]), np.array([1])).reshape(batch_size, 1)
        return pr_actions, adv_actions
    
    def to(self, device):
        pass


class WorstCaseAgentWrapper(EvalWrapper):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def eval(self, *args, **kwargs):
        return self
    
    def get_action(self, *args, **kwargs):
        return self.model.get_action(*args, **kwargs)
    
    def get_batch_actions(self, *args, batch_size=1, **kwargs):
        return self.model.get_batch_actions(*args, batch_size=batch_size, **kwargs)


class WorstCaseAgent:
    def __init__(self, action_space, version):
        self.action_space = action_space
        self.version = version

    def eval(self, *args, **kwargs):
        return WorstCaseAgentWrapper(self)

    def get_action(self, pr_action, *args, **kwargs):
        if self.version == "v1":
            if np.all(pr_action == np.array([0])):
                return np.array([0])
            elif np.all(pr_action == np.array([1])):
                return np.array([1])
            else:
                raise ValueError(f"Invalid pr_action: {pr_action}")
        elif self.version == "v2":
            if np.all(pr_action == np.array([0, 0])):
                return np.array([1])
            elif np.all(pr_action == np.array([0, 1])):
                return np.array([0])
            elif np.all(pr_action == np.array([1, 0])):
                return np.array(np.random.choice([0, 1]))
            elif np.all(pr_action == np.array([1, 1])):
                return np.array(np.random.choice([0, 1]))
            else:
                raise ValueError(f"Invalid pr_action: {pr_action}")
    
    def get_batch_actions(self, states, pr_actions, *args, **kwargs):
        batch_size = states.shape[0]
        adv_actions = np.zeros((batch_size, 1))
        for i, pr_action in enumerate(pr_actions):
            pr_action = pr_action.reshape(1, self.action_space)
            adv_actions[i] = self.get_action(pr_action)
        return pr_actions.reshape(batch_size, self.action_space), adv_actions.reshape(batch_size, 1)

    def to(self, device):
        pass
