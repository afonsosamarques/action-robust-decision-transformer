import torch

from evaluation_protocol.eval_utils import EvalWrapper


class SBEvalWrapper(EvalWrapper):
    def __init__(self, model, **kwargs):
        super().__init__(model)

    def eval(self, *args, **kwargs):
        # necessary to simplify evaluation code
        return self

    def to(self, device=torch.device('cpu')):
        # necessary to simplify evaluation code
        pass
    
    def get_action(self, state):
        action = self.model.predict(state)[0]
        return action, action * 0.0
