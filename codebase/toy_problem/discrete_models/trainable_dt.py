import numpy as np
import torch

from transformers import DecisionTransformerModel

from .ardt_utils import DecisionTransformerOutput, DTEvalWrapper


class TrainableDT(DecisionTransformerModel):
    def __init__(self, config, logger=None):
        config.action_tanh = False
        super().__init__(config)
        self.logger = logger
        self.step = 0
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, is_train=True, returns_to_go_scaled=None, **kwargs):
        new_kwargs = kwargs.copy()
        if "pr_actions" in new_kwargs:
            # change to be able to utilise the default code
            new_kwargs["actions"] = new_kwargs.pop("pr_actions")
            new_kwargs.pop("adv_actions")
        
        batch_size, seq_length = new_kwargs["states"].shape[0], new_kwargs["states"].shape[1]
        output = self.sigmoid(super().forward(**new_kwargs)[1])
        if is_train:
            action_preds = output.reshape(-1, self.config.act_dim)
            action_targets = new_kwargs["actions"].reshape(-1, self.config.act_dim)
            loss = torch.nn.functional.binary_cross_entropy(action_preds, action_targets)
            return {"loss": loss}
        else:
            # simply return predictions
            action_preds = (output > 0.5).to(torch.int32).reshape(batch_size, -1, self.config.act_dim)
            if not kwargs.get("return_dict", False):
                return (action_preds)
            return DecisionTransformerOutput(action_preds=action_preds)
    
    def eval(self, **kwargs):
        return DTEvalWrapper(self)
    
    def get_actions(self, states, actions, rewards, returns_to_go, timesteps, device, batch_size=1):
        states = states.reshape(batch_size, -1, self.config.state_dim)
        actions = actions.reshape(batch_size, -1, self.config.act_dim)
        returns_to_go = returns_to_go.reshape(batch_size, -1, 1)
        timesteps = timesteps.reshape(batch_size, -1)

        action_preds = self.forward(
            is_train=False,
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=torch.ones((batch_size, states.shape[1]), device=device, dtype=torch.long),
            return_dict=False,
        )

        return action_preds[:, -1], torch.ones_like(action_preds[:, -1]) * -1.0
