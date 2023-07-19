import numpy as np
import torch

from transformers import DecisionTransformerModel

from .ardt_utils import DTEvalWrapper


class TrainableDT(DecisionTransformerModel):
    def __init__(self, config, logger):
        super().__init__(config)
        self.logger = logger
        self.step = 0

    def forward(self, **kwargs):
        new_kwargs = kwargs.copy()
        if "pr_actions" in new_kwargs:
            # change to be able to utilise the default code
            new_kwargs["actions"] = new_kwargs.pop("pr_actions")
            new_kwargs.pop("adv_actions")
        output = super().forward(**new_kwargs)

        # add the DT loss; applied only to non-padding values in action head
        action_targets = kwargs["pr_actions"]
        attention_mask = kwargs["attention_mask"]
        action_preds = output[1]
        act_dim = action_preds.shape[2]
        
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        self.step += 1
        loss = torch.mean((action_preds - action_targets) ** 2)

        if self.logger is not None and (self.step == 0 or self.step % self.config.log_interval_steps == 0):
            self.logger.add_entry(
                step=self.step,
                hyperparams=None,
                tr_losses={"loss": loss},
                dist_params=None,
                log=True
            )

        return {"loss": loss}

    def original_forward(self, **kwargs):
        new_kwargs = kwargs.copy()
        if "pr_actions" in new_kwargs:
            # change to be able to utilise the default code
            new_kwargs["actions"] = new_kwargs.pop("pr_actions")
            new_kwargs.pop("adv_actions")
        return super().forward(**new_kwargs)
    
    def eval(self, **kwargs):
        return DTEvalWrapper(self)
    
    def get_action(model, states, actions, rewards, returns_to_go, timesteps, device):
        # NOTE this implementation does not condition on past rewards
        # reshape to model input format
        states = states.reshape(1, -1, model.config.state_dim)
        actions = actions.reshape(1, -1, model.config.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # normalisation constants
        state_mean = torch.from_numpy(np.array(model.config.state_mean).astype(np.float32)).to(device=device)
        state_std = torch.from_numpy(np.array(model.config.state_std).astype(np.float32)).to(device=device)

        # retrieve window of observations based on context length
        states = states[:, -model.config.context_size :]
        actions = actions[:, -model.config.context_size :]
        returns_to_go = returns_to_go[:, -model.config.context_size :]
        timesteps = timesteps[:, -model.config.context_size :]

        # normalisation
        states = (states - state_mean) / state_std

        # pad all tokens to sequence length
        padlen = model.config.context_size - states.shape[1]
        attention_mask = torch.cat([torch.zeros(padlen, device=device), torch.ones(states.shape[1], device=device)]).to(dtype=torch.long).reshape(1, -1)
        states = torch.cat([torch.zeros((1, padlen, model.config.state_dim), device=device), states], dim=1).float()
        actions = torch.cat([torch.zeros((1, padlen, model.config.act_dim), device=device), actions], dim=1).float()
        returns_to_go = torch.cat([torch.zeros((1, padlen, 1), device=device), returns_to_go], dim=1).float()
        timesteps = torch.cat([torch.zeros((1, padlen), dtype=torch.long, device=device), timesteps], dim=1)

        # forward pass
        _, action_preds, _ = model.original_forward(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return action_preds[0, -1], action_preds[0, -1] * 0.0