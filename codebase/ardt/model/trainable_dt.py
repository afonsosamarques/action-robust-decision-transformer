import numpy as np
import torch

from transformers import DecisionTransformerModel

from .ardt_utils import DTEvalWrapper


class TrainableDT(DecisionTransformerModel):
    def __init__(self, config, logger=None):
        super().__init__(config)
        self.logger = logger
        self.step = 0

    def forward(self, is_train=True, returns_to_go_scaled=None, **kwargs):
        new_kwargs = kwargs.copy()
        if "pr_actions" in new_kwargs:
            # change to be able to utilise the default code
            new_kwargs["actions"] = new_kwargs.pop("pr_actions")
            new_kwargs.pop("adv_actions")
        
        if is_train:
            # add the DT loss; applied only to non-padding values in action head
            self.step += 1

            attention_mask = new_kwargs["attention_mask"]
            output = super().forward(**new_kwargs)
            
            action_preds = output[1].reshape(-1, self.config.act_dim)[attention_mask.reshape(-1) > 0]
            action_targets = new_kwargs["actions"].reshape(-1, self.config.act_dim)[attention_mask.reshape(-1) > 0]
            
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
        else:
            # simply return predictions
            return super().forward(**new_kwargs)
    
    def eval(self, **kwargs):
        return DTEvalWrapper(self)
    
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, device):
        # NOTE this implementation does not condition on past rewards
        # reshape to model input format
        states = states.reshape(1, -1, self.config.state_dim)
        actions = actions.reshape(1, -1, self.config.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # normalisation constants
        state_mean = torch.from_numpy(np.array(self.config.state_mean).astype(np.float32)).to(device=device)
        state_std = torch.from_numpy(np.array(self.config.state_std).astype(np.float32)).to(device=device)

        # retrieve window of observations based on context length
        states = states[:, -self.config.context_size :]
        actions = actions[:, -self.config.context_size :]
        returns_to_go = returns_to_go[:, -self.config.context_size :]
        timesteps = timesteps[:, -self.config.context_size :]

        # normalisation
        states = (states - state_mean) / state_std

        # pad all tokens to sequence length
        padlen = self.config.context_size - states.shape[1]
        attention_mask = torch.cat([torch.zeros(padlen, device=device), torch.ones(states.shape[1], device=device)]).to(dtype=torch.long).reshape(1, -1)
        states = torch.cat([torch.zeros((1, padlen, self.config.state_dim), device=device), states], dim=1).float()
        actions = torch.cat([torch.zeros((1, padlen, self.config.act_dim), device=device), actions], dim=1).float()
        returns_to_go = torch.cat([torch.zeros((1, padlen, 1), device=device), returns_to_go], dim=1).float()
        timesteps = torch.cat([torch.zeros((1, padlen), dtype=torch.long, device=device), timesteps], dim=1)

        # forward pass
        _, action_preds, _ = self.forward(
            is_train=False,
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return action_preds[0, -1], action_preds[0, -1] * 0.0
    
    def get_batch_actions(self, batch_size, states, actions, rewards, returns_to_go, timesteps, device):
        # NOTE this implementation does not condition on past rewards
        # reshape to model input format
        states = states.reshape(batch_size, -1, self.config.state_dim)
        actions = actions.reshape(batch_size, -1, self.config.act_dim)
        returns_to_go = returns_to_go.reshape(batch_size, -1, 1)
        timesteps = timesteps.reshape(batch_size, -1)

        # normalisation constants
        state_mean = torch.from_numpy(np.array(self.config.state_mean).astype(np.float32)).to(device=device)
        state_std = torch.from_numpy(np.array(self.config.state_std).astype(np.float32)).to(device=device)

        # retrieve window of observations based on context length
        states = states[:, -self.config.context_size :]
        actions = actions[:, -self.config.context_size :]
        returns_to_go = returns_to_go[:, -self.config.context_size :]
        timesteps = timesteps[:, -self.config.context_size :]

        # normalisation
        states = (states - state_mean) / state_std

        padlen = self.config.context_size - states.shape[1]
        attention_mask = torch.cat([torch.zeros((batch_size, padlen), device=device), torch.ones((batch_size, states.shape[1]), device=device)], dim=1).to(dtype=torch.long)
        states = torch.cat([torch.zeros((batch_size, padlen, self.config.state_dim), device=device), states], dim=1).float()
        actions = torch.cat([torch.zeros((batch_size, padlen, self.config.act_dim), device=device), actions], dim=1).float()
        returns_to_go = torch.cat([torch.zeros((batch_size, padlen, 1), device=device), returns_to_go], dim=1).float()
        timesteps = torch.cat([torch.zeros((batch_size, padlen), dtype=torch.long, device=device), timesteps], dim=1)

        # forward pass
        _, action_preds, _ = self.forward(
            is_train=False,
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return action_preds[:, -1], action_preds[:, -1] * 0.0
