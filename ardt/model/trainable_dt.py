import random
from dataclasses import dataclass

import numpy as np
import torch

from datasets import load_dataset
from huggingface_hub import login, list_models
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments


class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)

        # add the DT loss; applied only to non-padding values in action head
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        action_preds = output[1]
        act_dim = action_preds.shape[2]
        
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        return {"loss": torch.mean((action_preds - action_targets) ** 2)}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)
    
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
        _, action_preds, _ = model.forward(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return action_preds[0, -1]
