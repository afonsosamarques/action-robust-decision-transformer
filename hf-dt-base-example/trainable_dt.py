import random
from dataclasses import dataclass

import numpy as np
import torch

from datasets import load_dataset
from huggingface_hub import login, list_models
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments


@dataclass
class DecisionTransformerGymDataCollator:
    return_tensors: str = "pt"  # pytorch tensors
    context_size: int = 1  # length of trajectories we use in training
    state_dim: int = 1  # size of state space
    act_dim: int = 1  # size of action space
    max_ep_len: int = 9999999  # max episode length in the dataset
    scale: float = 1  # normalisation of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution weighing episodes by trajectory lengths
    n_traj: int = 0  # to store the number of trajectories in the dataset

    def __init__(self, dataset, context_size, returns_scale):
        self.dataset = dataset
        # get dataset-specific features
        self.max_ep_len = max([len(traj["rewards"]) for traj in dataset])
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["observations"][0])
        self.context_size = context_size
        self.returns_scale = returns_scale
        # collect some statistics about the dataset
        states = []
        traj_lens = []
        for obs in dataset["observations"]:
            states.extend(obs)
            traj_lens.append(len(obs))
        traj_lens = np.array(traj_lens)
        states = np.vstack(states)
        # use stats to produce normalisation constants
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-8
        self.n_traj = traj_lens.shape[0]
        self.p_sample = traj_lens / sum(traj_lens)

    def __call__(self, features):

        def _discount_cumsum(x, gamma):
            # return-to-go calculation
            discount_cumsum = np.zeros_like(x)
            discount_cumsum[-1] = x[-1]
            for t in reversed(range(x.shape[0] - 1)):
                discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
            return discount_cumsum

        # FIXME this is a bit of a hack to be able to sample from a non-uniform distribution
        # the idea is that we re-sample with replacement from the dataset rather than just taking the batch
        # this also means we can sample according to the length of the trajectories
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=len(features),
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )

        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        
        for ind in batch_inds:
            traj = self.dataset[int(ind)]
            start = random.randint(0, len(traj["rewards"]) - 1)  # FIXME we are again randomising which feels dumb

            # get sequences from the dataset
            s.append(np.array(traj["observations"][start : start + self.context_size]).reshape(1, -1, self.state_dim))
            a.append(np.array(traj["actions"][start : start + self.context_size]).reshape(1, -1, self.act_dim))
            r.append(np.array(traj["rewards"][start : start + self.context_size]).reshape(1, -1, 1))
            d.append(np.array(traj["dones"][start : start + self.context_size]).reshape(1, -1))
            timesteps.append(np.arange(start, start + s[-1].shape[1]).reshape(1, -1))
            # FIXME feels hacky/dumb/unnecessary timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                _discount_cumsum(np.array(traj["rewards"][start:]), gamma=1.0)[
                    : s[-1].shape[1]  # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )

            # FIXME hacky... can't see the purpose; could be tied to +1 removed above
            # if rtg[-1].shape[1] < s[-1].shape[1]:
            #     rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # normalising and padding
            tlen = s[-1].shape[1]
            padlen = self.context_size - tlen
            
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            s[-1] = np.concatenate(
                [np.zeros((1, padlen, self.state_dim)) * 1.0, s[-1]], 
                axis=1,
            )
            
            a[-1] = np.concatenate(
                [np.ones((1, padlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )

            r[-1] = np.concatenate(
                [np.zeros((1, padlen, 1)) * 1.0, r[-1]], 
                axis=1,
            )

            d[-1] = np.concatenate(
                [np.ones((1, padlen)) * 2.0, d[-1]], 
                axis=1,
            )

            rtg[-1] /= self.scale
            rtg[-1] = np.concatenate(
                [np.zeros((1, padlen, 1)) * 1.0, rtg[-1]], 
                axis=1,
            ) 

            timesteps[-1] = np.concatenate([np.zeros((1, padlen)), timesteps[-1]], axis=1)

            # masking: disregard padded values
            mask.append(np.concatenate([np.zeros((1, padlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }
    

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
        _, action_preds, _ = self.original_forward(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return action_preds[0, -1]
    