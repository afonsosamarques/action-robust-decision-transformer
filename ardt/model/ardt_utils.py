import random
import numpy as np
import torch

from dataclasses import dataclass
from transformers.utils import ModelOutput


@dataclass
class DecisionTransformerOutput(ModelOutput):
    state_preds: torch.FloatTensor = None
    pr_action_preds: torch.FloatTensor = None
    adv_action_pred: torch.FloatTensor = None
    return_preds: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None


@dataclass
class DecisionTransformerGymDataCollator:
    return_tensors: str = "pt"  # pytorch tensors
    context_size: int = 1  # length of trajectories we use in training
    state_dim: int = 1  # size of state space
    pr_act_dim: int = 1  # size of protagonist action space
    adv_act_dim: int = 1  # size of antagonist action space
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
        self.pr_act_dim = len(dataset[0]["pr_actions"][0])
        self.adv_act_dim = len(dataset[0]["adv_actions"][0])
        self.state_dim = len(dataset[0]["observations"][0])
        self.context_size = context_size
        self.returns_scale = returns_scale
        # retrieve lower bounds for actions
        self.pr_act_lb = min(0.0, (int(np.min([np.min(traj["pr_actions"]) for traj in dataset])) - 1) * 5.0)
        self.adv_act_lb = min(0.0, (int(np.min(np.min([traj["adv_actions"] for traj in dataset]))) - 1) * 5.0)
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
        # sample with replacement according to trajectory length
        batch_idx = np.random.choice(
            np.arange(self.n_traj),
            size=len(features),
            replace=True,
            p=self.p_sample,
        )

        # a batch of dataset features
        s, a_pr, a_adv, r, d, rtg, tsteps, mask = [], [], [], [], [], [], [], []
        
        for idx in batch_idx:
            traj = self.dataset[int(idx)]
            start = random.randint(0, len(traj["rewards"]) - 1)

            # get sequences from the dataset
            s.append(np.array(traj["observations"][start : start + self.context_size]).reshape(1, -1, self.state_dim))
            a_pr.append(np.array(traj["pr_actions"][start : start + self.context_size]).reshape(1, -1, self.pr_act_dim))
            a_adv.append(np.array(traj["adv_actions"][start : start + self.context_size]).reshape(1, -1, self.adv_act_dim))
            r.append(np.array(traj["rewards"][start : start + self.context_size]).reshape(1, -1, 1))
            d.append(np.array(traj["dones"][start : start + self.context_size]).reshape(1, -1))

            tsteps.append(np.arange(start, start + s[-1].shape[1]).reshape(1, -1))
            tsteps[-1][tsteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff

            rewards = np.array(traj["rewards"][start:])
            rtg.append(np.cumsum(rewards[::-1])[::-1][:self.context_size].reshape(1, -1, 1))

            # normalising and padding; we pad with zeros to the left of the sequence
            # except for actions, where we need to pad with some negative number well outside of domain
            tlen = s[-1].shape[1]
            padlen = self.context_size - tlen
            
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            s[-1] = np.concatenate(
                [np.zeros((1, padlen, self.state_dim)) * 1.0, s[-1]], 
                axis=1,
            )
            
            a_pr[-1] = np.concatenate(
                [np.ones((1, padlen, self.pr_act_dim)) * self.pr_act_lb, a_pr[-1]],
                axis=1,
            )

            a_adv[-1] = np.concatenate(
                [np.ones((1, padlen, self.adv_act_dim)) * self.adv_act_lb, a_adv[-1]],
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

            tsteps[-1] = np.concatenate([np.zeros((1, padlen)), tsteps[-1]], axis=1)

            # masking: disregard padded values
            mask.append(np.concatenate([np.zeros((1, padlen)), np.ones((1, tlen))], axis=1))

        # stack everything into tensors and return
        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a_pr = torch.from_numpy(np.concatenate(a_pr, axis=0)).float()
        a_adv = torch.from_numpy(np.concatenate(a_adv, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        tsteps = torch.from_numpy(np.concatenate(tsteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        return {
            "states": s,
            "pr_actions": a_pr,
            "adv_actions": a_adv,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": tsteps,
            "attention_mask": mask,
        }
    

class BetaParamsSquashFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, min_log=-5.0, max_log=4.0):
        return min_log + 0.5 * (max_log - min_log) * (torch.tanh(p) + 1.0)


class StdSquashFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, min_log_std=-5.0, max_log_std=0.0):
        return min_log_std + 0.5 * (max_log_std - min_log_std) * (torch.tanh(p) + 1.0)


class ExpFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)
