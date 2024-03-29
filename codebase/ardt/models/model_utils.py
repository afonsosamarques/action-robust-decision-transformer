import random

import numpy as np
import torch

from dataclasses import dataclass
from transformers.utils import ModelOutput

from evaluation_protocol.helpers import EvalWrapper


@dataclass
class DecisionTransformerOutput(ModelOutput):
    state_preds: torch.FloatTensor = None
    pr_action_preds: torch.FloatTensor = None
    adv_action_pred: torch.FloatTensor = None
    return_preds: torch.FloatTensor = None
    next_returns_preds: torch.FloatTensor = None
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
    max_ep_return: int = 9999999  # max episode return in the dataset
    returns_scale: float = 1  # normalisation of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution weighing episodes by trajectory lengths
    n_traj: int = 0  # to store the number of trajectories in the dataset
    
    def __init__(self, dataset, context_size, returns_scale, is_multipart):
        self.is_multipart = is_multipart
        # process dataset: add returns to go
        compute_rtg = lambda ds: {'returns_to_go': np.cumsum(ds["rewards"][::-1])[::-1]}
        compute_ret = lambda ds: {'returns': np.cumsum(ds["rewards"])}
        dataset = dataset.map(compute_rtg)
        dataset = dataset.map(compute_ret)
        self.dataset = dataset
        # get dataset settings
        self.max_ep_len = max([len(traj["rewards"]) for traj in dataset])
        self.pr_act_dim = len(dataset[0]["pr_actions"][0])
        self.adv_act_dim = len(dataset[0]["adv_actions"][0])
        self.state_dim = len(dataset[0]["observations"][0])
        self.context_size = context_size
        self.returns_scale = returns_scale
        # process returns data
        self.max_ep_rtg = None
        self.min_ep_rtg = None
        for traj in dataset:
            self.max_ep_rtg = np.max(traj['returns_to_go']) if self.max_ep_rtg is None else max(self.max_ep_rtg, np.max(traj['returns_to_go']))
            self.min_ep_rtg = np.min(traj['returns_to_go']) if self.min_ep_rtg is None else min(self.min_ep_rtg, np.min(traj['returns_to_go']))
        self.rtg_shift = 0 if self.min_ep_rtg > 0 else (-self.min_ep_rtg + 1e-4)
        # retrieve lower bounds for actions
        self.pr_act_lb = min(0.0, (int(np.min([np.min(traj["pr_actions"]) for traj in dataset])) - 1) * 5.0)
        self.adv_act_lb = min(0.0, (int(np.min([np.min(traj["adv_actions"]) for traj in dataset])) - 1) * 5.0)
        # collect statistics about states to produce normalisation constants
        states = []
        traj_lens = []
        for obs in dataset["observations"]:
            states.extend(obs)
            traj_lens.append(len(obs))
        states = np.vstack(states)
        traj_lens = np.array(traj_lens)
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
        s, a_pr, a_adv, r, rtg, next_rtg, d, tsteps, mask = [], [], [], [], [], [], [], [], []

        for idx in batch_idx:
            traj = self.dataset[int(idx)]

            # get sequences from the dataset
            if self.is_multipart:
                window_size = self.context_size + 1
                start = random.randint(1, len(traj["rewards"]) - 1)
            else:
                window_size = self.context_size
                start = random.randint(0, len(traj["rewards"]) - 1)

            s.append(np.array(traj["observations"][start : start + self.context_size]).reshape(1, -1, self.state_dim))
            pr_actions = np.array(traj["pr_actions"][start : start + self.context_size]).reshape(1, -1, self.pr_act_dim)
            a_pr.append(pr_actions)
            adv_actions = np.array(traj["adv_actions"][start : start + self.context_size]).reshape(1, -1, self.adv_act_dim)
            a_adv.append(adv_actions)       
            r.append(np.array(traj["rewards"][start : start + self.context_size]).reshape(1, -1, 1))
            rtg.append(np.array(traj["returns_to_go"][start : start + self.context_size]).reshape(1, -1, 1))
            d.append(np.array(traj["dones"][start : start + self.context_size]).reshape(1, -1))
            tsteps.append(np.arange(start, start + s[-1].shape[1]).reshape(1, -1))
            tsteps[-1][tsteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff

            # normalising and padding; we pad with zeros to the left of the sequence
            # except for actions, where we need to pad with some negative number well outside of domain
            tlen = s[-1].shape[1]
            padlen = window_size - tlen
            
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            s[-1] = np.concatenate(
                [np.zeros((1, padlen, self.state_dim)) * 1.0, s[-1]], axis=1,
            )

            a_pr[-1] = np.concatenate(
                [np.ones((1, padlen, self.pr_act_dim)) * self.pr_act_lb, a_pr[-1]], axis=1,
            )

            a_adv[-1] = np.concatenate(
                [np.ones((1, padlen, self.adv_act_dim)) * self.adv_act_lb, a_adv[-1]], axis=1,
            )

            r[-1] /= self.returns_scale
            r[-1] = np.concatenate(
                [np.zeros((1, padlen, 1)) * 1.0, r[-1]], axis=1,
            )

            rtg[-1] /= self.returns_scale
            rtg[-1] = np.concatenate(
                [np.zeros((1, padlen, 1)) * 1.0, rtg[-1]], axis=1,
            )

            d[-1] = np.concatenate(
                [np.ones((1, padlen)) * 2.0, d[-1]], axis=1,
            )

            tsteps[-1] = np.concatenate([np.zeros((1, padlen)), tsteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, padlen)), np.ones((1, tlen))], axis=1))  # disregard padded values

        # stack everything into tensors and return
        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a_pr = torch.from_numpy(np.concatenate(a_pr, axis=0)).float()
        a_adv = torch.from_numpy(np.concatenate(a_adv, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
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


class DTEvalWrapper(EvalWrapper):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.returns_scale = self.model.config.returns_scale if 'returns_scale' in self.model.config.to_dict().keys() else self.model.config.scale
        self.device = next(model.parameters()).device
        self.has_started = False
        self.batch_size = 1
        self.t = 0
    
    def new_eval(self, start_state, eval_target):
        self.has_started = True
        # from environment
        self.target_return = torch.tensor(eval_target/self.returns_scale, device=self.device, dtype=torch.float32).reshape(1, 1)
        self.states = torch.from_numpy(start_state).reshape(1, self.model.config.state_dim).to(device=self.device, dtype=torch.float32)
        # independent
        self.t = 0
        self.actions = torch.zeros((0, self.model.config.act_dim), device=self.device, dtype=torch.float32)
        self.rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        self.timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)

    def new_batch_eval(self, start_states, eval_target):
        self.has_started = True
        self.batch_size = len(start_states)
        # from environment
        self.target_returns = torch.zeros((self.batch_size, 1, 1), device=self.device, dtype=torch.float32)
        self.target_returns[:, 0] = torch.tensor(eval_target/self.returns_scale, device=self.device, dtype=torch.float32)
        self.states = torch.from_numpy(start_states).reshape(self.batch_size, 1, self.model.config.state_dim).to(device=self.device, dtype=torch.float32)
        # independent
        self.t = 0
        self.actions = torch.zeros((self.batch_size, 0, self.model.config.act_dim), device=self.device, dtype=torch.float32)
        self.rewards = torch.zeros((self.batch_size, 0), device=self.device, dtype=torch.float32)
        self.timesteps = torch.zeros((self.batch_size, 1), device=self.device, dtype=torch.long)

    def get_action(self, **kwargs):
        if not self.has_started:
            raise RuntimeError("Must call new_eval before get_action.")
        self.actions = torch.cat([self.actions, torch.zeros((1, self.model.config.act_dim), device=self.device)], dim=0)
        self.rewards = torch.cat([self.rewards, torch.zeros(1, device=self.device)])

        pr_action, adv_action = self.model.get_actions(
            self.states,
            self.actions,
            self.rewards,
            self.target_return,
            self.timesteps,
            self.device,
        )
        return pr_action.detach().cpu().numpy(), adv_action.detach().cpu().numpy()
    
    def get_batch_actions(self, **kwargs):
        if not self.has_started:
            raise RuntimeError("Must call new_eval before get_batch_actions.")
        self.actions = torch.cat([self.actions, torch.zeros((self.batch_size, 1, self.model.config.act_dim), device=self.device)], dim=1)
        self.rewards = torch.cat([self.rewards, torch.zeros((self.batch_size, 1), device=self.device)], dim=1)

        pr_actions, adv_actions = self.model.get_actions(
            self.states,
            self.actions,
            self.rewards,
            self.target_returns,
            self.timesteps,
            self.device,
            batch_size=self.batch_size,
        )
        return pr_actions.detach().cpu().numpy(), adv_actions.detach().cpu().numpy()
    
    def update_history(self, pr_action, adv_action, state, reward, timestep):
        if not self.has_started:
            raise RuntimeError("Must call new_eval before update_history.")
        self.actions[-1] = torch.tensor(pr_action).to(device=self.device)
        self.rewards[-1] = reward

        cur_state = torch.from_numpy(state.astype(np.float32)).to(device=self.device).reshape(1, self.model.config.state_dim)
        self.states = torch.cat([self.states, cur_state], dim=0)
        
        pred_return = self.target_return[0, -1] - (reward / self.returns_scale)
        self.target_return = torch.cat([self.target_return, pred_return.reshape(1, 1)], dim=1)

        self.t = timestep
        self.timesteps = torch.cat([self.timesteps, torch.ones((1, 1), device=self.device, dtype=torch.long) * (self.t + 1)], dim=1)

    def update_batch_history(self, pr_actions, adv_actions, states, rewards, timestep):
        if not self.has_started:
            raise RuntimeError("Must call new_eval before update_batch_history.")
        self.actions[:, -1] = torch.tensor(pr_actions).to(device=self.device)
        rewards_tsr = torch.from_numpy(rewards.astype(np.float32)).to(device=self.device)
        self.rewards[:, -1] = rewards_tsr

        cur_states = torch.from_numpy(states.astype(np.float32)).to(device=self.device).reshape(self.batch_size, 1, self.model.config.state_dim)
        self.states = torch.cat([self.states, cur_states], dim=1)
        
        pred_returns = self.target_returns[:, -1, :] - (rewards_tsr.reshape(self.batch_size, 1) / self.returns_scale)
        self.target_returns = torch.cat([self.target_returns, pred_returns.reshape(self.batch_size, 1, 1)], dim=1)

        self.t = timestep
        self.timesteps = torch.cat([self.timesteps, torch.ones((self.batch_size, 1), device=self.device, dtype=torch.long) * (self.t + 1)], dim=1)


class ADTEvalWrapper(EvalWrapper):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.returns_scale = self.model.config.returns_scale if 'returns_scale' in self.model.config.to_dict().keys() else self.model.config.scale
        self.device = next(model.parameters()).device
        self.has_started = False
        self.t = 0
    
    def new_eval(self, start_state, eval_target):
        self.has_started = True
        # from environment
        self.target_return = torch.tensor(eval_target/self.returns_scale, device=self.device, dtype=torch.float32).reshape(1, 1)
        self.states = torch.from_numpy(start_state).reshape(1, self.model.config.state_dim).to(device=self.device, dtype=torch.float32)
        # independent
        self.t = 0
        self.pr_actions = torch.zeros((0, self.model.config.pr_act_dim), device=self.device, dtype=torch.float32)
        self.adv_actions = torch.zeros((0, self.model.config.adv_act_dim), device=self.device, dtype=torch.float32)
        self.rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        self.timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)

    def new_batch_eval(self, start_states, eval_target):
        self.has_started = True
        self.batch_size = len(start_states)
        # from environment
        self.target_returns = torch.zeros((self.batch_size, 1, 1), device=self.device, dtype=torch.float32)
        self.target_returns[:, 0] = torch.tensor(eval_target/self.returns_scale, device=self.device, dtype=torch.float32)
        self.states = torch.from_numpy(start_states).reshape(self.batch_size, 1, self.model.config.state_dim).to(device=self.device, dtype=torch.float32)
        # independent
        self.t = 0
        self.pr_actions = torch.zeros((self.batch_size, 0, self.model.config.pr_act_dim), device=self.device, dtype=torch.float32)
        self.adv_actions = torch.zeros((self.batch_size, 0, self.model.config.adv_act_dim), device=self.device, dtype=torch.float32)
        self.rewards = torch.zeros((self.batch_size, 0), device=self.device, dtype=torch.float32)
        self.timesteps = torch.zeros((self.batch_size, 1), device=self.device, dtype=torch.long)

    def get_action(self, pr_action=None, **kwargs):
        if not self.has_started:
            raise RuntimeError("Must call new_eval before get_action.")
        new_pr_action = torch.tensor(pr_action, device=self.device).reshape(1, -1) if pr_action is not None else torch.zeros((1, self.model.config.pr_act_dim), device=self.device)
        self.pr_actions = torch.cat([self.pr_actions, new_pr_action], dim=0)
        self.adv_actions = torch.cat([self.adv_actions, torch.zeros((1, self.model.config.adv_act_dim), device=self.device)], dim=0)
        self.rewards = torch.cat([self.rewards, torch.zeros(1, device=self.device)])

        pr_action, adv_action = self.model.get_actions(
            self.states,
            self.pr_actions,
            self.adv_actions,
            self.rewards,
            self.target_return,
            self.timesteps,
            self.device,
        )
        return pr_action.detach().cpu().numpy(), adv_action.detach().cpu().numpy()
    
    def get_batch_actions(self, pr_actions=None, **kwargs):
        if not self.has_started:
            raise RuntimeError("Must call new_eval before get_batch_actions.")
        new_pr_actions = torch.tensor(pr_actions, device=self.device).reshape(self.batch_size, 1, -1) if pr_actions is not None else torch.zeros((self.batch_size, 1, self.model.config.pr_act_dim), device=self.device)
        self.pr_actions = torch.cat([self.pr_actions, new_pr_actions], dim=1)
        self.adv_actions = torch.cat([self.adv_actions, torch.zeros((self.batch_size, 1, self.model.config.adv_act_dim), device=self.device)], dim=1)
        self.rewards = torch.cat([self.rewards, torch.zeros((self.batch_size, 1), device=self.device)], dim=1)

        pr_actions, adv_actions = self.model.get_actions(
            self.states,
            self.pr_actions,
            self.adv_actions,
            self.rewards,
            self.target_returns,
            self.timesteps,
            self.device,
            batch_size=self.batch_size,
        )
        return pr_actions.detach().cpu().numpy(), adv_actions.detach().cpu().numpy()
    
    def update_history(self, pr_action, adv_action, state, reward, timestep):
        if not self.has_started:
            raise RuntimeError("Must call new_eval before get_action.")
        self.pr_actions[-1] = torch.tensor(pr_action).to(device=self.device)
        self.adv_actions[-1] = torch.tensor(adv_action).to(device=self.device)
        self.rewards[-1] = reward

        cur_state = torch.from_numpy(state.astype(np.float32)).to(device=self.device).reshape(1, self.model.config.state_dim)
        self.states = torch.cat([self.states, cur_state], dim=0)
        
        pred_target_return = self.target_return[0, -1] - (reward / self.returns_scale)
        self.target_return = torch.cat([self.target_return, pred_target_return.reshape(1, 1)], dim=1)

        self.t = timestep
        self.timesteps = torch.cat([self.timesteps, torch.ones((1, 1), device=self.device, dtype=torch.long) * (self.t + 1)], dim=1)

    def update_batch_history(self, pr_actions, adv_actions, states, rewards, timestep):
        if not self.has_started:
            raise RuntimeError("Must call new_eval before update_batch_history.")
        self.pr_actions[:, -1] = torch.tensor(pr_actions).to(device=self.device)
        self.adv_actions[:, -1] = torch.tensor(adv_actions).to(device=self.device)
        rewards_tsr = torch.from_numpy(rewards.astype(np.float32)).to(device=self.device)
        self.rewards[:, -1] = rewards_tsr

        cur_states = torch.from_numpy(states.astype(np.float32)).to(device=self.device).reshape(self.batch_size, 1, self.model.config.state_dim)
        self.states = torch.cat([self.states, cur_states], dim=1)
        
        pred_target_returns = self.target_returns[:, -1, :] - (rewards_tsr.reshape(self.batch_size, 1) / self.returns_scale)
        self.target_returns = torch.cat([self.target_returns, pred_target_returns.reshape(self.batch_size, 1, 1)], dim=1)

        self.t = timestep
        self.timesteps = torch.cat([self.timesteps, torch.ones((self.batch_size, 1), device=self.device, dtype=torch.long) * (self.t + 1)], dim=1)
