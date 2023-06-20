import random
from dataclasses import dataclass

import numpy as np
import torch

from transformers import DecisionTransformerModel, DecisionTransformerGPT2Model
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
    

class GFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert min(x) >= 0 and max(x) <= 1
        return x / (1 - x)


class AdversarialDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = torch.nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_pr_action = torch.nn.Linear(config.pr_act_dim, config.hidden_size)
        self.embed_adv_action = torch.nn.Linear(config.adv_act_dim, config.hidden_size)
        self.embed_ln = torch.nn.LayerNorm(config.hidden_size)

        self.predict_alpha = torch.nn.Linear(config.hidden_size, 1)
        self.predict_epsilon = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, 1)] + [torch.nn.Sigmoid()] + [GFunc()])
        )
        self.predict_adv_action = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, config.adv_act_dim)] + ([torch.nn.Tanh()] if config.action_tanh else []))
        )

        self.post_init()

    def forward(
        self,
        is_train=True,
        states=None,
        pr_actions=None,
        adv_actions=None,
        rewards=None,
        returns_to_go=None,
        timesteps=None,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,):
        # TO COMMENT
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        pr_action_embeddings = self.embed_pr_action(pr_actions)
        adv_action_embeddings = self.embed_adv_action(adv_actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings += time_embeddings
        pr_action_embeddings += time_embeddings
        adv_action_embeddings += time_embeddings
        returns_embeddings += time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, a'_1, R_2, s_2, a_2, a'_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, pr_action_embeddings, adv_action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 4 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 4 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=stacked_inputs.device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), pr_actions (2) or adv_actions (3); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # FIXME the original indexing seems off, changed it but it's definitely going to break something
        alpha_preds = self.predict_alpha(x[:, :])  # predict next return given everything
        epsilon_preds = self.predict_epsilon(x[:, :])  # predict next return given everything
        adv_action_preds = self.predict_adv_action(x[:, :3])  # predict next action given return, state and pr_action

        if is_train:
            # return loss
            rtg_dist = torch.distributions.beta.Beta(alpha_preds + epsilon_preds, alpha_preds)
            scaled_rtg_obs = returns_to_go / self.config.max_return
            rtg_log_prob = rtg_dist.log_prob(scaled_rtg_obs)
            rtg_entropy = rtg_dist.entropy()
            rtg_loss = -rtg_log_prob - self.config.lambda1 * rtg_entropy

            adv_action_preds = adv_action_preds.reshape(-1, self.config.adv_act_dim)[attention_mask.reshape(-1) > 0]
            adv_action_targets = adv_actions.reshape(-1, self.config.adv_act_dim)[attention_mask.reshape(-1) > 0]
            adv_action_loss = self.config.lambda2 * torch.mean((adv_action_preds - adv_action_targets) ** 2)

            return {"loss": rtg_loss + adv_action_loss, 
                    "rtg_loss": rtg_loss, 
                    "rtg_log_prob": rtg_log_prob, 
                    "rtg_entropy": rtg_entropy, 
                    "adv_action_loss": adv_action_loss,
                    "rtg_preds": self.config.max_return * rtg_dist.rsample(),}
        else:
            # return predictions
            if not return_dict:
                return (self.config.max_return * rtg_dist.mean, adv_action_preds)

            return DecisionTransformerOutput(
                rtg_preds=self.config.max_return * rtg_dist.mean,
                adv_action_preds=adv_action_preds,
                hidden_states=encoder_outputs.hidden_states,
                last_hidden_state=encoder_outputs.last_hidden_state,
                attentions=encoder_outputs.attentions,
            )
        

class ExpFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)
    

class StochasticDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = torch.nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_pr_action = torch.nn.Linear(config.pr_act_dim, config.hidden_size)
        self.embed_ln = torch.nn.LayerNorm(config.hidden_size)

        self.predict_mu = torch.nn.Linear(config.hidden_size, config.pr_act_dim)
        self.predict_sigma_exp = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, config.pr_act_dim)] + [ExpFunc()])  # keeping it to diag matrix for now
        )

        self.post_init()

    def forward(
        self,
        is_train=True,
        states=None,
        pr_actions=None,
        adv_actions=None,
        rewards=None,
        returns_to_go=None,
        timesteps=None,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,):
        # TO COMMENT
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        pr_action_embeddings = self.embed_pr_action(pr_actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings += time_embeddings
        pr_action_embeddings += time_embeddings
        returns_embeddings += time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, a'_1, R_2, s_2, a_2, a'_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, pr_action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=stacked_inputs.device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # return (0), states (1) or pr_actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # FIXME the original indexing seems off, changed it but it's definitely going to break something
        mu_preds = self.predict_mu(x[:, :2])  # predict next action dist. mean given return and state
        sigma_preds = self.predict_sigma_exp(x[:, :2])  # predict next action dist. sigma given return and state

        if is_train:
            # return loss
            pr_action_dist = torch.distributions.Normal(mu_preds, sigma_preds)
            pr_action_log_prob = pr_action_dist.log_prob(pr_actions)
            pr_action_entropy = pr_action_dist.entropy()
            pr_action_loss = -pr_action_log_prob - self.config.lambda1 * pr_action_entropy
            return {"loss": pr_action_loss, 
                    "pr_action_log_prob": pr_action_log_prob, 
                    "pr_action_entropy": pr_action_entropy}
        else:
            # return predictions
            if not return_dict:
                return (pr_action_dist.mean)

            return DecisionTransformerOutput(
                pr_action_preds=pr_action_dist.mean,
                hidden_states=encoder_outputs.hidden_states,
                last_hidden_state=encoder_outputs.last_hidden_state,
                attentions=encoder_outputs.attentions,
            )


class ExpFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)
    

class SingleAgentRobustDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = torch.nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_pr_action = torch.nn.Linear(config.pr_act_dim, config.hidden_size)
        self.embed_adv_action = torch.nn.Linear(config.adv_act_dim, config.hidden_size)
        self.embed_ln = torch.nn.LayerNorm(config.hidden_size)

        self.predict_mu = torch.nn.Linear(config.hidden_size, config.pr_act_dim)
        self.predict_sigma_exp = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, config.pr_act_dim)] + [ExpFunc()])  # keeping it to diag matrix for now
        )
        self.predict_adv_action = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, config.adv_act_dim)] + ([torch.nn.Tanh()] if config.action_tanh else []))
        )

        self.post_init()

    def forward(
        self,
        is_train=True,
        states=None,
        pr_actions=None,
        adv_actions=None,
        rewards=None,
        returns_to_go=None,
        timesteps=None,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,):
        #
        # TO COMMENT
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        pr_action_embeddings = self.embed_pr_action(pr_actions)
        adv_action_embeddings = self.embed_adv_action(adv_actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings += time_embeddings
        pr_action_embeddings += time_embeddings
        adv_action_embeddings += time_embeddings
        returns_embeddings += time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, a'_1, R_2, s_2, a_2, a'_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, pr_action_embeddings, adv_action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 4 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 4 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=stacked_inputs.device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), pr_actions (2) or adv_actions (3); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # FIXME check indexing
        mu_preds = self.predict_mu(x[:, 1])  # predict next pr action dist. mean given return and state
        sigma_preds = self.predict_sigma_exp(x[:, 1])  # predict next pr action dist. sigma given return and state
        pr_action_dist = torch.distributions.Normal(mu_preds, sigma_preds)
        
        adv_action_preds = self.predict_adv_action(x[:, 2])  # predict next adv action given return, state and pr_action

        if is_train:
            # return loss
            pr_action_log_prob = pr_action_dist.log_prob(pr_actions).sum()
            pr_action_entropy = pr_action_dist.entropy().sum()
            pr_action_loss = -pr_action_log_prob - self.config.lambda1 * pr_action_entropy

            adv_action_preds = adv_action_preds.reshape(-1, self.config.adv_act_dim)[attention_mask.reshape(-1) > 0]
            adv_action_targets = adv_actions.reshape(-1, self.config.adv_act_dim)[attention_mask.reshape(-1) > 0]
            adv_action_loss = self.config.lambda2 * torch.mean((adv_action_preds - adv_action_targets) ** 2)

            return {"loss": pr_action_loss + adv_action_loss, 
                    "pr_action_loss": pr_action_loss, 
                    "pr_action_log_prob": pr_action_log_prob, 
                    "pr_action_entropy": pr_action_entropy, 
                    "adv_action_loss": adv_action_loss}
        else:
            # return predictions
            if not return_dict:
                return (pr_action_dist.mean, adv_action_preds)

            return DecisionTransformerOutput(
                pr_action_preds=pr_action_dist.mean,
                adv_action_preds=adv_action_preds,
                hidden_states=encoder_outputs.hidden_states,
                last_hidden_state=encoder_outputs.last_hidden_state,
                attentions=encoder_outputs.attentions,
            )
        