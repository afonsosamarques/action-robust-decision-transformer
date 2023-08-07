import numpy as np
import torch

from transformers import DecisionTransformerModel, DecisionTransformerGPT2Model

from .ardt_utils import DecisionTransformerOutput, ADTEvalWrapper
from .ardt_utils import StdReturnSquashFunc, StdSquashFunc, ExpFunc


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

        self.predict_mu = torch.nn.Linear(config.hidden_size, 1)
        self.predict_sigma = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, 1)] + [StdReturnSquashFunc()] + [ExpFunc()])
        )
        self.predict_adv_action = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, config.adv_act_dim)] + ([torch.nn.Tanh()]))
        )

        self.post_init()

    def forward(
        self,
        is_train=True,
        states=None,
        pr_actions=None,
        pr_actions_filtered=None,
        adv_actions=None,
        adv_actions_filtered=None,
        rewards=None,
        returns_to_go=None,
        returns_to_go_scaled=None,
        timesteps=None,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,):
        #
        # setting configurations for what goes into the output
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
        pr_action_embeddings = self.embed_pr_action(pr_actions_filtered)
        adv_action_embeddings = self.embed_adv_action(adv_actions_filtered)
        returns_embeddings = self.embed_return(returns_to_go_scaled)
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
        mu_preds = self.predict_mu(x[:, 1])
        sigma_preds = self.predict_sigma(x[:, 1])  
        rtg_dist = torch.distributions.Normal(mu_preds, sigma_preds)

        adv_action_preds = self.predict_adv_action(x[:, 2])  # predict next action given return, state and pr_actions (latest pr action is zero-ed out)

        if is_train:
            # return loss
            rtg_log_prob = rtg_dist.log_prob(returns_to_go)[attention_mask > 0].mean()

            adv_action_preds_mask = adv_action_preds.reshape(-1, self.config.adv_act_dim)[attention_mask.reshape(-1) > 0]
            adv_action_targets_mask = adv_actions.reshape(-1, self.config.adv_act_dim)[attention_mask.reshape(-1) > 0]
            adv_action_loss = torch.mean((adv_action_preds_mask - adv_action_targets_mask) ** 2)
                
            return {"loss": -rtg_log_prob + self.config.lambda2 * adv_action_loss,
                    "rtg_log_prob": rtg_log_prob, 
                    "rtg_loss": -rtg_log_prob,
                    "adv_action_loss": adv_action_loss,
                    "mu": mu_preds,
                    "sigma": sigma_preds,
                    "rtg_preds": rtg_dist.mean,
                    "adv_action_preds": adv_action_preds}
        else:
            # return predictions
            if not return_dict:
                return (rtg_dist.icdf(torch.tensor([0.025], device=stacked_inputs.device)), adv_action_preds)

            return DecisionTransformerOutput(
                rtg_preds=rtg_dist.icdf(torch.tensor([0.025], device=stacked_inputs.device)),
                adv_action_preds=adv_action_preds,
                # hidden_states=encoder_outputs.hidden_states,
                # last_hidden_state=encoder_outputs.last_hidden_state,
                # attentions=encoder_outputs.attentions,
            )
    

class StochasticDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = torch.nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_adv_action = torch.nn.Linear(config.adv_act_dim, config.hidden_size)
        self.embed_pr_action = torch.nn.Linear(config.pr_act_dim, config.hidden_size)
        self.embed_ln = torch.nn.LayerNorm(config.hidden_size)

        self.predict_mu = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, config.pr_act_dim)] + ([torch.nn.Tanh()]))
        )
        self.predict_sigma = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, config.pr_act_dim)] + [StdSquashFunc()] + [ExpFunc()])
        )

        self.post_init()

    def forward(
        self,
        is_train=True,
        states=None,
        pr_actions=None,
        pr_actions_filtered=None,
        adv_actions=None,
        rewards=None,
        returns_to_go=None,
        returns_to_go_scaled=None,
        timesteps=None,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,):
        #
        # setting configurations for what goes into the output
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
        pr_action_embeddings = self.embed_pr_action(pr_actions_filtered)
        adv_action_embeddings = self.embed_adv_action(adv_actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings += time_embeddings
        pr_action_embeddings += time_embeddings
        adv_action_embeddings += time_embeddings
        returns_embeddings += time_embeddings

        # this makes the sequence look like (R'_1, s_1, a_1, R'_2, s_2, a_2, ...)
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
        # return (0), states (1), adv_actions (2) or pr_actions (3); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions 
        mu_preds = self.predict_mu(x[:, 1])
        sigma_preds = self.predict_sigma(x[:, 1])
        pr_action_dist = torch.distributions.Normal(mu_preds, sigma_preds)  # predict next action given return, state, adv

        if is_train:
            # HACK HACK HACK
            if self.config.lambda1 == 0:
                pass
            elif self.config.lambda1 > 0.1:
                self.config.lambda1 = self.config.lambda1 - 0.9/4000
            else:
                self.config.lambda1 = max(0, self.config.lambda1 - 0.1/4000)
            
            # return loss
            pr_action_log_prob = pr_action_dist.log_prob(pr_actions).sum(axis=2)[attention_mask > 0].mean()
            pr_action_entropy = -pr_action_dist.log_prob(pr_action_dist.rsample((batch_size,))).mean(axis=0).sum(axis=2).mean()
            pr_action_loss = -(pr_action_log_prob + self.config.lambda1 * pr_action_entropy)

            return {"loss": pr_action_loss,
                    "pr_action_log_prob": pr_action_log_prob, 
                    "pr_action_entropy": pr_action_entropy,
                    "mu": mu_preds,
                    "sigma": sigma_preds}
        else:
            # return predictions
            if not return_dict:
                return (pr_action_dist.mean)

            return DecisionTransformerOutput(
                pr_action_preds=pr_action_dist.mean,
                # hidden_states=encoder_outputs.hidden_states,
                # last_hidden_state=encoder_outputs.last_hidden_state,
                # attentions=encoder_outputs.attentions,
            )


class TwoAgentRobustDT(DecisionTransformerModel):

    def __init__(self, config, logger=None):
        super().__init__(config)
        self.config = config
        self.logger = logger
        self.adt = AdversarialDT(config)
        self.sdt = StochasticDT(config)
        self.step = 1

    def forward(
        self,
        is_train=True,
        states=None,
        pr_actions=None,
        pr_actions_filtered=None,
        adv_actions=None,
        adv_actions_filtered=None,
        rewards=None,
        returns_to_go=None,
        returns_to_go_scaled=None,
        timesteps=None,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,):
        #
        # easier if we simply separate between training and testing straight away
        if is_train:
            # initially we train only the ADT, which will work as a sort of "teacher" for the SDT
            # but then we train the SDT for a few iterations to make sure it can learn from the ADT
            self.step += 1
            loss = 0

            adt_out = self.adt.forward(
                is_train=is_train,
                states=states,
                pr_actions=pr_actions,
                pr_actions_filtered=pr_actions_filtered,
                adv_actions=adv_actions,
                adv_actions_filtered=adv_actions_filtered,
                rewards=rewards,
                returns_to_go=returns_to_go,
                returns_to_go_scaled=(returns_to_go if returns_to_go_scaled is None else returns_to_go_scaled),
                timesteps=timesteps,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            loss += adt_out['loss']

            sdt_out = None
            if self.step > self.config.warmup_steps:
                adv_actions_hal = adv_actions_filtered.clone()
                adv_actions_hal[:, -1, :] = adt_out['adv_action_preds'][:, -1, :]

                sdt_out = self.sdt.forward(
                    is_train=is_train,
                    states=states,
                    pr_actions=pr_actions,
                    pr_actions_filtered=pr_actions_filtered,
                    adv_actions=adv_actions_hal,
                    rewards=rewards,
                    returns_to_go=returns_to_go,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions,
                    return_dict=return_dict,
                )
                loss += sdt_out['loss']

            dist_params = {}
            if adt_out is not None:
                for i in range(adt_out['mu'].shape[2]):
                    dist_params[f"mu_{i}_rtg"] =  torch.mean(adt_out['mu'][:, :, i]).item()
                    dist_params[f"sigma_{i}_rtg"] =  torch.mean(adt_out['sigma'][:, :, i]).item()

            if sdt_out is not None:
                for i in range(sdt_out['mu'].shape[2]):
                    dist_params[f"mu_{i}"] =  torch.mean(sdt_out['mu'][:, :, i]).item()
                    dist_params[f"sigma_{i}"] =  torch.mean(sdt_out['sigma'][:, :, i]).item()
        
            if self.logger is not None and (self.step == 0 or self.step % self.config.log_interval_steps == 0):
                self.logger.add_entry(
                    step=self.step,
                    hyperparams={"lambda1": self.config.lambda1, "lambda2": self.config.lambda2},
                    tr_losses={"loss": loss,
                               "rtg_loss": adt_out['rtg_loss'], 
                               "rtg_log_prob": adt_out['rtg_log_prob'], 
                               "adv_action_loss": adt_out['adv_action_loss'],
                               "pr_action_loss": 0 if sdt_out is None else sdt_out['loss'],
                               "pr_action_log_prob": 0 if sdt_out is None else sdt_out['pr_action_log_prob'], 
                               "pr_action_entropy": 0 if sdt_out is None else sdt_out['pr_action_entropy'],},
                    dist_params=dist_params,
                    log=True
                )

            return {"loss": loss}
        else:
            adt_out = self.adt.forward(
                is_train=is_train,
                states=states,
                pr_actions=pr_actions,
                pr_actions_filtered=pr_actions,
                adv_actions=adv_actions,
                adv_actions_filtered=adv_actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                returns_to_go_scaled=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            if not return_dict:
                rtg_preds, adv_action_preds = adt_out[0], adt_out[1]
            else:
                rtg_preds, adv_action_preds = adt_out.rtg_preds, adt_out.adv_action_preds

            sdt_out = self.sdt.forward(
                is_train=is_train,
                states=states,
                pr_actions=pr_actions,
                pr_actions_filtered=pr_actions,
                adv_actions=adv_action_preds,
                rewards=rewards,
                returns_to_go=rtg_preds,
                returns_to_go_scaled=rtg_preds,
                timesteps=timesteps,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            if not return_dict:
                pr_action_preds = sdt_out
            else:
                pr_action_preds = sdt_out.pr_action_preds

            if not return_dict:
                return (pr_action_preds, adv_action_preds)
            else:
                return DecisionTransformerOutput(
                    pr_action_preds=pr_action_preds,
                    adv_action_preds=adv_action_preds,
                    # hidden_states=encoder_outputs.hidden_states,
                    # last_hidden_state=encoder_outputs.last_hidden_state,
                    # attentions=encoder_outputs.attentions,
                )

    def eval(self, **kwargs):
        return ADTEvalWrapper(self)
    
    def get_actions(self, states, pr_actions, adv_actions, rewards, returns_to_go, timesteps, device, batch_size=1):
        # NOTE this implementation does not condition on past rewards
        # reshape to model input format
        states = states.reshape(batch_size, -1, self.config.state_dim)
        pr_actions = pr_actions.reshape(batch_size, -1, self.config.pr_act_dim)
        adv_actions = adv_actions.reshape(batch_size, -1, self.config.adv_act_dim)
        returns_to_go = returns_to_go.reshape(batch_size, -1, 1)
        timesteps = timesteps.reshape(batch_size, -1)

        # normalisation constants
        state_mean = torch.from_numpy(np.array(self.config.state_mean).astype(np.float32)).to(device=device)
        state_std = torch.from_numpy(np.array(self.config.state_std).astype(np.float32)).to(device=device)

        # retrieve window of observations based on context length
        states = states[:, -self.config.context_size :]
        pr_actions = pr_actions[:, -self.config.context_size :]
        adv_actions = adv_actions[:, -self.config.context_size :]
        returns_to_go = returns_to_go[:, -self.config.context_size :]
        timesteps = timesteps[:, -self.config.context_size :]

        # normalising states
        states = (states - state_mean) / state_std

        # pad all tokens to sequence length
        padlen = self.config.context_size - states.shape[1]
        attention_mask = torch.cat([torch.zeros((batch_size, padlen), device=device), torch.ones((batch_size, states.shape[1]), device=device)], dim=1).to(dtype=torch.long)
        states = torch.cat([torch.zeros((batch_size, padlen, self.config.state_dim), device=device), states], dim=1).float()
        pr_actions = torch.cat([torch.zeros((batch_size, padlen, self.config.pr_act_dim), device=device), pr_actions], dim=1).float()
        adv_actions = torch.cat([torch.zeros((batch_size, padlen, self.config.adv_act_dim), device=device), adv_actions], dim=1).float()
        returns_to_go = torch.cat([torch.zeros((batch_size, padlen, 1), device=device), returns_to_go], dim=1).float()
        timesteps = torch.cat([torch.zeros((batch_size, padlen), dtype=torch.long, device=device), timesteps], dim=1)

        # forward pass
        pr_action_preds, adv_action_preds = self.forward(
            is_train=False,
            states=states,
            pr_actions=pr_actions,
            adv_actions=adv_actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            returns_to_go_scaled=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return pr_action_preds[:, -1], adv_action_preds[:, -1]
