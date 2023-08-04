import numpy as np
import torch

from transformers import DecisionTransformerModel, DecisionTransformerGPT2Model

from .ardt_utils import DecisionTransformerOutput, ADTEvalWrapper
from .ardt_utils import StdReturnSquashFunc, StdSquashFunc, ExpFunc
from .ardt_utils import initialise_weights


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

        self.predict_return = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, 1)])
        )
        self.predict_adv_action = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, config.adv_act_dim)] + ([torch.nn.Sigmoid()]))
        )

        self.post_init()
        self.predict_mu.apply(initialise_weights)
        self.predict_sigma.apply(initialise_weights)
        self.predict_adv_action.apply(initialise_weights)

    def forward(
        self,
        is_train=True,
        pred_adv=True,
        states=None,
        pr_actions=None,
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
        pr_action_embeddings = self.embed_pr_action(pr_actions)
        adv_action_embeddings = self.embed_adv_action(adv_actions)
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
        rtg_preds = self.predict_return(x[:, 1])  # predict next return given return and state
        adv_action_preds = self.predict_adv_action(x[:, 2])  # predict next action given state and pr_action

        if is_train:
            # return loss
            rtg_loss = torch.mean((rtg_preds.reshape(-1) - returns_to_go.reshape(-1)) ** 2)

            adv_action_preds = adv_action_preds.reshape(-1, self.config.adv_act_dim)
            adv_action_targets = adv_actions.reshape(-1, self.config.adv_act_dim)
            adv_action_loss = torch.nn.functional.binary_cross_entropy(adv_action_preds, adv_action_targets)
                
            return {"loss": rtg_loss + adv_action_loss,
                    "rtg_log_prob": 0, 
                    "rtg_loss": rtg_loss,
                    "adv_action_loss": adv_action_loss,
                    "rtg_preds": rtg_preds}
        else:
            adv_action_preds = (adv_action_preds > 0.5).astype(np.int32).reshape(-1, self.config.adv_act_dim)

            if not return_dict:
                return (rtg_preds, adv_action_preds)

            return DecisionTransformerOutput(
                rtg_preds=rtg_preds,
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
        self.embed_pr_action = torch.nn.Linear(config.pr_act_dim, config.hidden_size)
        self.embed_ln = torch.nn.LayerNorm(config.hidden_size)

        self.predict_action = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, config.pr_act_dim)] + ([torch.nn.Sigmoid()]))
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
        pr_action_embeddings = self.embed_pr_action(pr_actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings += time_embeddings
        pr_action_embeddings += time_embeddings
        returns_embeddings += time_embeddings

        # this makes the sequence look like (R'_1, s_1, a_1, R'_2, s_2, a_2, ...)
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
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions 
        action_preds = self.predict_mu(x[:, 1])  # predict next action dist. mean given returns and states

        if is_train:
            pr_action_preds = pr_action_preds.reshape(-1, self.config.pr_act_dim)
            pr_action_targets = pr_actions.reshape(-1, self.config.pr_act_dim)
            pr_action_loss = torch.nn.functional.binary_cross_entropy(pr_action_preds, pr_action_targets)

            return {"loss": pr_action_loss,
                    "action_preds": action_preds}
        else:
            # return predictions
            pr_action_preds = (pr_action_preds > 0.5).astype(np.int32).reshape(-1, self.config.pr_act_dim)

            if not return_dict:
                return (pr_action_preds)

            return DecisionTransformerOutput(
                pr_action_preds=pr_action_preds,
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
        self.post_init()

    def forward(
        self,
        is_train=True,
        states=None,
        pr_actions=None,
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
        # easier if we simply separate between training and testing straight away
        if is_train:
            # initially we train only the ADT, which will work as a sort of "teacher" for the SDT
            # but then we train the SDT for a few iterations to make sure it can learn from the ADT
            self.step += 1
            loss = 0

            adt_out = self.adt.forward(
                is_train=is_train,
                pred_adv=(self.step > self.config.warmup_steps),
                states=states,
                pr_actions=pr_actions,
                adv_actions=adv_actions,
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
                sdt_out = self.sdt.forward(
                    is_train=is_train,
                    states=states,
                    pr_actions=pr_actions,
                    adv_actions=adv_actions,
                    rewards=rewards,
                    returns_to_go=adt_out['rtg_preds'],
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions,
                    return_dict=return_dict,
                )
                loss += sdt_out['loss']

            dist_params = {}
            # if adt_out is not None:
            #     for i in range(adt_out['mu'].shape[2]):
            #         dist_params[f"mu_{i}_rtg"] =  torch.mean(adt_out['mu'][:, :, i]).item()
            #         dist_params[f"sigma_{i}_rtg"] =  torch.mean(adt_out['sigma'][:, :, i]).item()

            # if sdt_out is not None:
            #     for i in range(sdt_out['mu'].shape[2]):
            #         dist_params[f"mu_{i}"] =  torch.mean(sdt_out['mu'][:, :, i]).item()
            #         dist_params[f"sigma_{i}"] =  torch.mean(sdt_out['sigma'][:, :, i]).item()
        
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
                adv_actions=adv_actions,
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
                adv_actions=adv_actions,
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
        states = states.reshape(batch_size, -1, self.config.state_dim)
        pr_actions = pr_actions.reshape(batch_size, -1, self.config.pr_act_dim)
        adv_actions = adv_actions.reshape(batch_size, -1, self.config.adv_act_dim)
        returns_to_go = returns_to_go.reshape(batch_size, -1, 1)
        timesteps = timesteps.reshape(batch_size, -1)

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
            attention_mask=torch.ones((batch_size, states.shape[1]), device=device, dtype=torch.long),
            return_dict=False,
        )

        return pr_action_preds[:, -1], adv_action_preds[:, -1]
