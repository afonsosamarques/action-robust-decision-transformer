import numpy as np
import torch

from transformers import DecisionTransformerModel, DecisionTransformerGPT2Model

from .ardt_utils import DecisionTransformerOutput, DTEvalWrapper


class SimpleRobustDT(DecisionTransformerModel):
    def __init__(self, config, logger=None):
        super().__init__(config)
        self.config = config
        self.logger = logger
        self.step = 0

        self.hidden_size = config.hidden_size

        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = torch.nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_pr_action = torch.nn.Linear(config.pr_act_dim, config.hidden_size)
        self.embed_adv_action = torch.nn.Linear(config.adv_act_dim, config.hidden_size)
        self.embed_ln = torch.nn.LayerNorm(config.hidden_size)

        self.predict_pr_action = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, config.pr_act_dim)] + ([torch.nn.Tanh()] if config.action_tanh else []))
        )
        self.predict_adv_action = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, config.adv_act_dim)] + ([torch.nn.Tanh()] if config.action_tanh else []))
        )

        self.post_init()

    def forward(
        self,
        is_train=True,
        pred_adv=None,
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
        pr_action_preds = self.predict_pr_action(x[:, 1])  # predict next pr action given return and state
        adv_action_preds = self.predict_adv_action(x[:, 2])  # predict next adv action given return, state and pr_action

        if is_train:
            self.step += 1

            # return loss
            pr_action_preds = pr_action_preds.reshape(-1, self.config.pr_act_dim)[attention_mask.reshape(-1) > 0]
            pr_action_targets = pr_actions.reshape(-1, self.config.pr_act_dim)[attention_mask.reshape(-1) > 0]
            pr_action_loss = torch.mean((pr_action_preds - pr_action_targets) ** 2)

            adv_action_preds = adv_action_preds.reshape(-1, self.config.adv_act_dim)[attention_mask.reshape(-1) > 0]
            adv_action_targets = adv_actions.reshape(-1, self.config.adv_act_dim)[attention_mask.reshape(-1) > 0]
            adv_action_loss = self.config.lambda2 * torch.mean((adv_action_preds - adv_action_targets) ** 2)

            if self.logger is not None:
                self.logger.add_entry(
                    step=self.step,
                    hyperparams={"lambda2": self.config.lambda2},
                    tr_losses={"loss": (pr_action_loss + adv_action_loss).item(), 
                               "pr_action_loss": pr_action_loss.item(), 
                               "adv_action_loss": adv_action_loss.item()},
                    dist_params=None,
                    log=True
                )

            return {"loss": pr_action_loss + adv_action_loss}
        else:
            # return predictions
            if not return_dict:
                return (pr_action_preds, adv_action_preds)

            return DecisionTransformerOutput(
                pr_action_preds=pr_action_preds,
                adv_action_preds=adv_action_preds,
                # hidden_states=encoder_outputs.hidden_states,
                # last_hidden_state=encoder_outputs.last_hidden_state,
                # attentions=encoder_outputs.attentions,
            )
        
    def eval(self, **kwargs):
        return DTEvalWrapper(self)
    
    def get_action(self, states, pr_actions, adv_actions, rewards, returns_to_go, timesteps, device):
        # NOTE this implementation does not condition on past rewards
        # reshape to model input format
        states = states.reshape(1, -1, self.config.state_dim)
        pr_actions = pr_actions.reshape(1, -1, self.config.pr_act_dim)
        adv_actions = adv_actions.reshape(1, -1, self.config.adv_act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

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
        attention_mask = torch.cat([torch.zeros(padlen, device=device), torch.ones(states.shape[1], device=device)]).to(dtype=torch.long).reshape(1, -1)
        states = torch.cat([torch.zeros((1, padlen, self.config.state_dim), device=device), states], dim=1).float()
        pr_actions = torch.cat([torch.zeros((1, padlen, self.config.pr_act_dim), device=device), pr_actions], dim=1).float()
        adv_actions = torch.cat([torch.zeros((1, padlen, self.config.adv_act_dim), device=device), adv_actions], dim=1).float()
        returns_to_go = torch.cat([torch.zeros((1, padlen, 1), device=device), returns_to_go], dim=1).float()
        timesteps = torch.cat([torch.zeros((1, padlen), dtype=torch.long, device=device), timesteps], dim=1)

        # forward pass
        pr_action_preds, adv_action_preds = self.forward(
            is_train=False,
            states=states,
            pr_actions=pr_actions,
            adv_actions=adv_actions,
            rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

        return pr_action_preds[0, -1], adv_action_preds[0, -1]
