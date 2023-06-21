import torch

from transformers import DecisionTransformerModel, DecisionTransformerGPT2Model

from ardt_utils import DecisionTransformerOutput
from ardt_utils import ExpFunc
    

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
        # FIXME check indexing
        mu_preds = self.predict_mu(x[:, 1])  # predict next pr action dist. mean given return and state
        sigma_preds = self.predict_sigma_exp(x[:, 1])  # predict next pr action dist. sigma given return and state
        pr_action_dist = torch.distributions.Normal(mu_preds, sigma_preds)
        
        adv_action_preds = self.predict_adv_action(x[:, 2])  # predict next adv action given return, state and pr_action

        if is_train:
            # return loss
            pr_action_log_prob = -pr_action_dist.log_prob(pr_actions).mean()
            pr_action_entropy = -pr_action_dist.entropy().mean()
            pr_action_loss = pr_action_log_prob + self.config.lambda1 * pr_action_entropy

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
        