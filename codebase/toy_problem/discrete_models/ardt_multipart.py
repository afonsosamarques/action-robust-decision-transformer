import numpy as np
import torch

from transformers import DecisionTransformerModel, DecisionTransformerGPT2Model

from .model_utils import DecisionTransformerOutput, ADTEvalWrapper


class ReturnsDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = torch.nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_pr_action = torch.nn.Linear(config.pr_act_dim, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_ln = torch.nn.LayerNorm(config.hidden_size)

        self.predict_rtg = torch.nn.Linear(config.hidden_size, len(config.discrete_returns))

        self.post_init()
        
    def one_hot_encode(self, targets, all_possible_returns):
        return_to_index = {ret: i for i, ret in enumerate(all_possible_returns)}
        target_indices = torch.tensor([return_to_index[target.item()] for target in targets], dtype=torch.long)
        one_hot_targets = torch.nn.functional.one_hot(target_indices, num_classes=len(all_possible_returns))
        return one_hot_targets.to(device=targets.device, dtype=torch.float32)
    
    def predict_values(self, probabilities, all_possible_returns):
        all_possible_returns = np.array(all_possible_returns)
        _, predicted_indices = torch.max(probabilities, dim=-1)
        predicted_values = all_possible_returns[predicted_indices.cpu().numpy()]
        return torch.tensor(predicted_values, device=probabilities.device, dtype=torch.float32)

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
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states)
        pr_action_embeddings = self.embed_pr_action(pr_actions)
        returns_embeddings = self.embed_return(returns_to_go)

        # time embeddings are treated similar to positional embeddings
        state_embeddings += time_embeddings
        pr_action_embeddings += time_embeddings
        returns_embeddings += time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, a'_1, R_2, s_2, a_2, a'_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((state_embeddings, pr_action_embeddings, returns_embeddings), dim=1)
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
        # states (0), pr_actions (1), returns (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        rtg_logits = self.predict_rtg(x[:, 1])
        rtg_preds = self.predict_values(torch.nn.functional.softmax(rtg_logits, dim=-1), self.config.discrete_returns).reshape(returns_to_go.shape[0], returns_to_go.shape[1], -1)

        if is_train:
            one_hot_rtg = self.one_hot_encode(returns_to_go.view(-1), self.config.discrete_returns).reshape(returns_to_go.shape[0], returns_to_go.shape[1], -1)
            target_indices = torch.argmax(one_hot_rtg, dim=-1)
            ce = torch.nn.functional.cross_entropy(rtg_logits.view(-1, len(self.config.discrete_returns)), target_indices.view(-1), reduction='none')
            er_mask = (returns_to_go > rtg_preds).view(-1)
            rdt_loss = (0.01 * ce * er_mask) + (0.99 * ce * ~er_mask)
            rdt_loss = torch.mean(rdt_loss)
        else:
            # return predictions
            if not return_dict:
                return (rtg_preds)

            return DecisionTransformerOutput(
                rtg_preds=rtg_preds,
                # hidden_states=encoder_outputs.hidden_states,
                # last_hidden_state=encoder_outputs.last_hidden_state,
                # attentions=encoder_outputs.attentions,
            )
    

class ProtagonistDT(DecisionTransformerModel):
    def __init__(self, config, logger=None):
        super().__init__(config)
        self.config = config
        self.logger = logger

        self.hidden_size = config.hidden_size

        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = torch.nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_pr_action = torch.nn.Linear(config.pr_act_dim, config.hidden_size)
        self.embed_ln = torch.nn.LayerNorm(config.hidden_size)

        self.predict_pr_action = torch.nn.Sequential(
            *([torch.nn.Linear(config.hidden_size, config.pr_act_dim)] + ([torch.nn.Sigmoid()]))
        )

        self.post_init()

    def forward(
        self,
        is_train=True,
        states=None,
        pr_actions=None,
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
        time_embeddings = self.embed_timestep(timesteps)
        returns_embeddings = self.embed_return(returns_to_go)
        state_embeddings = self.embed_state(states)
        pr_action_embeddings = self.embed_pr_action(pr_actions)

        # time embeddings are treated similar to positional embeddings
        returns_embeddings += time_embeddings
        state_embeddings += time_embeddings
        pr_action_embeddings += time_embeddings

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
        # returns (0), states (1), pr_actions (2) or adv_actions (3); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get own predictions
        pr_action_preds = self.predict_pr_action(x[:, 1])  # predict next pr action given return and state

        if is_train:
            # compute loss
            pr_action_preds = pr_action_preds.reshape(-1, self.config.pr_act_dim)
            pr_action_targets = pr_actions.reshape(-1, self.config.pr_act_dim)
            pr_bce = torch.nn.functional.binary_cross_entropy(pr_action_preds, pr_action_targets, reduction='none')
            pr_action_loss = (pr_bce).mean()
    
            return {"pdt_loss": pr_action_loss}
        else:
            # return predictions
            pr_action_preds = (pr_action_preds > 0.5).to(torch.int32).reshape(batch_size, -1, self.config.pr_act_dim)

            if not return_dict:
                return (pr_action_preds)

            return DecisionTransformerOutput(
                pr_action_preds=pr_action_preds,
                # hidden_states=encoder_outputs.hidden_states,
                # last_hidden_state=encoder_outputs.last_hidden_state,
                # attentions=encoder_outputs.attentions,
            )


class MultipartARDT(DecisionTransformerModel):
    def __init__(self, config, logger=None):
        super().__init__(config)
        self.config = config
        self.logger = logger
        self.rdt = ReturnsDT(config)
        self.pdt = ProtagonistDT(config)
        self.step = 0

    def forward(
        self,
        is_train=True,
        states=None,
        pr_actions=None,
        adv_actions=None,
        rewards=None,
        returns_to_go=None,
        next_returns_to_go=None,
        timesteps=None,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,):
        #
        # easier if we simply separate between training and testing straight away
        if is_train:
            self.step += 1
            loss = 0
            rdt_out = None
            pdt_out = None

            rdt_out = self.rdt.forward(
                is_train=is_train,
                states=states,
                pr_actions=pr_actions,
                adv_actions=adv_actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            loss += rdt_out['rdt_loss']

            if self.step >= self.config.warmup_steps:
                pdt_out = self.pdt.forward(
                    is_train=is_train,
                    states=states,
                    pr_actions=pr_actions,
                    rewards=rewards,
                    returns_to_go=rdt_out['rdt_preds'],
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions,
                    return_dict=return_dict,
                )
                loss += pdt_out['pdt_loss']

            return {"loss": loss}
        else:
            pdt_out = self.pdt.forward(
                is_train=is_train,
                states=states,
                pr_actions=pr_actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            if not return_dict:
                pr_action_preds = pdt_out
            else:
                pr_action_preds = pdt_out.pr_action_preds

            if not return_dict:
                return (pr_action_preds)
            else:
                return DecisionTransformerOutput(
                    pr_action_preds=pr_action_preds
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
        pr_action_preds = self.forward(
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
         
        return pr_action_preds[:, -1], pr_action_preds[:, -1] * 0.0
