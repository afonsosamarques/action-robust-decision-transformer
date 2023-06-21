import json
import gymnasium as gym
import numpy as np
import torch

from collections import defaultdict
from huggingface_hub import login
from transformers import DecisionTransformerModel, DecisionTransformerConfig

from model.ardt_naive import SingleAgentRobustDT
from model.ardt_full import TwoAgentRobustDT

from access_tokens import WRITE_TOKEN


def get_action_no_adv(model, states, actions, rewards, returns_to_go, timesteps, device):
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

    # normlisation
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


def get_action_with_adv(model, states, pr_actions, adv_actions, rewards, returns_to_go, timesteps, device):
    # NOTE this implementation does not condition on past rewards
    # reshape to model input format
    states = states.reshape(1, -1, model.config.state_dim)
    pr_actions = pr_actions.reshape(1, -1, model.config.pr_act_dim)
    adv_actions = adv_actions.reshape(1, -1, model.config.adv_act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    # normalisation constants
    state_mean = torch.from_numpy(np.array(model.config.state_mean).astype(np.float32)).to(device=device)
    state_std = torch.from_numpy(np.array(model.config.state_std).astype(np.float32)).to(device=device)

    # retrieve window of observations based on context length
    states = states[:, -model.config.context_size :]
    pr_actions = pr_actions[:, -model.config.context_size :]
    adv_actions = adv_actions[:, -model.config.context_size :]
    returns_to_go = returns_to_go[:, -model.config.context_size :]
    timesteps = timesteps[:, -model.config.context_size :]

    # normalising states
    states = (states - state_mean) / state_std

    # pad all tokens to sequence length
    padlen = model.config.context_size - states.shape[1]
    attention_mask = torch.cat([torch.zeros(padlen, device=device), torch.ones(states.shape[1], device=device)]).to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padlen, model.config.state_dim), device=device), states], dim=1).float()
    pr_actions = torch.cat([torch.zeros((1, padlen, model.config.pr_act_dim), device=device), pr_actions], dim=1).float()
    adv_actions = torch.cat([torch.zeros((1, padlen, model.config.adv_act_dim), device=device), adv_actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padlen, 1), device=device), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padlen), dtype=torch.long, device=device), timesteps], dim=1)

    # forward pass
    pr_action_preds, adv_action_preds = model.forward(
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


def load_model(model_to_use):
    if model_to_use == "dt-halfcheetah-v2":
        return DecisionTransformerModel.from_pretrained(f"afonsosamarques/{model_to_use}", use_auth_token=True), False
    elif model_to_use == "ardt-naive-halfcheetah-v0":
        config = DecisionTransformerConfig.from_pretrained(f"afonsosamarques/{model_to_use}", use_auth_token=True)
        model = SingleAgentRobustDT(config)
        return model.from_pretrained(f"afonsosamarques/{model_to_use}", use_auth_token=True), True
    elif model_to_use == "ardt-halfcheetah-v2":
        config = DecisionTransformerConfig.from_pretrained(f"afonsosamarques/{model_to_use}", use_auth_token=True)
        model = TwoAgentRobustDT(config)
        return model.from_pretrained(f"afonsosamarques/{model_to_use}", use_auth_token=True), True
        

if __name__ == "__main__":
    login(token=WRITE_TOKEN)

    # picking environment
    envs_in_gym = {
        0: "Walker2d-v4",
        1: "HalfCheetah-v4",
    }

    default_tr_per_1000 = {
        "Walker2d-v4": 7200,
        "HalfCheetah-v4": 12000
    }

    chosen_env = envs_in_gym[1]
    env_target_per_1000 = default_tr_per_1000[chosen_env]

    # agent possibilities
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    hf_models_to_use = ["dt-halfcheetah-v2", "ardt-naive-halfcheetah-v0", "ardt-halfcheetah-v2"]

    # evaluate
    eval_dict = {}
    for model_name in hf_models_to_use:
        print("==================================================")
        print(f"Evaluating model {model_name}...")
        model, is_adv = load_model(model_name)
        model.to(device)
        eval_dict[model_name] = defaultdict(list)

        for i in range(30):
            if i % 10 == 0:
                print(f"Run number {i}...")
            
            with torch.no_grad():
                env = gym.make(chosen_env, render_mode="rgb_array")
                np.random.seed(i*3)
                torch.manual_seed(i*3)
                env.seed = i*3
                env.action_space.seed = i*3

                state, _ = env.reset()

                returns_scale = model.config.returns_scale if "returns_scale" in model.config.to_dict().keys() else 1000.0  # FIXME compatibility
                episode_return, episode_length = 0, 0
                target_return = torch.tensor(env_target_per_1000/returns_scale, device=device, dtype=torch.float32).reshape(1, 1)
                states = torch.from_numpy(state).reshape(1, model.config.state_dim).to(device=device, dtype=torch.float32)
                if is_adv:
                    pr_actions = torch.zeros((0, model.config.pr_act_dim), device=device, dtype=torch.float32)
                    adv_actions = torch.zeros((0, model.config.adv_act_dim), device=device, dtype=torch.float32)
                else:
                    actions = torch.zeros((0, model.config.act_dim), device=device, dtype=torch.float32)
                rewards = torch.zeros(0, device=device, dtype=torch.float32)
                timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

                for t in range(model.config.max_ep_len):
                    if is_adv:
                        pr_actions = torch.cat([pr_actions, torch.zeros((1, model.config.pr_act_dim), device=device)], dim=0)
                        adv_actions = torch.cat([adv_actions, torch.zeros((1, model.config.adv_act_dim), device=device)], dim=0)
                    else:
                        actions = torch.cat([actions, torch.zeros((1, model.config.act_dim), device=device)], dim=0)
                
                    rewards = torch.cat([rewards, torch.zeros(1, device=device)])

                    if is_adv:
                        pr_action, adv_action = get_action_with_adv(
                            model,
                            states,
                            pr_actions,
                            adv_actions,
                            rewards,
                            target_return,
                            timesteps,
                            device,
                        )
                        pr_actions[-1] = pr_action
                        adv_actions[-1] = adv_action
                        action = pr_action.detach().cpu().numpy()
                    else:
                        action = get_action_no_adv(
                            model,
                            states,
                            actions,
                            rewards,
                            target_return,
                            timesteps,
                            device,
                        )
                        actions[-1] = action
                        action = action.detach().cpu().numpy()

                    state, reward, done, _, _ = env.step(action)

                    cur_state = torch.from_numpy(state.astype(np.float32)).to(device=device).reshape(1, model.config.state_dim)
                    states = torch.cat([states, cur_state], dim=0)
                    rewards[-1] = reward

                    pred_return = target_return[0, -1] - (reward / returns_scale)
                    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
                    timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

                    episode_return += reward
                    episode_length += 1

                    if done or t == model.config.max_ep_len - 1:
                        eval_dict[model_name]['iter'].append(i)
                        eval_dict[model_name]['env_seed'].append(i*3)
                        eval_dict[model_name]['init_target_return'].append(target_return[0][0].item())
                        eval_dict[model_name]['final_target_return'].append(target_return[0][-1].item())
                        eval_dict[model_name]['ep_length'].append(episode_length)
                        eval_dict[model_name]['ep_return'].append(episode_return)
                        break
    
    # save eval_dict as json
    with open('./eval_outputs/output_stats.json', 'w') as f:
        json.dump(eval_dict, f)

    # show some simple statistics
    print("\n==========================================================================================\n")
    for model_name, metrics in eval_dict.items():
        print(f"================== {model_name} ==================")
        print(f"Initial target returns | Avg: {np.round(np.mean(metrics['init_target_return']), 4)} | Std: {np.round(np.std(metrics['init_target_return']), 4)} | Min: {np.round(np.min(metrics['init_target_return']), 4)} | Median: {np.round(np.median(metrics['init_target_return']), 4)} | Max: {np.round(np.max(metrics['init_target_return']), 4)}")
        print(f"Final target returns | Avg: {np.round(np.mean(metrics['final_target_return']), 4)} | Std: {np.round(np.std(metrics['final_target_return']), 4)} | Min: {np.round(np.min(metrics['final_target_return']), 4)} | Median: {np.round(np.median(metrics['final_target_return']), 4)} | Max: {np.round(np.max(metrics['final_target_return']), 4)}")
        print(f"Episode lengths | Avg: {np.round(np.mean(metrics['ep_length']), 4)} | Std: {np.round(np.std(metrics['ep_length']), 4)} | Min: {np.round(np.min(metrics['ep_length']), 4)} | Median: {np.round(np.median(metrics['ep_length']), 4)} | Max: {np.round(np.max(metrics['ep_length']), 4)}")
        print(f"Episode returns | Avg: {np.round(np.mean(metrics['ep_return']), 4)} | Std: {np.round(np.std(metrics['ep_return']), 4)} | Min: {np.round(np.min(metrics['ep_return']), 4)} | Median: {np.round(np.median(metrics['ep_return']), 4)} | Max: {np.round(np.max(metrics['ep_return']), 4)}")
        print("\n")
