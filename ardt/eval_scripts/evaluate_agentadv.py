import json

import gymnasium as gym
import numpy as np
import torch

from collections import defaultdict
from requests.exceptions import HTTPError

from utils.config_utils import load_model
from utils.helpers import set_seed_everywhere, find_root_dir, scrappy_print_eval_dict


def evaluate(
        pr_model_name, 
        pr_model_type,
        adv_model_name,
        adv_model_type,
        env_name,
        env_type,
        eval_iters,
        eval_target,
        hf_project="afonsosamarques",
        run_suffix='',
        verbose=False,
        device='cpu',
        model_path_local=None
    ):
    # load models
    try:
        pr_model, _ = load_model(pr_model_type, pr_model_name, hf_project=hf_project, model_path_local=model_path_local)
    except HTTPError as e:
        print(f"Could not load protagonist model {pr_model_name} from repo.")
    pr_model.to(device)

    try:
        adv_model, _ = load_model(adv_model_type, adv_model_name, hf_project=hf_project, model_path_local=model_path_local)
    except HTTPError as e:
        print(f"Could not load protagonist model {adv_model_name} from repo.")
    adv_model.to(device)
    
    # evaluation loop
    print("\n================================================")
    print(f"Evaluating protagonist model {pr_model_name} on environment {env_type} against adversarial model {adv_model_name}.")

    eval_dict = defaultdict(list)
    run_idx = 0
    n_runs = 0

    while True:
        run_idx += 1
        if n_runs >= eval_iters:
            break
        if (run_idx % min(5, eval_iters/2)) == 0:
            print(f"Run number {run_idx}...")
        
        with torch.no_grad():
            # set up environment for run
            env = gym.make(env_name)
            set_seed_everywhere(run_idx, env)

            # set up adversary action space
            if hasattr(adv_model, 'config') and hasattr(adv_model.config, 'adv_act_dim'):
                adv_act_dim = adv_model.config.adv_act_dim
            else:
                # for now we just assume it's the same as that of the protagonist model
                adv_act_dim = pr_model.config.pr_act_dim
            
            # reset environment
            state, _ = env.reset()

            # set up episode variables
            episode_return, episode_length = 0, 0
            returns_scale = pr_model.config.returns_scale if 'returns_scale' in pr_model.config.to_dict().keys() else pr_model.config.scale  # FIXME backwards compatibility
            target_return = torch.tensor(eval_target/returns_scale, device=device, dtype=torch.float32).reshape(1, 1)
            states = torch.from_numpy(state).reshape(1, pr_model.config.state_dim).to(device=device, dtype=torch.float32)
            pr_actions = torch.zeros((0, pr_model.config.pr_act_dim), device=device, dtype=torch.float32)
            adv_actions = torch.zeros((0, adv_act_dim), device=device, dtype=torch.float32)
            rewards = torch.zeros(0, device=device, dtype=torch.float32)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

            # run episode
            for t in range(pr_model.config.max_ep_len):
                pr_actions = torch.cat([pr_actions, torch.zeros((1, pr_model.config.pr_act_dim), device=device)], dim=0)
                adv_actions = torch.cat([adv_actions, torch.zeros((1, adv_act_dim), device=device)], dim=0)
            
                rewards = torch.cat([rewards, torch.zeros(1, device=device)])

                pr_action, _ = pr_model.get_action(
                    states,
                    pr_actions,
                    adv_actions,
                    rewards,
                    target_return,
                    timesteps,
                    device,
                )
                pr_actions[-1] = pr_action

                _, adv_action = adv_model.get_action(
                    states,
                    pr_actions,
                    adv_actions,
                    rewards,
                    target_return,
                    timesteps,
                    device,
                )
                adv_actions[-1] = adv_action

                # # FIXME from rllab
                # class temp_action(object): pro=None; adv=None;
                # cum_a = temp_action()
                # cum_a.pro = pr_action.detach().cpu().numpy()
                # cum_a.adv = adv_action.detach().cpu().numpy()
                # state, reward, done, _ = env.step(cum_a)

                # FIXME for now we will just sum the two actions... (assuming same action space)
                cumul_action = (pr_action + adv_action).detach().cpu().numpy()
                state, reward, done, _, _ = env.step(cumul_action)

                cur_state = torch.from_numpy(state.astype(np.float32)).to(device=device).reshape(1, pr_model.config.state_dim)
                states = torch.cat([states, cur_state], dim=0)
                rewards[-1] = reward

                pred_return = target_return[0, -1] - (reward / returns_scale)
                target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
                timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

                episode_return += reward
                episode_length += 1

                # finish and log episode
                if done or t == pr_model.config.max_ep_len - 1:
                    n_runs += 1
                    eval_dict['iter'].append(run_idx)
                    eval_dict['env_seed'].append(run_idx)
                    eval_dict['init_target_return'].append(target_return[0][0].item())
                    eval_dict['final_target_return'].append(target_return[0][-1].item())
                    eval_dict['ep_length'].append(episode_length)
                    eval_dict['ep_return'].append(episode_return / returns_scale)
                    break
    
    # show some simple statistics
    if verbose:
        scrappy_print_eval_dict(pr_model_name, eval_dict, other_model_name=adv_model_name)

    # save eval_dict as json
    with open(f'{find_root_dir()}/eval-outputs{run_suffix}/{pr_model_name}/{adv_model_name}.json', 'w') as f:
        json.dump(eval_dict, f)
