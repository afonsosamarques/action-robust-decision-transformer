import json
import os

import gymnasium as gym
import numpy as np
import torch

from collections import defaultdict
from requests.exceptions import HTTPError

from datasets import Dataset

from .config_utils import load_model
from .helpers import set_seed_everywhere, find_root_dir, scrappy_print_eval_dict


def evaluate(
        pr_model_name, 
        pr_model_type,
        pr_model_path,
        adv_model_name,
        adv_model_type,
        adv_model_path,
        omni_adv,
        env_name,
        env_type,
        env_steps,
        eval_iters,
        eval_target,
        run_suffix='',
        record_data=False,
        verbose=False,
        hf_project="afonsosamarques",
        device=torch.device('cpu'),
    ):
    # load models
    try:
        pr_model, is_adv_pr_model = load_model(pr_model_type, pr_model_name, model_path=(pr_model_path if pr_model_path is not None else hf_project + f"/{pr_model_name}"))
    except HTTPError as e:
        print(f"Could not load protagonist model {pr_model_name} from repo.")
        raise e
    pr_model.to(device)
    pr_model = pr_model.eval(mdp_type=('pr_mdp' if 'pr_mdp' in pr_model_path else ('nr_mdp' if 'nr_mdp' in pr_model_path else None)))

    try:
        adv_model, is_adv_adv_model = load_model(adv_model_type, adv_model_name, model_path=(adv_model_path if adv_model_path is not None else hf_project + f"/{adv_model_name}"))
    except HTTPError as e:
        print(f"Could not load adversary model {adv_model_name} from repo.")
        raise e
    adv_model.to(device)
    adv_model = adv_model.eval(mdp_type=('pr_mdp' if 'pr_mdp' in adv_model_path else ('nr_mdp' if 'nr_mdp' in adv_model_path else None)))
    if not is_adv_adv_model:
        raise RuntimeError("Adversarial agent does not include any adversarial element, therefore cannot be used as such.")

    # evaluation loop
    print("\n================================================")
    print(f"Evaluating protagonist model {pr_model_name} on environment {env_type} against adversarial model {adv_model_name}.")

    eval_dict = defaultdict(list)
    data_dict = defaultdict(list)
    run_idx = 0
    n_runs = 0

    while True:
        run_idx += 1
        if n_runs >= eval_iters:
            break

        with torch.no_grad():
            # set up environment for run
            env = gym.make(env_name)
            set_seed_everywhere(run_idx, env)

            # set up episode variables
            episode_return, episode_length = 0, 0
            episode_data = defaultdict(list)
            
            # reset environment
            state, _ = env.reset()
            pr_model.new_eval(start_state=state, eval_target=eval_target)
            adv_model.new_eval(start_state=state, eval_target=0)

            # run episode
            for t in range(env_steps):
                pr_action, est_adv_action = pr_model.get_action(state=state)
                if omni_adv:
                    _, adv_action = adv_model.get_action(state=state, pr_action=pr_action)
                else:
                    _, adv_action = adv_model.get_action(state=state)
                
                if t == 1:
                    print(f"Starting episode {run_idx}. Checking that adversary is active. Adversarial action: ", adv_action)

                cumul_action = (pr_action + adv_action).squeeze()
                state, reward, done, trunc, _ = env.step(cumul_action)

                pr_model.update_history(
                    pr_action=pr_action, 
                    adv_action=est_adv_action,
                    state=state, 
                    reward=reward,
                    timestep=t
                )
                adv_model.update_history(
                    pr_action=pr_action, 
                    adv_action=adv_action, 
                    state=state, 
                    reward=reward,
                    timestep=t
                )

                episode_return += reward
                episode_length += 1

                if record_data:
                    episode_data['observations'].append(state)
                    episode_data['pr_actions'].append(pr_action)
                    episode_data['adv_actions'].append(adv_action)
                    episode_data['rewards'].append(reward)
                    episode_data['dones'].append(done)

                # finish and log episode
                if done or trunc or t == env_steps - 1:
                    n_runs += 1
                    # log episode outcome
                    eval_dict['iter'].append(run_idx)
                    eval_dict['env_seed'].append(run_idx)
                    eval_dict['init_target_return'].append(eval_target)
                    eval_dict['ep_length'].append(episode_length)
                    eval_dict['ep_return'].append(episode_return)
                    # log episode data
                    if record_data:
                        data_dict['observations'].append(episode_data['observations'])
                        data_dict['pr_actions'].append(episode_data['pr_actions'])
                        data_dict['adv_actions'].append(episode_data['adv_actions'])
                        data_dict['rewards'].append(episode_data['rewards'])
                        data_dict['dones'].append(episode_data['dones'])
                    break
    
    # show some simple statistics
    if verbose:
        scrappy_print_eval_dict(pr_model_name, eval_dict, other_model_name=adv_model_name)

    # save eval_dict as json
    dir_path = f'{find_root_dir()}/eval-outputs{run_suffix}/{pr_model_name}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(f"{dir_path}/{adv_model_name}{'-omni' if omni_adv else ''}.json", 'w') as f:
        json.dump(eval_dict, f)

    # save data_dict as hf dataset
    if record_data:
        data_ds = Dataset.from_dict(data_dict)
        data_ds.save_to_disk(f'{find_root_dir()}/datasets/{pr_model_name}_{adv_model_name}_eval_{env_type}')
