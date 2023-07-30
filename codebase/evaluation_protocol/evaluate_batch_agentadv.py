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

    # setting up environments for parallel runs
    envs = []
    start_states = []
    for n in range(eval_iters):
        env = gym.make(env_name)
        set_seed_everywhere(n, env)
        state, _ = env.reset()
        envs.append(env)
        start_states.append(state)
    start_states = np.array(start_states)

    # evaluation loop
    print("\n================================================")
    print(f"Evaluating protagonist model {pr_model_name} on environment {env_type} against adversarial model {adv_model_name}.")

    episode_returns = np.zeros(eval_iters)
    episode_lengths = np.zeros(eval_iters)
    eval_dict = defaultdict(list)
    data_dict = defaultdict(list)
    for i in range(eval_iters):
        data_dict['observations'].append([])
        data_dict['pr_actions'].append([])
        data_dict['adv_actions'].append([])
        data_dict['rewards'].append([])
        data_dict['dones'].append([])

    with torch.no_grad():
        pr_model.new_batch_eval(start_states=start_states, eval_target=eval_target)
        adv_model.new_batch_eval(start_states=start_states, eval_target=eval_target)
        
        # run episode
        for t in range(env_steps):
            pr_actions, est_adv_actions = pr_model.get_batch_actions(states=start_states)
            if omni_adv:
                _, adv_actions = adv_model.get_batch_actions(states=start_states, pr_actions=pr_actions)
            else:
                _, adv_actions = adv_model.get_batch_actions(states=start_states)
            
            if t == 1 and adv_model_name not in ['zero', 'zeroagent']:
                print(f"Starting timestep {t}. Checking that adversary is active. Adversarial action example: ", adv_actions[0])
            elif t == 1 and adv_model_name in ['zero', 'zeroagent']:
                print(f"Starting timestep {t}. Checking that adversary is not active. Adversarial action example: ", adv_actions[0])

            cumul_actions = (pr_actions + adv_actions)
            states = np.zeros_like(start_states)
            rewards = np.zeros(eval_iters)
            for i, env in enumerate(envs):
                # FIXME does not deal with possibility that they might end at different times!
                state, reward, done, trunc, _ = env.step(cumul_actions[i])

                states[i] = state
                rewards[i] = reward
                episode_returns[i] += reward
                episode_lengths[i] += 1

                if record_data:
                    # log episode data
                    data_dict['observations'][i].append(state)
                    data_dict['pr_actions'][i].append(pr_actions[i])
                    data_dict['adv_actions'][i].append(adv_actions[i])
                    data_dict['rewards'][i].append(reward)
                    data_dict['dones'][i].append(done)

                # finish and log episode
                if done or trunc or t == env_steps - 1:
                    # log episode outcome
                    eval_dict['iter'].append(i)
                    eval_dict['env_seed'].append(i)
                    eval_dict['init_target_return'].append(eval_target)
                    eval_dict['ep_length'].append(episode_lengths[i])
                    eval_dict['ep_return'].append(episode_returns[i])

            pr_model.update_batch_history(
                pr_actions=pr_actions, 
                adv_actions=est_adv_actions,
                states=states, 
                rewards=rewards,
                timestep=t
            )

            adv_model.update_batch_history(
                pr_actions=pr_actions, 
                adv_actions=adv_actions, 
                states=start_states, 
                rewards=rewards,
                timestep=t
            )

            if t == env_steps - 1:
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
