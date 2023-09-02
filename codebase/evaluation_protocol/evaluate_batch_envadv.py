import json
import os

import gymnasium as gym
import numpy as np
import torch

from collections import defaultdict
from requests.exceptions import HTTPError

from .config_utils import load_model
from .helpers import set_seed_everywhere, find_root_dir, scrappy_print_eval_dict
        

MULTIPLIERS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]


def vary_body_mass(env, multiplier=1.0):
    mb = env.model.body_mass
    env.model.body_mass = np.array(mb) * multiplier
    return env


def vary_friction(env, multiplier=1.0):
    mb = env.model.geom_friction
    env.model.geom_friction = np.array(mb) * multiplier
    return env
    

def evaluate(
        model_name, 
        model_type,
        env_name,
        env_type,
        env_steps,
        eval_iters,
        eval_target,
        is_adv_eval=False,
        run_suffix='',
        record_data=False,
        verbose=False,
        model_path=None,
        hf_project=None,
        device=torch.device('cpu'),
    ):
    # load model
    try:
        model, is_adv_model = load_model(model_type, model_name, model_path=(model_path if model_path is not None else hf_project + f"/{model_name}"))
    except HTTPError as e:
        print(f"Could not load model {model_name} from repo.")
    model.to(device)
    model = model.eval(mdp_type=('pr_mdp' if 'pr_mdp' in model_path else ('nr_mdp' if 'nr_mdp' in model_path else None)))

    for p in [0, 1]:
        variation_type = 'body-mass' if p == 0 else 'friction'
        variation_func = vary_body_mass if p == 0 else vary_friction

        for k in range(len(MULTIPLIERS)):
            # setting up environments for parallel runs
            envs = []
            start_states = []
            for n in range(eval_iters):
                env = gym.make(env_name)
                set_seed_everywhere(n, env)
                env = variation_func(env, multiplier=MULTIPLIERS[k])
                state, _ = env.reset()
                envs.append(env)
                start_states.append(state)
            start_states = np.array(start_states)
            
            # evaluation loop
            print("\n================================================")
            print(f"Evaluating model {model_name} on environment {env_type} varying {variation_type} with multiplier {MULTIPLIERS[k]}.")

            episode_returns = np.zeros(eval_iters)
            episode_lengths = np.zeros(eval_iters)
            dones = np.zeros(eval_iters).astype(bool)
            eval_dict = defaultdict(list)
            data_dict = defaultdict(list)
            for i in range(eval_iters):
                data_dict['observations'].append([])
                data_dict['pr_actions'].append([])
                data_dict['adv_actions'].append([])
                data_dict['rewards'].append([])
                data_dict['dones'].append([])
            
            with torch.no_grad():
                try:
                    return_target = model.model.config.max_ep_return
                except Exception:
                    return_target = eval_target

                model.new_batch_eval(start_states=start_states, eval_target=return_target)

                # run episode
                for t in range(env_steps):
                    pr_actions, est_adv_actions = model.get_batch_actions(states=start_states)

                    states = np.zeros_like(start_states)
                    rewards = np.zeros(eval_iters)
                    for i, env in enumerate(envs):
                        if dones[i] == False:
                            state, reward, done, trunc, _ = env.step(pr_actions[i])

                            states[i] = state
                            rewards[i] = reward
                            dones[i] = done or trunc
                            episode_returns[i] += reward
                            episode_lengths[i] += 1

                            if record_data:
                                # log episode data
                                data_dict['observations'][i].append(state)
                                data_dict['pr_actions'][i].append(pr_actions[i])
                                data_dict['est_adv_actions'][i].append(est_adv_actions[i])
                                data_dict['rewards'][i].append(reward)
                                data_dict['dones'][i].append(done)

                            # finish and log episode
                            if done or trunc or t == env_steps - 2:
                                # log episode outcome
                                eval_dict['iter'].append(i)
                                eval_dict['env_seed'].append(i)
                                eval_dict['init_target_return'].append(eval_target)
                                eval_dict['ep_length'].append(episode_lengths[i])
                                eval_dict['ep_return'].append(episode_returns[i])

                    model.update_batch_history(
                        pr_actions=pr_actions, 
                        adv_actions=est_adv_actions,
                        states=states, 
                        rewards=rewards,
                        timestep=t
                    )

                    if t == env_steps - 2:
                        break
            
            # show some simple statistics
            if verbose:
                scrappy_print_eval_dict(model_name, eval_dict)

            # save eval_dict as json
            dir_path = f'{find_root_dir()}/eval-outputs{run_suffix}/{env_type}/{model_name}/env-adv-{variation_type}'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(f"{dir_path}/{MULTIPLIERS[k]}.json", 'w') as f:
                json.dump(eval_dict, f)
