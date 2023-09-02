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
    
    # evaluation loop
    print("\n================================================")
    print(f"Evaluating model {model_name} on environment {env_type}.")

    for p in [0, 1]:
        variation_type = 'body-mass' if p == 0 else 'friction'
        variation_func = vary_body_mass if p == 0 else vary_friction

        for k in range(len(MULTIPLIERS)):
            eval_dict = defaultdict(list)

            for n in range(eval_iters):

                with torch.no_grad():
                    # set up environment for run
                    env = gym.make(env_name)
                    set_seed_everywhere(n, env)
                    if is_adv_eval:
                        env = variation_func(env, multiplier=MULTIPLIERS[k])

                    # set up episode variables
                    episode_return, episode_length = 0, 0
                    
                    # reset environment
                    try:
                        return_target = model.model.config.max_ep_return
                    except Exception:
                        return_target = eval_target

                    state, _ = env.reset()
                    model.new_eval(start_state=state, eval_target=return_target)

                    # run episode
                    for t in range(env_steps):
                        if t == 1:
                            print(f"Starting episode {n}.")

                        pr_action, adv_action = model.get_action(state=state)
                        state, reward, done, trunc, _ = env.step(pr_action.squeeze())
                        model.update_history(
                            pr_action=pr_action, 
                            adv_action=adv_action, 
                            state=state, 
                            reward=reward,
                            timestep=t
                        )
                        episode_return += reward
                        episode_length += 1

                        # finish and log episode
                        if done or trunc or t == env_steps - 2:
                            eval_dict['iter'].append(n)
                            eval_dict['env_seed'].append(n)
                            eval_dict['init_target_return'].append(eval_target)
                            eval_dict['ep_length'].append(episode_length)
                            eval_dict['ep_return'].append(episode_return)
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
