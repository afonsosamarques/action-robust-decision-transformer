import json
import os

import gymnasium as gym
import numpy as np
import torch
import warnings

from collections import defaultdict
from requests.exceptions import HTTPError

from ..config_utils import load_model
from ..helpers import set_seed_everywhere, find_root_dir, scrappy_print_eval_dict
        

def sample_env_params(env):
    mb = env.model.body_mass
    mb = torch.tensor(mb)
    gauss = torch.distributions.Normal(mb, torch.ones_like(mb)*0.8)
    mb = gauss.sample()
    env.model.body_mass = np.array(mb)
    
    mb = env.model.opt.gravity
    mb = torch.tensor(mb)
    gauss = torch.distributions.Normal(mb, torch.ones_like(mb)*0.8)
    mb = gauss.sample()
    env.model.opt.gravity = np.array(mb)

    mb = env.model.geom_friction
    mb = torch.tensor(mb)
    gauss = torch.distributions.Normal(mb, torch.ones_like(mb)*0.1)
    mb = gauss.sample()
    env.model.geom_friction = np.array(mb)

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
        verbose=False,
        device=torch.device('cpu'),
        model_path=None,
        hf_project=None,
    ):
    # linked to mujoco exception explained below
    warnings.filterwarnings('error')

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

    eval_dict = defaultdict(list)
    run_fails = 0
    run_idx = 0
    n_runs = 0

    while True:
        run_idx += 1
        if n_runs >= eval_iters:
            break
        if (run_idx % min(5, eval_iters/2)) == 0:
            print(f"Run number {run_idx}...")
        
        try:
            with torch.no_grad():
                # set up environment for run
                env = gym.make(env_name)
                set_seed_everywhere(run_idx, env)
                if is_adv_eval:
                    env = sample_env_params(env)
                    print("Checking that sampling worked. Gravity: ", env.model.opt.gravity)

                # set up episode variables
                episode_return, episode_length = 0, 0
                
                # reset environment
                state, _ = env.reset()
                model.new_eval(start_state=state, eval_target=eval_target)

                # run episode
                for t in range(env_steps):
                    pr_action, adv_action = model.get_action(state=state)
                    state, reward, done, _, _ = env.step(pr_action)
                    model.update_history(
                        pr_action=pr_action, 
                        adv_action=adv_action, 
                        state=state, 
                        reward=reward,
                    )
                    episode_return += reward
                    episode_length += 1

                    # finish and log episode
                    if done or t == env_steps - 1:
                        n_runs += 1
                        eval_dict['iter'].append(run_idx)
                        eval_dict['env_seed'].append(run_idx)
                        eval_dict['init_target_return'].append(eval_target)
                        eval_dict['ep_length'].append(episode_length)
                        eval_dict['ep_return'].append(episode_return)
                        run_fails = 0
                        break
        
        # when mujoco throws a warning about environment instability
        except Warning as w:
            run_fails += 1
            if run_fails > 3:
                # workaround to deal with the fact that the environment sometimes crashes
                # and I can't seem to be able to use MujocoException to catch it
                print("Too many consecutive failed runs. Stopping evaluation.")
                raise w
            print(f"Run {run_idx} failed with warning {w}")
            continue
    
        # when mujoco throws an error about environment instability
        except Exception as e:
            run_fails += 1
            if run_fails > 3:
                # workaround to deal with the fact that the environment sometimes crashes
                # and I can't seem to be able to use MujocoException to catch it
                print("Too many consecutive failed runs. Stopping evaluation.")
                raise e
            print(f"Run {run_idx} failed with error {e}")
            continue
    
    # show some simple statistics
    if verbose:
        scrappy_print_eval_dict(model_name, eval_dict)

    # save eval_dict as json
    dir_path = f'{find_root_dir()}/eval-outputs{run_suffix}/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(f'{dir_path}/{model_name}.json', 'w') as f:
        json.dump(eval_dict, f)

    # cleanup admin
    warnings.resetwarnings()
