import json
import os
import warnings

import gymnasium as gym
import numpy as np
import torch

from collections import defaultdict
from requests.exceptions import HTTPError

from ..utils.config_utils import load_model
from ..utils.helpers import set_seed_everywhere, find_root_dir, scrappy_print_eval_dict
        

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
        eval_iters,
        eval_target,
        is_adv_eval=False,
        hf_project="afonsosamarques",
        run_suffix='',
        verbose=False,
        device='cpu',
        model_path_local=None
    ):
    # linked to mujoco exception explained below
    warnings.filterwarnings('error')

    # load model
    try:
        model, is_adv_model = load_model(model_type, model_name, hf_project=hf_project, model_path_local=model_path_local)
    except HTTPError as e:
        print(f"Could not load model {model_name} from repo.")
    model.to(device)
    
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

                # set up whether the model also models an adversary; if not adv_act_dim will not be available
                if is_adv_model:
                    adv_act_dim = model.config.adv_act_dim
                else:
                    adv_act_dim = model.config.pr_act_dim
                
                # reset environment
                state, _ = env.reset()

                # set up episode variables
                episode_return, episode_length = 0, 0
                returns_scale = model.config.returns_scale if 'returns_scale' in model.config.to_dict().keys() else model.config.scale  # FIXME backwards compatibility
                target_return = torch.tensor(eval_target/returns_scale, device=device, dtype=torch.float32).reshape(1, 1)
                states = torch.from_numpy(state).reshape(1, model.config.state_dim).to(device=device, dtype=torch.float32)
                pr_actions = torch.zeros((0, model.config.pr_act_dim), device=device, dtype=torch.float32)
                adv_actions = torch.zeros((0, adv_act_dim), device=device, dtype=torch.float32)
                rewards = torch.zeros(0, device=device, dtype=torch.float32)
                timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

                # run episode
                for t in range(model.config.max_ep_len):
                    pr_actions = torch.cat([pr_actions, torch.zeros((1, model.config.pr_act_dim), device=device)], dim=0)
                    adv_actions = torch.cat([adv_actions, torch.zeros((1, adv_act_dim), device=device)], dim=0)
                    rewards = torch.cat([rewards, torch.zeros(1, device=device)])

                    pr_action, adv_action = model.get_action(
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

                    state, reward, done, trunc, info = env.step(action)

                    cur_state = torch.from_numpy(state.astype(np.float32)).to(device=device).reshape(1, model.config.state_dim)
                    states = torch.cat([states, cur_state], dim=0)
                    rewards[-1] = reward

                    pred_return = target_return[0, -1] - (reward / returns_scale)
                    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
                    timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

                    episode_return += reward
                    episode_length += 1

                    # finish and log episode
                    if done or trunc or t == model.config.max_ep_len - 1:
                        n_runs += 1
                        eval_dict['iter'].append(run_idx)
                        eval_dict['env_seed'].append(run_idx)
                        eval_dict['init_target_return'].append(target_return[0][0].item())
                        eval_dict['final_target_return'].append(target_return[0][-1].item())
                        eval_dict['ep_length'].append(episode_length)
                        eval_dict['ep_return'].append(episode_return / returns_scale)
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
