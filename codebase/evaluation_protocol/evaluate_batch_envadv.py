import json
import os

import gymnasium as gym
import numpy as np
import torch

from collections import defaultdict
from requests.exceptions import HTTPError

from .config_utils import load_model
from .helpers import set_seed_everywhere, find_root_dir, scrappy_print_eval_dict
        

def sample_env_params(env):
    mb = env.model.body_mass
    mb = torch.tensor(mb)
    gauss = torch.distributions.Normal(mb, torch.ones_like(mb) * 0.5)
    mb = gauss.sample()
    env.model.body_mass = np.array(mb)
    
    # mb = env.model.opt.gravity
    # mb = torch.tensor(mb)
    # gauss = torch.distributions.Normal(mb, torch.ones_like(mb)*0.25)
    # mb = gauss.sample()
    # env.model.opt.gravity = np.array(mb)

    mb = env.model.geom_friction
    mb = torch.tensor(mb)
    gauss = torch.distributions.Normal(mb, torch.ones_like(mb) * 0.2)
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

    # setting up environments for parallel runs
    envs = []
    start_states = []
    for n in range(eval_iters):
        env = gym.make(env_name)
        set_seed_everywhere(n, env)
        if is_adv_eval:
            env = sample_env_params(env)
        state, _ = env.reset()
        envs.append(env)
        start_states.append(state)
    start_states = np.array(start_states)
    
    # evaluation loop
    print("\n================================================")
    print(f"Evaluating model {model_name} on environment {env_type}.")

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
        model.new_batch_eval(start_states=start_states, eval_target=eval_target)

        # run episode
        for t in range(env_steps):
            pr_actions, est_adv_actions = model.get_batch_actions(states=start_states)

            states = np.zeros_like(start_states)
            rewards = np.zeros(eval_iters)
            for i, env in enumerate(envs):
                # FIXME does not deal with possibility that they might end at different times!
                state, reward, done, trunc, _ = env.step(pr_actions[i])

                states[i] = state
                rewards[i] = reward
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
                if done or trunc or t == env_steps - 1:
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
    
    # show some simple statistics
    if verbose:
        scrappy_print_eval_dict(model_name, eval_dict)

    # save eval_dict as json
    dir_path = f'{find_root_dir()}/eval-outputs{run_suffix}/{model_name}/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(f"{dir_path}/{('env-adv' if is_adv_eval else 'no-adv')}.json", 'w') as f:
        json.dump(eval_dict, f)
