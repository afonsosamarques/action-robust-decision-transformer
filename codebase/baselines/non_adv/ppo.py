import argparse

import gymnasium as gym
import numpy as np
import torch
import random
import os
import subprocess

from collections import defaultdict
from datasets import Dataset
from stable_baselines3 import PPO

from .gym_wrapper import GymWrapperRecorder


def find_root_dir():
    try:
        root_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
    except Exception as e:
        root_dir = os.getcwd()[:os.getcwd().find('action-robust-decision-transformer')+len('action-robust-decision-transformer')]
    return root_dir + ('' if root_dir.endswith('action-robust-decision-transformer') else '/action-robust-decision-transformer') + "/codebase/baselines/non_adv"


def load_env_name(env_type):
    if env_type == "halfcheetah":
        return "HalfCheetah-v4"
    elif env_type == "hopper":
        return "Hopper-v4"
    elif env_type == "walker2d":
        return "Walker2d-v4"
    else:
        raise Exception(f"Environment {env_type} not available.")
    

def set_seed_everywhere(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    if env is not None:
        env.seed = seed
        env.action_space.seed = seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, required=True, help='For model naming purposes.')
    parser.add_argument('--env_name', type=str, default='halfcheetah', help='Environment name.')
    parser.add_argument('--train_steps', type=int, default=10**6, help='Total number of steps in training.')
    parser.add_argument('--max_ep_len', type=int, default=1000, help='Max steps per episode.')
    parser.add_argument('--eval_trajs', type=int, default=5000, help='Total number of trajectories to collect in evaluation.')
    args = parser.parse_args()

    env_name = load_env_name(args.env_name)
    model_name = f"ppo_{env_name.split('-')[0].lower()}_v{args.version}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train model
    print("Training model...")
    env = GymWrapperRecorder(gym.make(env_name))
    model = PPO("MlpPolicy", env, verbose=0, device=device, seed=args.version)
    model.learn(total_timesteps=args.train_steps, progress_bar=True, log_interval=args.train_steps//10)
    model.save(f"{find_root_dir()}/agents/{model_name}")
    
    # # in case we want to use an already trained model!!
    # model = PPO.load(f"{find_root_dir()}/agents/{model_name}")

    # store training data
    print("Storing training data...")
    td = env.get_all_episodes()
    d = defaultdict(list)
    returns = []
    for t in td:
        d['observations'].append([list(obs) for obs in t['observations']])
        d['actions'].append([list(act) for act in t['actions']])
        d['rewards'].append(list(t['rewards']))
        d['dones'].append(list(t['dones']))
        returns.append(sum(t['rewards']))

    print(f"Episode returns | Avg: {np.round(np.mean(returns), 4)} | Std: {np.round(np.std(returns), 4)} | Min: {np.round(np.min(returns), 4)} | Median: {np.round(np.median(returns), 4)} | Max: {np.round(np.max(returns), 4)}")
    ds = Dataset.from_dict(d)
    ds.save_to_disk(f'{find_root_dir()}/datasets/{model_name}_train_v{args.version}') 

    # collect and store evaluation data
    print("Collecting evaluation data...")
    eval_dict = defaultdict(list)
    env = GymWrapperRecorder(gym.make(env_name))

    for i in range(args.eval_trajs + 1):
        set_seed_everywhere(i)
        ep_return = 0
        ep_len = 0
        obs, _ = env.reset(i)

        while True:
            action, _ = model.predict(obs)
            obs, reward, done, trunc, info = env.step(action)

            ep_return += reward
            ep_len += 1

            if done or trunc or ep_len == args.max_ep_len:
                eval_dict['iter'].append(i)
                eval_dict['ep_length'].append(ep_len)
                eval_dict['ep_return'].append(ep_return)
                break

    print(f"Episode lengths | Avg: {np.round(np.mean(eval_dict['ep_length']), 4)} | Std: {np.round(np.std(eval_dict['ep_length']), 4)} | Min: {np.round(np.min(eval_dict['ep_length']), 4)} | Median: {np.round(np.median(eval_dict['ep_length']), 4)} | Max: {np.round(np.max(eval_dict['ep_length']), 4)}")
    print(f"Episode returns | Avg: {np.round(np.mean(eval_dict['ep_return']), 4)} | Std: {np.round(np.std(eval_dict['ep_return']), 4)} | Min: {np.round(np.min(eval_dict['ep_return']), 4)} | Median: {np.round(np.median(eval_dict['ep_return']), 4)} | Max: {np.round(np.max(eval_dict['ep_return']), 4)}")

    td = env.get_all_episodes()
    d = defaultdict(list)
    for t in td:
        d['observations'].append([list(obs) for obs in t['observations']])
        d['actions'].append([list(act) for act in t['actions']])
        d['rewards'].append(list(t['rewards']))
        d['dones'].append(list(t['dones']))

    ds = Dataset.from_dict(d)
    ds.save_to_disk(f'{find_root_dir()}/datasets/{model_name}_test_v{args.version}') 
