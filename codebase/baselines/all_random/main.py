import argparse
import json

import gymnasium as gym

from collections import defaultdict

from datasets import Dataset

from .helpers import load_env_name, find_root_dir
from .random_agent import RandomAgent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="halfcheetah", help='name of the environment to run')
    parser.add_argument('--num_steps', type=int, default=10**6, metavar='N', help='number of training steps (default: 10**6)')
    parser.add_argument('--max_ep_len', type=int, default=1000, help='maximum number of steps in an episode')
    parser.add_argument('--min_ep_len', type=int, default=20, help='minimum number of steps in an episode, corresponds to context size')
    args = parser.parse_args()

    env_name = load_env_name(args.env_name)
    steps_remaining = args.num_steps
    prev_steps_remaining = steps_remaining+1
    d = defaultdict(list)

    print("============== Running random agent. ==============")
    while steps_remaining > 0:
        if steps_remaining < prev_steps_remaining and steps_remaining % (args.num_steps/10) == 0:
            print("Steps remaining:", steps_remaining)
        env = gym.make(env_name)
        env.reset()
        agent = RandomAgent(env.action_space.shape[0])
        ep = defaultdict(list)

        for t in range(args.max_ep_len):
            pr_action, adv_action = agent.get_action()
            cumul_action = (pr_action + adv_action).detach().numpy()
            state, reward, done, trunc, info = env.step(cumul_action)
            ep['observations'].append(state)
            ep['pr_actions'].append(pr_action)
            ep['adv_actions'].append(adv_action)
            ep['rewards'].append(reward)
            ep['dones'].append(done)

            if done or trunc:
                prev_steps_remaining = steps_remaining
                if len(ep['observations']) > args.min_ep_len:
                    d['observations'].append(ep['observations'])
                    d['pr_actions'].append(ep['pr_actions'])
                    d['adv_actions'].append(ep['adv_actions'])
                    d['rewards'].append(ep['rewards'])
                    d['dones'].append(ep['dones'])
                    steps_remaining -= len(ep['observations'])
                break
    
    ds = Dataset.from_dict(d)
    ds.save_to_disk(f'{find_root_dir()}/datasets/randagent_{args.env_name}')

    with open(f'{find_root_dir()}/agents/randagent_{args.env_name}', 'w') as f:
        json.dump({'action_space': env.action_space.shape[0]}, f)
