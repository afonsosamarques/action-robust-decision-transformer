import argparse
import os
import gymnasium as gym
import numpy as np
import pickle
import math
import torch
import subprocess
import datetime 
import json

from tqdm import trange
from collections import defaultdict

from .ddpg import DDPG
from .normalized_actions import NormalizedActions
from .action_noise import NormalActionNoise
from .utils import save_model, vis_plot


def load_env_name(env_type):
    if env_type == "halfcheetah":
        return "HalfCheetah-v4"
    elif env_type == "hopper":
        return "Hopper-v4"
    elif env_type == "walker2d":
        return "Walker2d-v4"
    else:
        raise Exception(f"Environment {env_type} not available.")
    

def find_root_dir():
    try:
        root_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
    except Exception as e:
        root_dir = os.getcwd()[:os.getcwd().find('action-robust-decision-transformer')+len('action-robust-decision-transformer')]
    return root_dir + ('' if root_dir.endswith('action-robust-decision-transformer') else '/action-robust-decision-transformer') + "/codebase/baselines/arrl_sgld"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="halfcheetah",
                        help='name of the environment to run')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                        help='discount factor for model (default: 0.001)')
    parser.add_argument('--no_action_noise', default=False, action='store_true')
    parser.add_argument('--noise_scale', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=None, metavar='N',
                        help='number of epochs (default: None)')
    parser.add_argument('--num_epochs_cycles', type=int, default=20, metavar='N')
    parser.add_argument('--num_rollout_steps', type=int, default=100, metavar='N')
    parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                        help='number of training steps (default: 500000)')
    parser.add_argument('--hidden_size_dim0', type=int, default=64, metavar='N',
                        help='number of neurons in the hidden layers (default: 64)')
    parser.add_argument('--hidden_size_dim1', type=int, default=64)
    parser.add_argument('--number_of_train_steps', type=int, default=50, metavar='N')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--method', default='nr_mdp', choices=['mdp', 'nr_mdp'])
    parser.add_argument('--ratio', default=1, type=int)
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='control given to adversary (default: 0.1)')
    parser.add_argument('--exploration_method', default=None, choices=['mdp', 'nr_mdp'])
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='thermal noise (default: 1e-2 to 1e-5))')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta', type=float, default=0.9, help='beta (default: 0.9)')
    parser.add_argument('--optimizer', default='SGLD', choices=['SGLD', 'RMSprop', 'ExtraAdam'] )
    parser.add_argument('--one_player', default=False, action='store_true')
    parser.add_argument('--Kt', type=int, default=15)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    args.env_name = load_env_name(args.env_name)
    args.action_noise = not args.no_action_noise
    args.two_player = not args.one_player
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the folder the store the results
    if args.exploration_method is None:
        args.exploration_method = args.method

    env = NormalizedActions(gym.make(args.env_name))
    eval_env = NormalizedActions(gym.make(args.env_name))
    
    agent = DDPG(beta=args.beta, epsilon=args.epsilon, learning_rate=args.learning_rate, gamma=args.gamma, tau=args.tau, 
                    hidden_size_dim0=args.hidden_size_dim0, hidden_size_dim1=args.hidden_size_dim1, 
                    num_inputs=env.observation_space.shape[0], 
                    action_space=env.action_space, 
                    train_mode=True, alpha=args.alpha, replay_size=args.replay_size, optimizer=args.optimizer, two_player=args.two_player,
                    device=args.device)

    results_dict = {'eval_rewards': [],
                'value_losses': [],
                'policy_losses': [],
                'adversary_losses': [],
                'train_rewards': []
                }
    value_losses = []
    policy_losses = []
    adversary_losses = []

    if args.two_player:
        alpha = 0.1
    else:
        alpha = 0

    if args.two_player:
        base_dir = find_root_dir() + '/models/' + args.env_name + '/'
    else:
        base_dir = find_root_dir() + '/models_OnePlayer/' + args.env_name + '/'

    if args.optimizer == 'SGLD':
        base_dir += args.optimizer + '_thermal_' + str(args.epsilon) + '/'
    else:
        base_dir += args.optimizer + '/'

    if args.action_noise:
        base_dir += 'action_noise_' + str(args.noise_scale) + '/'
    else:
        base_dir += 'no_noise/'

    if args.exploration_method == args.method:
        if args.method != 'mdp':
            base_dir += args.method + '_' + str(alpha) + '_' + str(args.ratio) + '/'
        else:
            base_dir += 'non_robust/'
    else:
        if args.method != 'mdp':
            if args.flip_ratio:
                base_dir += 'flip_ratio_'
            base_dir += 'alternative_' + args.method + '_' + str(alpha) + '_' + str(args.ratio) + '/'
        else:
            base_dir += 'alternative_non_robust_' + args.exploration_method + '_' + str(alpha) + '_' + str(args.ratio) + '/'

    base_dir += 'beta_' + str(args.beta) + '/Kt_' + str(args.Kt) + '/'

    run_number = 0
    while os.path.exists(base_dir + str(run_number)):
        run_number += 1
    base_dir = base_dir + str(run_number)

    os.makedirs(base_dir)
    normalnoise = NormalActionNoise(mu=np.zeros(env.action_space.shape[0]),
                                    sigma=float(args.noise_scale) * np.ones(env.action_space.shape[0])
                                    ) if args.action_noise else None
    def reset_noise(a, a_noise):
        if a_noise is not None:
            a_noise.reset()

    total_steps = 0
    # print(base_dir)

    if args.num_steps is not None:
        assert args.num_epochs is None
        nb_epochs = int(args.num_steps) // (args.num_epochs_cycles * args.num_rollout_steps)
    else:
        nb_epochs = 500


    state = torch.tensor([env.reset()[0]], dtype=torch.float32)
    eval_state = torch.tensor([eval_env.reset()[0]], dtype=torch.float32)

    eval_reward = 0
    episode_reward = 0
    agent.train()

    reset_noise(agent, normalnoise)

    # if args.visualize:
    #     vis = visdom.Visdom(env=base_dir)
    # else:
    #     vis = None
    vis = None

    train_steps = 0
    ratio = args.ratio + 1
    hacky_store = defaultdict(list)
    ep = 0
    hacky_indict = None
    
    for epoch in trange(nb_epochs):
        for cycle in range(args.num_epochs_cycles):
            with torch.no_grad():
                # training stage
                for t_rollout in range(args.num_rollout_steps):
                    action, pr_mu, adv_mu = agent.select_action(state, action_noise=normalnoise, mdp_type=args.exploration_method)
                    next_state, reward, done, trunc, info = env.step(action.cpu().numpy()[0])

                    total_steps += 1
                    episode_reward += reward

                    action = torch.tensor(action, dtype=torch.float32)
                    mask = torch.tensor([not done])
                    next_state = torch.tensor([next_state], dtype=torch.float32)
                    reward = torch.tensor([reward], dtype=torch.float32)

                    agent.store_transition(state, action, mask, next_state, reward)
                    if hacky_indict is not None:
                        hacky_indict['pr_action'] = pr_mu.cpu().numpy()[0].tolist()
                        hacky_indict['adv_action'] = adv_mu.cpu().numpy()[0].tolist()
                        hacky_store[ep].append(hacky_indict)
                    hacky_indict = {'reward': str(reward.cpu().numpy()[0]), 'state': state.cpu().numpy()[0].tolist(), 'done': done, 'info': info} 

                    state = next_state
                    if done or trunc:
                        results_dict['train_rewards'].append((total_steps, np.mean(episode_reward)))
                        episode_reward = 0
                        ep += 1
                        hacky_indict = None
                        state = torch.tensor([env.reset()[0]], dtype=torch.float32)
                        reset_noise(agent, normalnoise)

            if len(agent.memory) > args.batch_size:
                # update the parameters
                for t_train in range(args.number_of_train_steps):
                    warmup = (math.floor(np.power(1 + 1e-5, train_steps)))
                    # warmup steps for SGLD + two_player
                    if (args.optimizer == 'SGLD' and args.two_player):
                        kt = np.minimum(args.Kt, warmup)                 
                        agent.initialize()
                    # noram setup for RMSPROP and SGLD + one_player
                    else:
                        kt = 1
            
                    for k in range(kt):
                        sgld_outer_update = (k == kt - 1)

                        value_loss, policy_loss, adversary_loss = agent.update_parameters(batch_size=args.batch_size,
                                                                                sgld_outer_update=sgld_outer_update,
                                                                                mdp_type=args.method,
                                                                                exploration_method=args.exploration_method)
                        value_losses.append(value_loss)
                        policy_losses.append(policy_loss)
                        adversary_losses.append(adversary_loss)
                    train_steps += 1
                
                results_dict['value_losses'].append((total_steps, np.mean(value_losses)))
                results_dict['policy_losses'].append((total_steps, np.mean(policy_losses)))
                results_dict['adversary_losses'].append((total_steps, np.mean(adversary_losses)))
                del value_losses[:]
                del policy_losses[:]
                del adversary_losses[:]
            
            # evaluation stage, with different environment from training stage
            with torch.no_grad():
                for t_rollout in range(args.num_rollout_steps):
                    action = agent.select_action(eval_state, mdp_type='mdp')

                    next_eval_state, reward, done, trunc, info = eval_env.step(action.cpu().numpy()[0])
                    eval_reward += reward

                    next_eval_state = torch.tensor([next_eval_state], dtype=torch.float32)

                    eval_state = next_eval_state
                    if done or trunc:
                        results_dict['eval_rewards'].append((total_steps, eval_reward))
                        eval_state = torch.tensor([eval_env.reset()[0]], dtype=torch.float32)
                        eval_reward = 0
            # save the model 
            save_model(agent=agent, actor=agent.actor, adversary=agent.adversary, basedir=base_dir, obs_rms=agent.obs_rms,
                rew_rms=agent.ret_rms)
            with open(base_dir + '/results', 'wb') as f:
                pickle.dump(results_dict, f)

            vis_plot(vis, results_dict)

    with open(find_root_dir() + '/results', 'wb') as f:
        pickle.dump(results_dict, f)
    save_model(agent=agent, actor=agent.actor, adversary=agent.adversary, basedir=base_dir, obs_rms=agent.obs_rms, rew_rms=agent.ret_rms)

    dir = find_root_dir() + '/datasets'
    os.makedirs(dir)
    with open(f'{dir}/arrl_sgld_raw_dataset-{args.env_name}-{datetime.datetime.now().strftime("%d%m_%H%M")}.json', 'w') as f:
        json.dump(hacky_store, f, indent=4)

    env.close()
