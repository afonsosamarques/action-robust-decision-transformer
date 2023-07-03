from collections import defaultdict

import gymnasium as gym
import numpy as np


class GymWrapperRecorder(gym.Wrapper):
    def __init__(self, env, max_ep_len=1000):
        super().__init__(env)
        self.max_ep_len = max_ep_len
        self.n_episodes = 0
        self.n_steps = 0
        self.episodes = []
        self.curr_episode = defaultdict(list)

    def step(self, action):
        # take some action
        observation, reward, done, trunc, info = self.env.step(action)
        self.n_steps += 1
        self.curr_episode['actions'].append(action)
        self.curr_episode['observations'].append(observation)
        self.curr_episode['rewards'].append(reward)
        done = True if self.n_steps == self.max_ep_len else done
        self.curr_episode['dones'].append(done)
        return observation, reward, done, trunc, info
    
    def reset(self, seed=None):
        # update current episode if episode ongoing
        if len(self.curr_episode) > 1:
            self.n_episodes += 1
            self.episodes.append(self.curr_episode)
        self.n_steps = 0
        self.curr_episode = defaultdict(list)
        # reset environment
        observation, info = self.env.reset()
        self.curr_episode['actions'].append(np.zeros_like(self.env.action_space.sample()))
        self.curr_episode['observations'].append(observation)
        self.curr_episode['rewards'].append(0.0)
        self.curr_episode['dones'].append(False)
        # start simulation
        return observation, info
    
    def get_all_episodes(self):
        # gather all the data
        return self.episodes
    
    def restart(self):
        # typically to restart the data collection
        self.n_episodes = 0
        self.n_steps = 0
        self.episodes = []
        self.curr_episode = defaultdict(list)
        return self.reset()
