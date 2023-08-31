import torch
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import os
import random
import numpy as np

from .network import Critic, Actor
from .utils import RunningMeanStd
from .replay_memory import ReplayMemory, Transition

from evaluation_protocol.helpers import EvalWrapper


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def normalize(x, stats, device):
    if stats is None:
        return x
    return (x - torch.tensor(stats.mean, dtype=torch.float32).to(device)) / torch.tensor(stats.var, dtype=torch.float32).sqrt().to(device)


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean


class DDPG:
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, train_mode, alpha, replay_size,
                 normalize_obs=True, normalize_returns=False, critic_l2_reg=1e-2, device=None):
        if device != None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.enabled = False
        else:
            self.device = torch.device('cpu')
    
        self.alpha = alpha
        self.train_mode = train_mode
        self.hidden_size = hidden_size
    
        self.num_inputs = num_inputs
        self.action_space = action_space
        self.critic_l2_reg = critic_l2_reg

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.adversary = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
        if self.train_mode:
            self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
            self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

            self.critic = Critic(hidden_size, self.num_inputs, self.action_space).to(self.device)
            self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space).to(self.device)
            self.critic_optim = Adam(self.critic.parameters(), lr=1e-3, weight_decay=critic_l2_reg)

            self.adversary_target = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
            self.adversary_perturbed = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
            self.adversary_optim = Adam(self.adversary.parameters(), lr=1e-4)

            hard_update(self.adversary_target, self.adversary)  # Make sure target is with the same weight
            hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
            hard_update(self.critic_target, self.critic)

        self.gamma = gamma
        self.tau = tau
        self.normalize_observations = normalize_obs
        self.normalize_returns = normalize_returns

        if self.normalize_observations:
            self.obs_rms = RunningMeanStd(shape=num_inputs)
        else:
            self.obs_rms = None

        if self.normalize_returns:
            self.ret_rms = RunningMeanStd(shape=1)
            self.ret = 0
            self.cliprew = 10.0
        else:
            self.ret_rms = None

        self.memory = ReplayMemory(replay_size)

    def eval(self, mdp_type=None, **kwargs):
        self.actor.eval()
        self.adversary.eval()
        if self.train_mode:
            self.critic.eval()
        return DDPGEvalWrapper(self, mdp_type)

    def train(self):
        self.actor.train()
        self.adversary.train()
        if self.train_mode:
            self.critic.train()

    def select_action(self, state, action_noise=None, param_noise=None, mdp_type='mdp'):
        state = normalize(Variable(state).to(self.device), self.obs_rms, self.device)

        if mdp_type != 'mdp':
            if mdp_type == 'nr_mdp':
                if param_noise is not None:
                    mu = self.actor_perturbed(state)
                else:
                    mu = self.actor(state)
                mu = mu.data
                if action_noise is not None:
                    mu += torch.tensor(action_noise(), dtype=torch.float32).to(self.device)
                
                pr_mu = mu.clamp(-1, 1)
                mu = pr_mu * (1 - self.alpha)

                if param_noise is not None:
                    adv_mu = self.adversary_perturbed(state)
                else:
                    adv_mu = self.adversary(state)

                adv_mu = adv_mu.data.clamp(-1, 1)
                mu += adv_mu  * self.alpha
            else:  # mdp_type == 'pr_mdp':
                #
                if param_noise is not None:
                    pr_mu = self.actor_perturbed(state)
                else:
                    pr_mu = self.actor(state)
                pr_mu = pr_mu.data
                if action_noise is not None:
                    pr_mu += torch.tensor(action_noise(), dtype=torch.float32).to(self.device)

                pr_mu = pr_mu.clamp(-1, 1)
                #
                if param_noise is not None:
                    adv_mu = self.adversary_perturbed(state)
                else:
                    adv_mu = self.adversary(state)

                adv_mu = adv_mu.data.clamp(-1, 1)
                
                if np.random.rand() < (1 - self.alpha):
                    mu = pr_mu
                else:
                    mu = adv_mu

        else:
            if param_noise is not None:
                mu = self.actor_perturbed(state)
            else:
                mu = self.actor(state)
            mu = mu.data
            if action_noise is not None:
                mu += torch.tensor(action_noise(), dtype=torch.float32).to(self.device)

            mu = mu.clamp(-1, 1)
            return mu

        return mu, pr_mu, adv_mu

    def update_robust(self, state_batch, action_batch, reward_batch, mask_batch, next_state_batch, adversary_update,
                      mdp_type, robust_update_type):
        # TRAIN CRITIC
        if robust_update_type == 'full':
            if mdp_type == 'nr_mdp':
                next_action_batch = (1 - self.alpha) * self.actor_target(next_state_batch) \
                                    + self.alpha * self.adversary_target(next_state_batch)

                next_state_action_values = self.critic_target(next_state_batch, next_action_batch)
            else:  # mdp_type == 'pr_mdp':
                next_action_actor_batch = self.actor_target(next_state_batch)
                next_action_adversary_batch = self.adversary_target(next_state_batch)

                next_state_action_values = self.critic_target(next_state_batch, next_action_actor_batch) * (
                            1 - self.alpha) \
                                           + self.critic_target(next_state_batch,
                                                                       next_action_adversary_batch) * self.alpha

            expected_state_action_batch = reward_batch + self.gamma * mask_batch * next_state_action_values

            self.critic_optim.zero_grad()

            state_action_batch = self.critic(state_batch, action_batch)

            value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
            value_loss.backward()
            self.critic_optim.step()
            value_loss = value_loss.item()
        else:
            value_loss = 0

        if adversary_update:
            # TRAIN ADVERSARY
            self.adversary_optim.zero_grad()

            if mdp_type == 'nr_mdp':
                with torch.no_grad():
                    real_action = self.actor_target(next_state_batch)
                action = (1 - self.alpha) * real_action + self.alpha * self.adversary(next_state_batch)
                adversary_loss = self.critic(state_batch, action)
            else:  # mdp_type == 'pr_mdp'
                action = self.adversary(next_state_batch)
                adversary_loss = self.critic(state_batch, action) * self.alpha

            adversary_loss = adversary_loss.mean()
            adversary_loss.backward()
            self.adversary_optim.step()
            adversary_loss = adversary_loss.item()
            policy_loss = 0
        else:
            if robust_update_type == 'full':
                # TRAIN ACTOR
                self.actor_optim.zero_grad()

                if mdp_type == 'nr_mdp':
                    with torch.no_grad():
                        adversary_action = self.adversary_target(next_state_batch)
                    action = (1 - self.alpha) * self.actor(next_state_batch) + self.alpha * adversary_action
                    policy_loss = -self.critic(state_batch, action)
                else:  # mdp_type == 'pr_mdp':
                    action = self.actor(next_state_batch)
                    policy_loss = -self.critic(state_batch, action) * (1 - self.alpha)

                policy_loss = policy_loss.mean()
                policy_loss.backward()
                self.actor_optim.step()

                policy_loss = policy_loss.item()
                adversary_loss = 0
            else:
                policy_loss = 0
                adversary_loss = 0

        return value_loss, policy_loss, adversary_loss

    def update_non_robust(self, state_batch, action_batch, reward_batch, mask_batch, next_state_batch):
        # TRAIN CRITIC

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        expected_state_action_batch = reward_batch + self.gamma * mask_batch * next_state_action_values

        self.critic_optim.zero_grad()

        state_action_batch = self.critic(state_batch, action_batch)

        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        # TRAIN ACTOR

        self.actor_optim.zero_grad()

        action = self.actor(next_state_batch)

        policy_loss = -self.critic(state_batch, action)

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        policy_loss = policy_loss.item()
        adversary_loss = 0

        return value_loss.item(), policy_loss, adversary_loss

    def store_transition(self, state, action, mask, next_state, reward):
        B = state.shape[0]
        for b in range(B):
            self.memory.push(state[b], action[b], mask[b], next_state[b], reward[b])
            if self.normalize_observations:
                self.obs_rms.update(state[b].cpu().numpy())
            if self.normalize_returns:
                self.ret = self.ret * self.gamma + reward[b]
                self.ret_rms.update(np.array([self.ret]))
                if mask[b] == 0:  # if terminal is True
                    self.ret = 0

    def update_parameters(self, batch_size, mdp_type='mdp', adversary_update=False, exploration_method='mdp'):
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        if mdp_type != 'mdp':
            robust_update_type = 'full'
        elif exploration_method != 'mdp':
            robust_update_type = 'adversary'
        else:
            robust_update_type = None

        state_batch = normalize(Variable(torch.stack(batch.state)).to(self.device), self.obs_rms, self.device)
        action_batch = Variable(torch.stack(batch.action)).to(self.device)
        reward_batch = normalize(Variable(torch.stack(batch.reward)).to(self.device).unsqueeze(1), self.ret_rms, self.device)
        mask_batch = Variable(torch.stack(batch.mask)).to(self.device).unsqueeze(1)
        next_state_batch = normalize(Variable(torch.stack(batch.next_state)).to(self.device), self.obs_rms, self.device)

        if self.normalize_returns:
            reward_batch = torch.clamp(reward_batch, -self.cliprew, self.cliprew)

        value_loss = 0
        policy_loss = 0
        adversary_loss = 0
        if robust_update_type is not None:
            _value_loss, _policy_loss, _adversary_loss = self.update_robust(state_batch, action_batch, reward_batch,
                                                                            mask_batch, next_state_batch,
                                                                            adversary_update,
                                                                            mdp_type,
                                                                            robust_update_type)
            value_loss += _value_loss
            policy_loss += _policy_loss
            adversary_loss += _adversary_loss
        if robust_update_type != 'full':
            _value_loss, _policy_loss, _adversary_loss = self.update_non_robust(state_batch, action_batch,
                                                                                reward_batch,
                                                                                mask_batch, next_state_batch)
            value_loss += _value_loss
            policy_loss += _policy_loss
            adversary_loss += _adversary_loss

        self.soft_update()

        return value_loss, policy_loss, adversary_loss

    def soft_update(self):
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.adversary_target, self.adversary, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += torch.randn(param.shape).to(self.device) * param_noise.current_stddev

        """Apply parameter noise to adversary model, for exploration"""
        hard_update(self.adversary_perturbed, self.adversary)
        params = self.adversary_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += torch.randn(param.shape).to(self.device) * param_noise.current_stddev

    def to(self, device=torch.device('cpu')):
        self.device = device


class DDPGEvalWrapper(EvalWrapper):
    def __init__(self, model, mdp_type="nr_mdp", **kwargs):
        super().__init__(model)
        self.mdp_type = mdp_type
    
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action, pr_action, adv_action = self.model.select_action(state, mdp_type=self.mdp_type)
        return pr_action.detach().cpu().numpy(), self.model.alpha * adv_action.detach().cpu().numpy()

    def get_batch_actions(self, states):
        states = torch.tensor(states, dtype=torch.float32)
        actions, pr_actions, adv_actions = self.model.select_action(states, mdp_type=self.mdp_type)
        return pr_actions.detach().cpu().numpy(), self.model.alpha * adv_actions.detach().cpu().numpy()
