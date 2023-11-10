import torch
from torch import nn
from torch.optim import Adam
import torch.distributions as tdist
import torch.nn.functional as F
import sys
from copy import deepcopy
import os
import numpy as np

from Sources.algo.base_algo import Algorithm
from Sources.buffer import RolloutBuffer_PPO_lag
from Sources.network import StateIndependentPolicy,StateFunction


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)

def calculate_gae_cost(values, rewards, dones, next_values, gamma, lambd):

    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean())#/ (gaes.std() + 1e-8)

class PPO_continuous(Algorithm):
    def __init__(self,env_name, state_shape, action_shape, device, seed, gamma,cost_gamma,
        buffer_size, mix, hidden_units_actor, hidden_units_critic,units_clfs,batch_size,
        lr_actor, lr_critic,lr_cost_critic,lr_penalty,lr_clfs, epoch_ppo,epoch_clfs, clip_eps, lambd, coef_ent, 
        max_grad_norm,reward_factor,max_episode_length,cost_limit,risk_level,
        num_envs,primarive=True):
        super().__init__(device, seed, gamma)

        if (primarive):
            self.buffer = RolloutBuffer_PPO_lag(
                buffer_size=buffer_size,
                state_shape=state_shape,
                action_shape=action_shape,
                device=device,
                mix=mix
            )

            # Actor.
            self.actor = StateIndependentPolicy(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=hidden_units_actor,
                hidden_activation=nn.ReLU()
            ).to(device)

            # Critic.
            self.critic = StateFunction(
                state_shape=state_shape,
                hidden_units=hidden_units_critic,
                hidden_activation=nn.ReLU()
            ).to(device)

            self.cost_critic = StateFunction(
                state_shape=state_shape,
                hidden_units=hidden_units_critic,
                hidden_activation=nn.ReLU()
            ).to(device)

            self.penalty = torch.tensor(0.0).to(self.device)
            self.penalty.requires_grad = True

            self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
            self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)
            self.optim_cost_critic = Adam(self.cost_critic.parameters(), lr=lr_cost_critic)
            self.optim_penalty = Adam([self.penalty], lr=lr_penalty)

        self.rollout_length = buffer_size
        self.epoch_ppo = epoch_ppo
        self.epoch_clfs = epoch_clfs
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.reward_factor = reward_factor
        self.env_length = []
        self.max_episode_length = max_episode_length
        self.return_cost = []
        self.return_reward = []
        self.cost_limit = cost_limit
        self.num_envs = num_envs
        self.cost_gamma = cost_gamma
        self.target_kl = 0.05
        self.tmp_buffer = [[] for _ in range(self.num_envs)]
        self.tmp_return_cost = [0 for _ in range(self.num_envs)]
        self.tmp_return_reward = [0 for _ in range(self.num_envs)]
        self.start_train_good = False

    def step(self, env, state, ep_len):
        action, log_pi = self.explore(state)
        next_state, reward, c, done, _, _  = env.step(action)
        for idx in range(self.num_envs):
            ep_len[idx] += 1
            mask = False if ep_len[idx] >= self.max_episode_length else done[idx]
            self.tmp_buffer[idx].append((state[idx], action[idx], reward[idx] * self.reward_factor,
            c[idx], mask, log_pi[idx], next_state[idx]))
            self.tmp_return_cost[idx] += c[idx]
            self.tmp_return_reward[idx] += reward[idx]
            if (self.max_episode_length and ep_len[idx]>=self.max_episode_length):
                done[idx] = True

            if (done[idx]):
                for (tmp_state,tmp_action,tmp_reward,tmp_c,tmp_mask,tmp_log_pi,tmp_next_state) in self.tmp_buffer[idx]:
                    self.buffer.append(tmp_state, tmp_action, tmp_reward,self.tmp_return_reward[idx],
                     tmp_c, tmp_mask, tmp_log_pi, tmp_next_state)
                self.tmp_buffer[idx] = []
                self.return_cost.append(self.tmp_return_cost[idx])
                self.return_reward.append(self.tmp_return_reward[idx])
                self.tmp_return_cost[idx] = 0
                self.tmp_return_reward[idx] = 0

        if(done[0]):
            next_state,_ = env.reset()
            ep_len = [0 for _ in range(self.num_envs)]

        return next_state, ep_len

    def is_update(self,step):
        return step % self.rollout_length == 0

    def update(self,log_info):
        self.learning_steps += 1
        states, actions, env_rewards,env_totals, costs, dones, log_pis, next_states = \
            self.buffer.get()
        env_rewards = env_rewards.clamp(min=-3.0,max=3.0)
        rewards = env_rewards
        print(f'[Train] R: {np.mean(self.return_reward):.2f}, C: {np.mean(self.return_cost):.2f}')
        self.update_ppo(states, actions, rewards, costs, dones, log_pis, next_states,log_info)
        log_info.update({
            'Train/return':np.mean(self.return_reward),
            'Train/cost':np.mean(self.return_cost),
            'update/env_reward':env_rewards.mean().item(),
            'update/log_pis':log_pis.mean().item(),
        })
        self.return_cost = []
        self.return_reward = []

    def update_ppo(self, states, actions, rewards,costs, dones, log_pis, next_states,log_info):
        with torch.no_grad():
            values = self.critic(states)
            cost_values = self.cost_critic(states)         
            next_values = self.critic(next_states)
            next_cost_values = self.cost_critic(next_states) 

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)
        cost_targets, cost_gaes = calculate_gae_cost(
            cost_values, costs, dones, next_cost_values, self.cost_gamma, self.lambd)
        
        for _ in range(self.epoch_ppo):
            self.update_critic(states, targets, cost_targets,log_info)
        
        app_kl = 0.0
        for _ in range(self.epoch_ppo):
            if (app_kl>self.target_kl):
                break
            app_kl = self.update_actor(states, actions,
                            log_pis, gaes,cost_gaes,log_info)
            
        log_info.update({
            'value_critic/values_mean':values.mean().item(),
            'value_critic/values_max':values.max().item(),
            'value_critic/values_min':values.min().item(),
            'value_critic/next_values_mean':next_values.mean().item(),
            'value_critic/next_values_max':next_values.max().item(),
            'value_critic/next_values_min':next_values.min().item(),
            
            'cost_critic/costs_mean':cost_values.mean().item(),
            'cost_critic/costs_max':cost_values.max().item(),
            'cost_critic/costs_min':cost_values.min().item(),
            'cost_critic/next_costs_mean':next_cost_values.mean().item(),
            'cost_critic/next_costs_max':next_cost_values.max().item(),
            'cost_critic/next_costs_min':next_cost_values.min().item(),
            
            'target/values_mean':targets.mean().item(),
            'target/values_max':targets.max().item(),
            'target/values_min':targets.min().item(),
            'target/costs_mean':cost_targets.mean().item(),
            'target/costs_max':cost_targets.max().item(),
            'target/costs_min':cost_targets.min().item(),

            'gaes/values_mean':gaes.mean().item(),
            'gaes/values_max':gaes.max().item(),
            'gaes/values_min':gaes.min().item(),
            'gaes/costs_mean':cost_gaes.mean().item(),
            'gaes/costs_max':cost_gaes.max().item(),
            'gaes/costs_min':cost_gaes.min().item(),
            
            'update/penalty':F.softplus(self.penalty).item(),
            'update/KL':app_kl,
        })

    def update_critic(self, states, targets,cost_targets,log_info):
        value_means = self.critic(states)
        loss_critic = (value_means - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, log_pis_old, gaes, cost_gaes,log_info):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()
        approx_kl = (log_pis_old - log_pis).mean().item()
        ratios = (log_pis - log_pis_old).exp_()

        total_gae = gaes

        loss_actor1 = -ratios * total_gae
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * total_gae
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()
        total_loss  = loss_actor - self.coef_ent * entropy 
        self.optim_actor.zero_grad()
        (total_loss).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()
        return approx_kl

    def save_models(self,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pth')
        torch.save(self.critic.state_dict(), f'{save_dir}/critic.pth')
        torch.save(self.cost_critic.state_dict(), f'{save_dir}/cost_critic.pth')

    def train(self):
        self.actor.train()
        self.critic.train()
        self.cost_critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.cost_critic.eval()

    def load_models(self,load_dir):
        if not os.path.exists(load_dir):
            raise
        self.actor.load_state_dict(torch.load(f'{load_dir}/actor.pth'))
        self.critic.load_state_dict(torch.load(f'{load_dir}/critic.pth'))
        self.cost_critic.load_state_dict(torch.load(f'{load_dir}/cost_critic.pth'))

    def copyNetworksFrom(self,algo):
        self.actor.load_state_dict(algo.actor.state_dict())

class PPO_lag(PPO_continuous):
    def __init__(self,env_name, state_shape, action_shape, device, seed, gamma,cost_gamma,
        buffer_size, mix, hidden_units_actor, hidden_units_critic,units_clfs,batch_size,
        lr_actor, lr_critic,lr_cost_critic,lr_penalty,lr_clfs, epoch_ppo,epoch_clfs, clip_eps, lambd, coef_ent, 
        max_grad_norm,reward_factor,max_episode_length,cost_limit,risk_level,
        num_envs,primarive=True):
        super().__init__(env_name, state_shape, action_shape, device, seed, gamma,cost_gamma,
        buffer_size, mix, hidden_units_actor, hidden_units_critic,units_clfs,batch_size,
        lr_actor, lr_critic,lr_cost_critic,lr_penalty,lr_clfs, epoch_ppo,epoch_clfs, clip_eps, lambd, coef_ent, 
        max_grad_norm,reward_factor,max_episode_length,cost_limit,risk_level,
        num_envs,primarive=True)
        self.target_cost = (
            self.cost_limit * (1 - self.cost_gamma**self.max_episode_length) / (1 - self.cost_gamma) / self.max_episode_length
        )

    def update_ppo(self, states, actions, rewards,costs, dones, log_pis, next_states,log_info):
        with torch.no_grad():
            values = self.critic(states)
            cost_values = self.cost_critic(states)         
            next_values = self.critic(next_states)
            next_cost_values = self.cost_critic(next_states) 

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)
        cost_targets, cost_gaes = calculate_gae_cost(
            cost_values, costs, dones, next_cost_values, self.cost_gamma, self.lambd)
        for _ in range(self.epoch_ppo):
            self.update_critic(states, targets, cost_targets,log_info)
        
        app_kl = 0.0
        for _ in range(self.epoch_ppo):
            if (app_kl>self.target_kl):
                break
            app_kl = self.update_actor(states, actions,
                        log_pis, gaes,cost_gaes,log_info)

        with torch.no_grad():
            cost_values = self.cost_critic(states)         
            cost_deviation = self.target_cost - cost_values
        loss_penalty = (F.softplus(self.penalty)*cost_deviation).mean()
        self.optim_penalty.zero_grad()
        loss_penalty.backward()
        self.optim_penalty.step()

        log_info.update({
            'value_critic/values_mean':values.mean().item(),
            'value_critic/values_max':values.max().item(),
            'value_critic/values_min':values.min().item(),
            'value_critic/next_values_mean':next_values.mean().item(),
            'value_critic/next_values_max':next_values.max().item(),
            'value_critic/next_values_min':next_values.min().item(),
            
            'cost_critic/costs_mean':cost_values.mean().item(),
            'cost_critic/costs_max':cost_values.max().item(),
            'cost_critic/costs_min':cost_values.min().item(),
            'cost_critic/next_costs_mean':next_cost_values.mean().item(),
            'cost_critic/next_costs_max':next_cost_values.max().item(),
            'cost_critic/next_costs_min':next_cost_values.min().item(),
            
            'target/values_mean':targets.mean().item(),
            'target/values_max':targets.max().item(),
            'target/values_min':targets.min().item(),
            'target/costs_mean':cost_targets.mean().item(),
            'target/costs_max':cost_targets.max().item(),
            'target/costs_min':cost_targets.min().item(),

            'gaes/values_mean':gaes.mean().item(),
            'gaes/values_max':gaes.max().item(),
            'gaes/values_min':gaes.min().item(),
            'gaes/costs_mean':cost_gaes.mean().item(),
            'gaes/costs_max':cost_gaes.max().item(),
            'gaes/costs_min':cost_gaes.min().item(),
            
            'update/penalty':F.softplus(self.penalty).item(),
            'update/KL':app_kl,
        })

    def update_actor(self, states, actions, log_pis_old, gaes, cost_gaes,log_info):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()
        approx_kl = (log_pis_old - log_pis).mean().item()
        ratios = (log_pis - log_pis_old).exp_()

        penalty = F.softplus(self.penalty).clamp(max=5).detach()
        total_gae = gaes - penalty * cost_gaes
        total_gae = total_gae/(penalty+1)

        loss_actor1 = -ratios * total_gae
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * total_gae
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()
        total_loss  = loss_actor - self.coef_ent * entropy 
        self.optim_actor.zero_grad()
        (total_loss).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()
        return approx_kl
    
    def update_critic(self, states, targets,cost_targets,log_info):
        value_means = self.critic(states)
        loss_critic = (value_means - targets).pow_(2).mean()
        cost_means = self.cost_critic(states)
        loss_cost_critic = (cost_means - cost_targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        self.optim_cost_critic.zero_grad()
        loss_cost_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
        self.optim_cost_critic.step()
