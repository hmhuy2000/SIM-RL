import os
import numpy as np
import torch

class Trajectory_Buffer_Continuous:

    def __init__(self,buffer_size, state_shape, action_shape, device):
        if (buffer_size):
            self.roll_n = 0
            self.roll_p = 0
            self.buffer_size = buffer_size
            self.roll_states = torch.empty(
                (buffer_size,*state_shape), dtype=torch.float, device=device)
            self.roll_next_states = torch.empty(
                (buffer_size,*state_shape), dtype=torch.float, device=device)
            self.roll_actions = torch.empty(
                (buffer_size,*action_shape), dtype=torch.float, device=device)
            self.roll_rewards = torch.empty(
                (buffer_size,1), dtype=torch.float, device=device)
            self.roll_costs = torch.empty(
                (buffer_size,1), dtype=torch.float, device=device)
            self.roll_dones = torch.empty(
                (buffer_size,1), dtype=torch.float, device=device)
    
    def sample_roll(self, batch_size):
        if (self.roll_n>0 and self.buffer_size>0):
            idxes = np.random.randint(low=0, high=self.roll_n, size=batch_size)
            return (
                self.roll_states[idxes],
                self.roll_actions[idxes],
                self.roll_next_states[idxes],
                self.roll_rewards[idxes],
                self.roll_costs[idxes],
                self.roll_dones[idxes],
            )
        else:
            return(
                None,
                None,
                None,
                None,
                None,
                None,
            )
    
    def append_roll(self,state, action,next_state,reward,cost,done):
        assert self.buffer_size>0
        self.roll_states[self.roll_p].copy_(torch.from_numpy(state))
        self.roll_actions[self.roll_p].copy_(torch.from_numpy(action))
        self.roll_next_states[self.roll_p].copy_(torch.from_numpy(next_state))
        self.roll_rewards[self.roll_p].copy_(torch.from_numpy(reward))
        self.roll_costs[self.roll_p].copy_(torch.from_numpy(cost))
        self.roll_dones[self.roll_p].copy_(torch.from_numpy(done))

        self.roll_p = (self.roll_p + 1) % self.buffer_size
        self.roll_n = min(self.roll_n + 1, self.buffer_size)

class RolloutBuffer_PPO_lag:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.total_rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.costs = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward,total_reward,cost, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.total_rewards[self._p] = float(total_reward)
        self.costs[self._p] = float(cost)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        start = 0
        idxes = slice(start, self._n)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.total_rewards[idxes],
            self.costs[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.total_rewards[idxes],
            self.costs[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )