"""
Independent Q-Learning Agents for Multi-UAV Coordination
Each UAV has its own Q-table. Shared coverage reward encourages
cooperative behaviour without centralised control.
"""

import numpy as np
import random
import pickle
import os


class QLearningAgent:
    def __init__(self, agent_id, state_bins=8, action_space=5,
                 alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.agent_id = agent_id
        self.state_bins = state_bins
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def _discretize(self, obs):
        """Discretize continuous observation into hashable state."""
        return tuple((obs * self.state_bins).astype(int))

    def choose_action(self, obs):
        state = self._discretize(obs)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
        return int(np.argmax(self.q_table[state]))

    def update(self, obs, action, reward, next_obs, done):
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space)

        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'epsilon': self.epsilon}, f)
        print(f"Agent {self.agent_id} saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
            print(f"Agent {self.agent_id} loaded from {path}")
