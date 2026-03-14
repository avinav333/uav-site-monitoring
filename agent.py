"""
Q-Learning Agent for UAV Coverage
Uses a simple lookup table — one entry per grid cell per agent.
"""

import numpy as np
import random
import pickle
import os


class QLearningAgent:
    def __init__(self, agent_id, n_states=100, action_space=4,
                 alpha=0.5, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.997, epsilon_min=0.05):
        self.agent_id      = agent_id
        self.action_space  = action_space
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        # Simple table: state = grid cell index (0..99 for 10x10)
        self.q_table = np.zeros((n_states, action_space))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.q_table = pickle.load(f)
