"""
Construction Site Environment for Multi-UAV Monitoring
Each UAV navigates a grid representing a construction site,
avoiding dynamic no-fly zones (heavy machinery, scaffolding).
"""

import numpy as np
import random

EMPTY = 0
OBSTACLE = 1
VISITED = 2
UAV = 3

class ConstructionSiteEnv:
    def __init__(self, grid_size=12, n_agents=3, n_obstacles=10):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_obstacles = n_obstacles
        self.action_space = 5  # 0=up, 1=down, 2=left, 3=right, 4=hover
        self.state_size = n_agents * 2 + n_obstacles * 2  # positions
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Place obstacles (machinery / scaffolding)
        self.obstacles = []
        while len(self.obstacles) < self.n_obstacles:
            r, c = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if (r, c) not in self.obstacles:
                self.obstacles.append((r, c))
                self.grid[r][c] = OBSTACLE

        # Place agents at corners / edges
        starts = [(0, 0), (0, self.grid_size - 1), (self.grid_size - 1, 0)]
        self.agent_positions = []
        for i in range(self.n_agents):
            pos = starts[i]
            while self.grid[pos[0]][pos[1]] == OBSTACLE:
                pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            self.agent_positions.append(list(pos))

        self.steps = 0
        self.max_steps = 200
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for pos in self.agent_positions:
            obs.extend([pos[0] / self.grid_size, pos[1] / self.grid_size])
        for obs_pos in self.obstacles:
            obs.extend([obs_pos[0] / self.grid_size, obs_pos[1] / self.grid_size])
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        self.steps += 1
        rewards = []
        dones = []

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

        for i, action in enumerate(actions):
            dr, dc = moves[action]
            nr = self.agent_positions[i][0] + dr
            nc = self.agent_positions[i][1] + dc

            # Boundary and obstacle check
            if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size
                    and self.grid[nr][nc] != OBSTACLE):
                self.agent_positions[i] = [nr, nc]

            r, c = self.agent_positions[i]
            reward = 0
            if not self.visited[r][c]:
                self.visited[r][c] = True
                reward = 1.0  # reward for new cell coverage
            else:
                reward = -0.05  # small penalty for revisiting

            rewards.append(reward)

        done = (self.steps >= self.max_steps)
        coverage = np.sum(self.visited) / (self.grid_size ** 2 - len(self.obstacles))

        return self._get_obs(), rewards, done, {"coverage": coverage, "steps": self.steps}

    def render(self):
        grid_display = self.grid.copy().astype(str)
        grid_display[grid_display == '0'] = '.'
        grid_display[grid_display == '1'] = 'X'

        for r, c in zip(*np.where(self.visited)):
            if grid_display[r][c] == '.':
                grid_display[r][c] = '*'

        for i, pos in enumerate(self.agent_positions):
            grid_display[pos[0]][pos[1]] = str(i + 1)

        print(f"\nStep: {self.steps}  |  Coverage: {np.sum(self.visited)}/{self.grid_size**2 - len(self.obstacles)}")
        print("  " + " ".join([str(i) for i in range(self.grid_size)]))
        for i, row in enumerate(grid_display):
            print(f"{i} " + " ".join(row))
        print("Legend: . = unvisited, * = visited, X = obstacle, 1/2/3 = UAVs")
