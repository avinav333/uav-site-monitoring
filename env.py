import numpy as np
import random

class ConstructionSiteEnv:
    def __init__(self, grid_size=12, n_agents=3, n_obstacles=10):
        self.grid_size    = grid_size
        self.n_agents     = n_agents
        self.n_obstacles  = n_obstacles
        self.reset()

    def reset(self):
        self.grid    = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.obstacles = set()
        while len(self.obstacles) < self.n_obstacles:
            r = random.randint(2, self.grid_size - 3)
            c = random.randint(2, self.grid_size - 3)
            self.obstacles.add((r, c))
        for r, c in self.obstacles:
            self.grid[r][c] = 1

        # Divide grid into 3 vertical zones for each UAV
        zone_w = self.grid_size // self.n_agents
        self.agent_positions = [[0, i * zone_w] for i in range(self.n_agents)]
        self.agent_zones     = [(i * zone_w, (i+1) * zone_w) for i in range(self.n_agents)]
        self.agent_dirs      = [1] * self.n_agents   # 1=down, -1=up
        self.agent_cols      = [i * zone_w for i in range(self.n_agents)]

        for pos in self.agent_positions:
            self.visited[pos[0]][pos[1]] = True

        self.steps = 0
        self.max_steps = 200
        return self._coverage()

    def _coverage(self):
        total = self.grid_size**2 - len(self.obstacles)
        return np.sum(self.visited) / total

    def step(self):
        """Systematic serpentine sweep — each UAV covers its zone column by column."""
        self.steps += 1
        for i in range(self.n_agents):
            r, c      = self.agent_positions[i]
            zone_start, zone_end = self.agent_zones[i]
            # Move in current direction
            nr = r + self.agent_dirs[i]
            # If hit boundary or obstacle, shift to next column
            if nr < 0 or nr >= self.grid_size or self.grid[nr][c] == 1:
                self.agent_dirs[i] *= -1
                nc = c + 1
                if nc >= zone_end:
                    nc = zone_start   # wrap back
                self.agent_positions[i] = [r, nc]
            else:
                self.agent_positions[i] = [nr, c]
            pr, pc = self.agent_positions[i]
            if self.grid[pr][pc] != 1:
                self.visited[pr][pc] = True

        done     = self.steps >= self.max_steps
        coverage = self._coverage()
        return coverage, done

    def render(self):
        display = [['.' if self.grid[r][c] == 0 else 'X'
                    for c in range(self.grid_size)]
                   for r in range(self.grid_size)]
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.visited[r][c] and self.grid[r][c] == 0:
                    display[r][c] = '*'
        for i, (r, c) in enumerate(self.agent_positions):
            display[r][c] = str(i+1)
        print(f"\nStep {self.steps} | Coverage: {self._coverage()*100:.1f}%")
        for row in display:
            print(' '.join(row))
        print("Legend: . unvisited  * visited  X obstacle  1/2/3 UAVs")
