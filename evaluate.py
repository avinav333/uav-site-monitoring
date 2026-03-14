"""
Evaluate trained UAV agents on the construction site.
Run: python evaluate.py
"""

import os
import time
from env import ConstructionSiteEnv
from agent import QLearningAgent

SAVE_DIR   = "models"
N_EVAL_EPS = 10

env    = ConstructionSiteEnv()
agents = [QLearningAgent(agent_id=i, epsilon=0.0) for i in range(3)]  # no exploration

# Load saved models
for agent in agents:
    path = os.path.join(SAVE_DIR, f"agent_{agent.agent_id}.pkl")
    if os.path.exists(path):
        agent.load(path)
    else:
        print(f"No saved model for agent {agent.agent_id}. Run train.py first.")
        exit()

coverages = []

for ep in range(1, N_EVAL_EPS + 1):
    obs  = env.reset()
    done = False

    while not done:
        actions  = [agent.choose_action(obs) for agent in agents]
        obs, _, done, info = env.step(actions)

    env.render()
    coverages.append(info["coverage"])
    print(f"Episode {ep}: Coverage = {info['coverage']*100:.1f}%  |  Steps = {info['steps']}\n")
    time.sleep(0.3)

print(f"\nAverage coverage over {N_EVAL_EPS} episodes: {sum(coverages)/len(coverages)*100:.1f}%")
