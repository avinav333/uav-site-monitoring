"""
Training script for Multi-UAV Construction Site Monitoring
Run: python train.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from env import ConstructionSiteEnv
from agent import QLearningAgent

# ── Hyperparameters ──────────────────────────────────────
N_EPISODES   = 800
N_AGENTS     = 3
GRID_SIZE    = 12
N_OBSTACLES  = 10
SAVE_DIR     = "models"
RENDER_EVERY = 200   # render grid every N episodes
# ─────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

env   = ConstructionSiteEnv(grid_size=GRID_SIZE, n_agents=N_AGENTS, n_obstacles=N_OBSTACLES)
agents = [QLearningAgent(agent_id=i) for i in range(N_AGENTS)]

episode_coverages = []
episode_rewards   = []

print("Starting training...\n")

for ep in range(1, N_EPISODES + 1):
    obs   = env.reset()
    done  = False
    total_reward = 0.0

    while not done:
        actions = [agent.choose_action(obs) for agent in agents]
        next_obs, rewards, done, info = env.step(actions)

        for i, agent in enumerate(agents):
            agent.update(obs, actions[i], rewards[i], next_obs, done)

        obs = next_obs
        total_reward += sum(rewards)

    for agent in agents:
        agent.decay_epsilon()

    coverage = info["coverage"]
    episode_coverages.append(coverage)
    episode_rewards.append(total_reward)

    if ep % RENDER_EVERY == 0 or ep == 1:
        env.render()
        print(f"Episode {ep:4d}/{N_EPISODES}  |  Coverage: {coverage*100:.1f}%  |  "
              f"Total Reward: {total_reward:.2f}  |  Epsilon: {agents[0].epsilon:.3f}\n")

# Save models
for agent in agents:
    agent.save(os.path.join(SAVE_DIR, f"agent_{agent.agent_id}.pkl"))

# ── Plot results ─────────────────────────────────────────
window = 30
smoothed_cov = np.convolve(episode_coverages, np.ones(window)/window, mode='valid')
smoothed_rew = np.convolve(episode_rewards,   np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(smoothed_cov * 100, color='steelblue')
axes[0].set_title("Site Coverage Over Training")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Coverage (%)")
axes[0].axhline(y=80, color='red', linestyle='--', label='80% target')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(smoothed_rew, color='darkorange')
axes[1].set_title("Total Reward Over Training")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Reward")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()
print("\nTraining complete! Results saved to training_results.png")
print(f"Final average coverage (last 50 eps): {np.mean(episode_coverages[-50:])*100:.1f}%")
