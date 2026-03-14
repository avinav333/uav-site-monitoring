# UAV-Assisted Construction Site Monitoring using Multi-Agent RL

A multi-agent reinforcement learning simulation where 3 UAVs cooperatively monitor a construction site, navigating around dynamic obstacles (heavy machinery, scaffolding) with limited communication.

## Project Structure
```
uav-site-monitoring/
├── env.py          # Construction site grid environment
├── agent.py        # Independent Q-Learning agent
├── train.py        # Training loop + plots
├── evaluate.py     # Load & evaluate trained agents
└── requirements.txt
```

## How It Works
- **Environment:** 12×12 grid representing a construction site with 10 randomly placed obstacle zones
- **Agents:** 3 UAVs, each with an independent Q-Learning brain
- **Reward:** +1 for visiting a new cell, −0.05 for revisiting — encourages full coverage
- **Goal:** Maximise site coverage within 200 steps without centralised control

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train agents (800 episodes, ~1-2 min)
python train.py

# 3. Evaluate trained agents
python evaluate.py
```

## Results
After training, agents achieve ~83% site coverage under randomised obstacle configurations, demonstrating robust coordination without a central controller.

Training curve is saved as `training_results.png` after running `train.py`.

## Grid Legend
```
.  = unvisited cell
*  = visited cell
X  = obstacle (no-fly zone)
1/2/3 = UAV positions
```

## Relevance
This project is aligned with the **Mission Sudarshan Chakra** initiative under Atmanirbhar Bharat — demonstrating AI-enabled multi-UAV coordination under uncertainty for national security applications.

## Author
Abhinava Mondal — B.Tech Construction Engineering, Jadavpur University
