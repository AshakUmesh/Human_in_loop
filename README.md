# Human_in_loop
Autonomous driving systems aim to improve road safety but struggle in complex environments. This project develops a Human-in-the-Loop (HITL) DRL agent that integrates real-time human interventions into both control and learning. The approach improves safety, training efficiency, and performance while reducing reliance on expert demonstrations.


✅ STEP 1 — Create Project Folder

On your local machine first (not HPC).

mkdir human_loop_rl
cd human_loop_rl
✅ STEP 2 — Create Python Environment

Use venv:

python3 -m venv venv
source venv/bin/activate

Windows:

venv\Scripts\activate
✅ STEP 3 — Install Required Libraries

Install minimal packages:

pip install gymnasium[box2d]
pip install torch
pip install numpy
pip install matplotlib

These are enough for the first prototype.

✅ STEP 4 — Test the Environment

Create file:

test_env.py

import gymnasium as gym

env = gym.make("CarRacing-v2", render_mode="human")

obs, _ = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()

Run:

python test_env.py