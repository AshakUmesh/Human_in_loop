🚗 Human-in-the-Loop Reinforcement Learning for Autonomous Driving

Autonomous driving systems promise safer and more efficient transportation. However, fully autonomous agents struggle in complex, unpredictable environments where purely automated decision-making can lead to unsafe behaviors.

This project explores a Human-in-the-Loop (HITL) Deep Reinforcement Learning framework where human interventions are integrated directly into the learning and control loop. By allowing humans to guide the agent during critical situations, the system can learn safer policies faster while reducing the need for expensive expert demonstrations.

The implementation uses Deep Reinforcement Learning with Gymnasium's CarRacing environment, enabling interactive human feedback during training.

🎯 Project Goals

The main objectives of this project are:

• Improve training efficiency of reinforcement learning agents
• Enhance safety during learning by allowing human override
• Reduce dependence on large expert datasets
• Study how human guidance shapes policy learning

This work aims to bridge the gap between fully autonomous systems and human-assisted learning frameworks.

🧠 Core Idea: Human-in-the-Loop Learning

Traditional reinforcement learning relies purely on environment rewards.

In contrast, Human-in-the-Loop Reinforcement Learning introduces a human supervisor who can:

• Intervene when the agent makes unsafe decisions
• Provide corrective actions
• Guide exploration in difficult states

This leads to:

✔ safer exploration
✔ faster convergence
✔ improved policy robustness

🏗️ Project Structure
human_loop_rl/
│
├── venv/                 # Python virtual environment
├── test_env.py           # Environment testing script
│
├── agents/               # RL agent implementations (future)
├── human_interface/      # Human control integration (future)
├── training/             # Training scripts
├── utils/                # Helper functions
│
└── README.md
⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/YOUR_USERNAME/human_loop_rl.git
cd human_loop_rl
2️⃣ Create Python Environment

Linux / Mac:

python3 -m venv venv
source venv/bin/activate

Windows:

python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies

Install the required libraries:

pip install gymnasium[box2d]
pip install torch
pip install numpy
pip install matplotlib

These packages are sufficient for the initial prototype.

🚀 Running the Environment

To verify that the setup works, run the test script.

Create the file:

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

If everything is installed correctly, a CarRacing simulation window will appear with a randomly driving agent.

🧪 Research Direction

This project will progressively implement:

• Deep Reinforcement Learning agents (PPO / SAC / DQN variants)
• Real-time human intervention interface
• Intervention-based reward shaping
• Human feedback learning
• Policy improvement using corrective demonstrations

Potential research topics include:

Human-guided exploration

Safety-aware reinforcement learning

Interactive policy shaping

Human feedback reward modeling

📊 Environment

The project uses:

Gymnasium – CarRacing-v2

Observation Space:

96 × 96 × 3 RGB image

Action Space:

[Steering, Gas, Brake]
Steering ∈ [-1, 1]
Gas      ∈ [0, 1]
Brake    ∈ [0, 1]
🧰 Tech Stack

Python
PyTorch
Gymnasium
NumPy
Matplotlib

🔬 Future Work

Planned improvements include:

• Human keyboard control integration
• Intervention logging
• Offline learning from interventions
• Reward shaping via human feedback
• Deployment on HPC for large-scale training

🤝 Contributions

Contributions are welcome. Feel free to open issues or submit pull requests for improvements.
