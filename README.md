# Human-in-the-Loop Deep Reinforcement Learning for Autonomous Driving in Complex Urban Scenarios

> **A TD3-based framework where real-time human intervention directly shapes agent learning in the CARLA simulator.**

---

## Overview

This repository implements a **Human-in-the-Loop (HITL) Deep Reinforcement Learning** pipeline for autonomous vehicle control. A human operator can intervene at any timestep during training — overriding the agent's steering action via mouse or keyboard — and those interventions are stored as privileged demonstrations that accelerate and guide the learning process.

The environment is built on **CARLA 0.9.15** and presents the ego vehicle with five concurrent hazard types in a single episode, requiring the agent to learn nuanced avoidance strategies that pure RL often struggles to discover from scratch.

---

## Key Contributions

- **Real-time human override loop**: holding the right mouse button activates human control; releasing it returns control to the AI. Intervention transitions are flagged (`intervention=1`) and handled separately in the replay buffer.
- **Multi-hazard CARLA environment**: five simultaneous obstacle types — wrong-way parked car, oncoming vehicle, jaywalking pedestrian, crosswalk pedestrian, and sidewalk walker.
- **Four algorithm variants** on a shared TD3 backbone, differing in how human feedback is incorporated into policy updates.
- **Multiple reward shaping strategies**: no shaping, intervention-penalty, potential-based, and RND (Random Network Distillation) intrinsic reward.
- **PID controller as virtual human**: enables automated testing of the HITL pipeline without a live operator.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Loop                           │
│                                                             │
│   ┌──────────┐    action_ai     ┌─────────────────────┐    │
│   │  TD3     │ ───────────────► │   CARLA Environment  │    │
│   │  Agent   │                  │  (5 hazard types)    │    │
│   └──────────┘                  └──────────┬──────────┘    │
│        ▲                                   │               │
│        │  learn()               state, reward, done        │
│        │                                   │               │
│   ┌────┴─────────────────────────────────┐ │               │
│   │         Replay Buffer                │◄┘               │
│   │  • AI transitions  (intervention=0)  │                 │
│   │  • Human transitions (intervention=1)│                 │
│   └──────────────────────────────────────┘                 │
│                        ▲                                    │
│              Human holds RMB → steer with mouse            │
│              Releases RMB   → AI resumes control           │
└─────────────────────────────────────────────────────────────┘
```

---

## Environment: Multi-Obstacle CARLA Scenario

The ego vehicle (Tesla blueprint) spawns at a fixed start point and must navigate 70 metres of road to a goal zone. Five obstacles are spawned simultaneously each episode:

| # | Obstacle Type | Behavior |
|---|--------------|----------|
| 1 | Wrong-way parked car | Static, occupying ego lane |
| 2 | Oncoming vehicle | Moving toward ego at −3 m/s |
| 3 | Jaywalking pedestrian | Crossing road at 1.2 m/s |
| 4 | Crosswalk pedestrian | Crossing at 0.8 m/s at a marked crossing |
| 5 | Sidewalk walker | Walking parallel to road at 0.5 m/s |

**State space** (12-dimensional): ego (x, y) + relative (Δx, Δy) to each of the 5 obstacles.

**Action space** (1-dimensional continuous): steering in [−1, 1]. Throttle is held fixed at 0.35.

**Reward function**:
```
r = 0.2                              # time-step survival bonus
  - 1.5 × |x_ego − x_lane_center|   # lane-keeping penalty
  - Σ proximity_penalty(obstacle_i)  # proximity to each obstacle
  + 10  (on reaching goal)           # terminal success bonus
  − 10  (on collision)               # terminal failure penalty
```

---

## Algorithm Variants

All four variants share the TD3 backbone (twin critics, delayed actor updates, target policy smoothing). They differ in how human feedback transitions are used during the learning step:

| Algorithm | Description |
|-----------|-------------|
| **TD3HUG** | Human-Guided: intervention transitions receive upweighted gradient updates |
| **TD3IARL** | Intervention-Aware RL: Q-value at moment of takeover is logged and used to shape the critic loss |
| **TD3HIRL** | Human-Initiated RL: policy gradient is suppressed when human and agent agree; amplified on disagreement |
| **TD3** (baseline) | Vanilla TD3, human interventions stored as ordinary transitions |

---

## Reward Shaping Strategies

| Flag | Strategy | Description |
|------|----------|-------------|
| `0` | None | Extrinsic reward only |
| `1` | Intervention penalty | −10 intrinsic reward at first intervention step per episode |
| `2` | Potential-based | Intrinsic reward proportional to remaining distance to goal |
| `3` | RND | Novelty-based intrinsic reward via Random Network Distillation; encourages exploration of unseen states |

---

## Controls (During Training)

| Input | Action |
|-------|--------|
| Hold **Right Mouse Button** | Activate human takeover |
| Move mouse left / right | Analog steering |
| **A** / **←** | Steer left (additive) |
| **D** / **→** | Steer right (additive) |
| **S** / **↓** | Centre steering |
| Release **RMB** | Return control to AI agent |

---

## Repository Structure

```
Human_In_Loop/
├── README.md
├── requirements.txt
├── env.py                                  # Pygame + CARLA env wrapper (mouse/keyboard HITL)
├── utils.py                                # Seed, signal handler, path generator, RND module
├── train_offline_latest.py                 # Main training script
├── train_offline.py                        # Earlier training script (for reference)
├── carla_hitl_multi_obstacle_environment.py # Multi-obstacle CARLA scene definition
├── wheel_config.ini                        # Config for G29 steering wheel (optional hardware)
├── TD3_based_DRL/                          # TD3 algorithm variants
│   ├── TD3.py
│   ├── TD3HUG.py
│   ├── TD3IARL.py
│   ├── TD3HIRL.py
│   └── checkpoints/
├── episode_data/                           # Saved episode trajectories (.mat)
└── results.png                             # Training reward curves
```

---

## Setup

### Requirements

- Ubuntu 20.04 / 22.04
- Python 3.8+
- CARLA 0.9.15 ([download](https://github.com/carla-simulator/carla/releases/tag/0.9.15))
- CUDA-capable GPU (recommended)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch CARLA server

```bash
./CarlaUE4.sh -RenderOffScreen   # headless
# or
./CarlaUE4.sh                    # with rendering
```

### Run training

```bash
# Vanilla TD3 (baseline)
python train_offline_latest.py --algorithm 3

# TD3HUG with RND reward shaping
python train_offline_latest.py --algorithm 0 --reward_shaping 3

# With PID controller as virtual human (no live operator needed)
python train_offline_latest.py --algorithm 0 --pid_controller_guidance

# Resume from checkpoint
python train_offline_latest.py --algorithm 0 --resume
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--algorithm` | `0` | 0=TD3HUG, 1=TD3IARL, 2=TD3HIRL, 3=TD3 |
| `--reward_shaping` | `0` | 0=none, 1=intervention, 2=potential, 3=RND |
| `--maximum_episode` | `1000` | Total training episodes |
| `--pid_controller_guidance` | `False` | Use PID as virtual human |
| `--warmup` | `False` | Collect random transitions before learning |
| `--device` | `cuda` | Training device |

### Monitor training

```bash
tensorboard --logdir TD3_based_DRL/checkpoints/log
```

---

## Results

![Training Results](results.png)

Episode rewards, intervention frequency, and loss curves are logged to TensorBoard and saved as `.mat` files for offline analysis in MATLAB/Python.

---

## Design Decisions

**Why TD3 over SAC or PPO?**
TD3's deterministic policy makes it straightforward to blend human and agent actions at inference time — there is no stochastic sampling step to reconcile. The twin critics also give a reliable Q-value signal at the moment of human takeover, which TD3IARL uses directly.

**Why log Q-values at takeover moments?**
When a human intervenes, the agent's Q-value at that state captures how confident the agent was in its (apparently wrong) action. Tracking this over training reveals whether the agent is learning to be uncertain in genuinely dangerous states — a useful diagnostic for HITL systems.

**Why RND for intrinsic reward?**
CARLA environments are large and sparsely rewarded. RND gives the agent an intrinsic signal to explore novel states rather than converging on a single safe trajectory, which matters when five obstacle configurations vary each episode.

**PID as virtual human**
The PID controller activates only when the ego vehicle deviates significantly from the reference path near obstacle clusters. This lets the HITL pipeline be evaluated and iterated without requiring a human operator for every run.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{ashakumesh2024hitl,
  author       = {Ashak Umesh},
  title        = {Human-in-the-Loop Deep Reinforcement Learning for Autonomous Driving},
  year         = {2024},
  howpublished = {\url{https://github.com/AshakUmesh/Human_In_Loop}}
}
```

---

## License

This project is licensed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE) for details.
