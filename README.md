# Habitat-Sim RL Color Sorting Environment

Reinforcement Learning environment for robotic color sorting using Habitat-Sim 0.3.3 and Stable-Baselines3.

## Overview

This project implements a custom RL environment where a Fetch robot with suction gripper learns to sort colored objects into matching zones using PPO (Proximal Policy Optimization).

## Features

- **Custom Habitat Task**: ColorSortingTask-v0 for robotic manipulation
- **Custom Sensors**: Track object positions, colors, and target zones
- **Gymnasium Wrapper**: Standard interface for Stable-Baselines3
- **PPO Training**: Tested on NVIDIA A100 GPU (around 99 FPS)
- **Action Space**: 10D continuous control (7-DOF arm + gripper + mobile base)
- **Reward System**: Sparse rewards for correct object placement

## Project Structure

```
habitat_rl_robotics/
├── configs/
│   ├── color_sorting.yaml       # Task configuration
│   └── env_config.yaml          # Environment settings
├── scripts/
│   └── train_ppo_real.py        # Main training script
├── src/
│   ├── color_sorting_task.py    # Custom Habitat task
│   ├── color_sorting_sensors.py # Sensors, rewards, metrics
│   └── gymnasium_wrapper.py     # Gym interface for SB3
├── data/
│   └── default.physics_config.json
└── test_env.py                  # Environment tests
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- Habitat-Sim 0.3.3
- Habitat-Lab

### Setup

```bash
# Install Habitat-Sim with GPU support
conda install habitat-sim=0.3.3 -c conda-forge -c aihabitat

# Install Habitat-Lab
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab

# Install RL dependencies
pip install stable-baselines3 gymnasium tensorboard

# Clone this repository
git clone https://github.com/KissOfTheVoid/habitat_rl_robotics.git
cd habitat_rl_robotics
```

## Quick Start

### Test Environment

```bash
python test_env.py
```

### Train PPO Agent

```bash
# Quick test (5 minutes)
python scripts/train_ppo_real.py --timesteps 10000

# Full training (1M timesteps, about 2 hours on A100)
python scripts/train_ppo_real.py --timesteps 1000000

# With nohup for persistent training
nohup python scripts/train_ppo_real.py --timesteps 1000000 > logs/training.log 2>&1 &
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir=logs --port=6006

# Watch logs
tail -f logs/training.log
```

## Action Space

**Box(10,)** - Continuous actions in range [-1, 1]:

| Index | Component | Description |
|-------|-----------|-------------|
| 0-6 | arm_action | 7-DOF Fetch arm joint positions |
| 7 | grip_action | Suction gripper (negative=off, positive=on) |
| 8 | base_velocity_forward | Base forward/backward movement (scaled by 20) |
| 9 | base_velocity_rotation | Base rotation (scaled by 20) |

## Observation Space

**Dict** with keys:
- **rgb**: Box(256, 256, 3, uint8) - RGB camera view
- **state**: Box(N, float32) - Concatenated state vector:
  - Joint positions (7D)
  - Gripper state (1D)
  - Object positions + colors (num_objects × 4)
  - Zone positions + colors (3 × 4)

## Reward Structure

```
reward = -0.01  # Time penalty per step

for each object in correct zone:
    if first_time:
        reward += 100.0  # Placement bonus
    reward += 10.0       # Holding bonus per step
```

## Training Results

**Initial results** (A100 GPU, 1M timesteps):

- **FPS**: around 99-110 steps/sec
- **Episode Length**: 277 → 46 steps (-83% improvement)
- **Episode Reward**: -2.77 → -0.46 (+83% improvement)
- **Training Time**: around 2-3 hours for 1M steps

## Configuration

Key parameters in configs/color_sorting.yaml

## Troubleshooting

### Episode crashes with "Episode over, call reset"
- Check gymnasium_wrapper.py handles episode_over flag

### Action space incompatibility
- Ensure wrapper creates flattened Box(10,) action space
- Check _flatten_action() converts to Habitat format

### Low GPU utilization
- Physics simulation is CPU-bound
- Increase batch_size and n_steps in PPO config
- Use parallel environments with VecEnv

## Documentation

- TRAINING_PIPELINE.md - Detailed pipeline documentation
- LONG_TRAINING_GUIDE.md - Guide for long training sessions

## License

MIT License

## Acknowledgments

- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) - Physics simulation
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) - Task framework
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
