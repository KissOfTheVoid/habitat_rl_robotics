# Habitat-Sim RL Color Sorting Environment

Reinforcement Learning environment for robotic color sorting using Habitat-Sim and Stable-Baselines3 PPO.

## Overview

This project implements a custom RL environment where a Fetch robot learns to sort colored objects into matching zones using dense reward shaping and parallel training.

## Key Features

- **Dense Reward Shaping**: Multi-phase reward (approach, pick, transport, place) for fast learning
- **Parallel Training**: 32 parallel environments via SubprocVecEnv (~3200 FPS effective)
- **Custom Habitat Task**: ColorSortingTask-v0 with dynamic object spawning
- **Advanced PPO**: Linear LR schedule, entropy monitoring, automatic checkpointing
- **Custom Sensors**: Track object/zone positions and colors via proprioceptive sensors
- **Gymnasium Wrapper**: Standard interface for Stable-Baselines3

## Project Structure

```
habitat_rl_sorting/
├── configs/
│   └── color_sorting.yaml       # Task configuration
├── scripts/
│   ├── train_ppo_improved.py    # Main training script (with dense rewards)
│   └── train_ppo_optimized.py   # Legacy sparse reward version
├── src/
│   ├── color_sorting_task.py    # Custom Habitat task
│   ├── color_sorting_sensors.py # Sensors, dense rewards, metrics
│   └── gymnasium_wrapper.py     # Gym interface for SB3
└── data/
    └── default.physics_config.json
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (required for training)
- Habitat-Sim 0.3.3
- Habitat-Lab

### Setup

```bash
# Install Habitat-Sim with GPU support
conda install habitat-sim=0.3.3 -c conda-forge -c aihabitat

# Install Habitat-Lab
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab && pip install -e habitat-lab

# Install RL dependencies
pip install stable-baselines3 gymnasium tensorboard

# Clone this repository
git clone <your-repo-url>
cd habitat_rl_sorting
```

## Quick Start

### Train PPO Agent (Recommended)

```bash
# Full training: 50M steps with 32 parallel envs
python scripts/train_ppo_improved.py \
  --timesteps 50000000 \
  --n-envs 32 \
  --checkpoint-freq 500000 \
  --eval-freq 100000

# Resume from checkpoint
python scripts/train_ppo_improved.py \
  --timesteps 50000000 \
  --n-envs 32 \
  --resume models/PPO_improved_*/best_model.zip
```

### Monitor Training

```bash
# TensorBoard (24 metrics tracked)
tensorboard --logdir=logs --port=6006

# Watch training logs
tail -f logs/PPO_improved_*/training.log
```

## Action Space

**Box(4,)** - Continuous actions for end-effector control:

| Index | Component | Range | Description |
|-------|-----------|-------|-------------|
| 0-2 | delta_xyz | [-0.1, 0.1] | End-effector displacement (m) |
| 3 | gripper | {-1, +1} | Gripper command (open/close) |

Actions are converted to joint positions via inverse kinematics solver.

## Observation Space

**Dict** with keys:
- **joint**: Box(7,) - Joint positions (rad)
- **is_holding**: Box(1,) - Binary grasp indicator
- **object_color_sensor**: Box(3, 4) - Object positions + color IDs
- **color_zone_sensor**: Box(3, 4) - Target zone positions + color IDs

## Dense Reward Structure

Multi-phase reward for efficient learning:

```python
r_t = r_approach + r_pick + r_transport + r_place + r_drop + r_completion + r_time

# APPROACH (not holding): +2.0 * distance_reduced_to_object
# PICK (grasp): +10.0
# TRANSPORT (holding): +2.0 * distance_reduced_to_zone  
# PLACE (correct): +50.0
# DROP (incorrect): -5.0
# COMPLETION (all placed): +100.0
# TIME (always): -0.001
```

This design guides the robot through: reach → grasp → carry → place.

## PPO Hyperparameters

```python
{
    "learning_rate": linear_schedule(3e-4, 1e-5),  # Decays over training
    "n_steps": 2048,           # Rollout length per env
    "batch_size": 4096,        # Minibatch for SGD
    "n_epochs": 10,            # Optimization epochs per rollout
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.02,          # CRITICAL: prevents entropy collapse
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}
```

**Effective batch size**: 2048 steps × 32 envs = 65,536 transitions per update.

## Training Performance

**Hardware**: NVIDIA GPU (A100/V100/RTX 3090+)

**Metrics** (after 50M steps):
- **Success rate**: 98-99% (reward > -5)
- **Median reward**: -0.60 (near-optimal)
- **FPS**: ~3200 steps/sec (32 parallel envs × 100 FPS each)
- **Training time**: ~4-5 hours for 50M steps

## Debugging & Monitoring

### Entropy Collapse Detection

**Symptom**: Policy becomes deterministic, reward degrades
**Cause**: `ent_coef` too low (e.g., 0.005)
**Solution**: Increase to 0.02, restart from best checkpoint

Monitor entropy in TensorBoard:
```python
if entropy < -15.0:
    print("WARNING: Entropy collapse detected!")
```

### TensorBoard Metrics

24 tracked metrics:
- `train/*`: policy loss, value loss, entropy
- `rollout/*`: mean reward, episode length
- `metrics/*`: success rate, reward percentiles (p25/p50/p75)
- `eval/*`: evaluation performance

## Troubleshooting

### Low GPU utilization
- Physics simulation is CPU-bound
- Use more parallel environments (`--n-envs 64`)
- Increase `n_steps` and `batch_size`

### Reward collapse
- Check `ent_coef` is not too low (< 0.01)
- Verify reward coefficients are balanced
- Monitor TensorBoard for catastrophic episodes

### Training instability
- Reduce learning rate
- Decrease `max_grad_norm`
- Check for NaN values in logs

## Advanced Usage

### Curriculum Learning
Gradually increase task difficulty:
```python
# Start with 1 object, increase to 3
num_objects: 1  # → 2 → 3
```

### Domain Randomization
```yaml
spawn_region:
  x_range: [-0.5, 0.5]  # Randomize object positions
  y_range: [1.2, 1.6]
  z_range: [-0.3, 0.3]
```

## Documentation

- **Lecture_RL_Sorting.md** - Academic-style guide with mathematical formalism
- **TRAINING_PIPELINE.md** - Detailed pipeline documentation
- **CHANGELOG.md** - Version history and updates

## Citation

If you use this code, please cite:
```
@misc{habitat_rl_sorting,
  title={Dense Reward Shaping for Robotic Manipulation in Habitat-Sim},
  year={2024},
  url={https://github.com/yourusername/habitat_rl_sorting}
}
```

## License

MIT License

## Acknowledgments

- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) - 3D simulation
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) - Task framework
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - PPO implementation
