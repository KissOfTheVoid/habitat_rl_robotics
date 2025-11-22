# Changelog

## [2.0.0] - 2024-11-22

### Summary
Implemented dense reward system to replace sparse rewards, significantly improving learning signal for the color sorting task.

### Added

#### New Reward Class: `ColorSortingDenseReward`
Multi-phase dense reward implementation in `src/color_sorting_sensors.py`:

| Phase | Reward | Description |
|-------|--------|-------------|
| APPROACH | Distance-based | Reward for moving EE towards nearest unplaced object |
| PICK | +10.0 | Bonus for successfully grasping an object |
| TRANSPORT | Distance-based | Reward for moving held object towards target zone |
| PLACE | +50.0 | Bonus for correct placement |
| COMPLETION | +100.0 | Extra bonus when all objects sorted |
| Time penalty | -0.001 | Reduced from -0.01 |
| Drop penalty | -5.0 | Penalty for dropping without placing |

#### Default Parameters (in code)
```python
dist_reward_scale = 1.0
pick_reward = 10.0
place_reward = 50.0
drop_penalty = -5.0
wrong_zone_penalty = -10.0
time_penalty = -0.001
completion_bonus = 100.0
success_distance = 0.1
```

### Changed

#### Configuration (`configs/color_sorting.yaml`)
- `reward_measure`: `color_sorting_reward` -> `color_sorting_dense_reward`
- `slack_reward`: -0.01 -> -0.001
- Added `color_sorting_dense_reward` measurement config

#### PPO Hyperparameters (`scripts/train_ppo_real.py`)
| Parameter | Sparse | Dense | Reason |
|-----------|--------|-------|--------|
| learning_rate | 3e-4 | 1e-4 | Stability with dense gradients |
| batch_size | 64 | 128 | Smoother gradient estimates |
| clip_range | 0.2 | 0.1 | Prevent large policy updates |
| ent_coef | 0.01 | 0.001 | More exploitation |
| vf_coef | - | 0.5 | Added value function coefficient |
| max_grad_norm | - | 0.5 | Added gradient clipping |

#### Training Script
- Added `--sparse` flag to use original sparse reward
- Separate hyperparameter profiles for dense/sparse rewards

### Kept (Backwards Compatibility)
- Original `ColorSortingReward` class preserved
- Use `--sparse` flag to train with original reward

### Files Modified
1. `src/color_sorting_sensors.py` - Added ColorSortingDenseReward class
2. `configs/color_sorting.yaml` - Updated reward_measure to dense
3. `scripts/train_ppo_real.py` - Added dense reward hyperparameters

### Initial Test Results
Quick test (1000 steps):
- ep_rew_mean: -0.236 (vs -3.94 with sparse)
- FPS: 178

---

## [1.0.0] - 2024-11-21

### Initial Release
- ColorSortingTask-v0 implementation
- Sparse reward system (ColorSortingReward)
- PPO training with Stable-Baselines3
- Gymnasium wrapper for Habitat-Sim
- Basic sensors: ColorZoneSensor, ObjectColorSensor
