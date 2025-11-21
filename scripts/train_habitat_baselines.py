#!/usr/bin/env python3
"""
Training with Habitat-Baselines PPO
Uses official Habitat RL framework
"""
import os
import sys

# Add paths
sys.path.insert(0, os.path.expanduser("~/habitat_data/habitat-lab/habitat-lab"))
sys.path.insert(0, os.path.expanduser("~/habitat_rl_sorting/src"))

# Import custom components to register them
import color_sorting_task
import color_sorting_sensors

from habitat import get_config
from habitat_baselines.config.default import get_config as get_baseline_config
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer

def main():
    print("=" * 80)
    print("Training with Habitat-Baselines PPO")
    print("=" * 80)
    
    # Load task config
    task_config_path = os.path.expanduser("~/habitat_rl_sorting/configs/color_sorting.yaml")
    task_config = get_config(task_config_path)
    
    print("\n✓ Task config loaded")
    print(f"  Task: {task_config.habitat.task.type}")
    print(f"  Reward measure: {task_config.habitat.task.reward_measure}")
    
    # Get baseline config
    baseline_config = get_baseline_config()
    
    # Override with our task config
    baseline_config.habitat = task_config.habitat
    
    # Set training parameters
    baseline_config.habitat_baselines.num_environments = 1
    baseline_config.habitat_baselines.trainer_name = "ppo"
    baseline_config.habitat_baselines.total_num_steps = 10000
    baseline_config.habitat_baselines.log_interval = 10
    baseline_config.habitat_baselines.checkpoint_interval = 1000
    
    # PPO parameters
    baseline_config.habitat_baselines.rl.ppo.num_steps = 128
    baseline_config.habitat_baselines.rl.ppo.num_mini_batch = 2
    baseline_config.habitat_baselines.rl.ppo.lr = 2.5e-4
    baseline_config.habitat_baselines.rl.ppo.eps = 1e-5
    baseline_config.habitat_baselines.rl.ppo.value_loss_coef = 0.5
    baseline_config.habitat_baselines.rl.ppo.entropy_coef = 0.01
    baseline_config.habitat_baselines.rl.ppo.use_gae = True
    baseline_config.habitat_baselines.rl.ppo.gamma = 0.99
    baseline_config.habitat_baselines.rl.ppo.tau = 0.95
    
    # Logging
    baseline_config.habitat_baselines.tensorboard_dir = os.path.expanduser("~/habitat_rl_sorting/logs/tensorboard")
    baseline_config.habitat_baselines.checkpoint_folder = os.path.expanduser("~/habitat_rl_sorting/models/checkpoints")
    baseline_config.habitat_baselines.log_file = os.path.expanduser("~/habitat_rl_sorting/logs/train.log")
    
    print("\n✓ Baseline config prepared")
    print(f"  Total steps: {baseline_config.habitat_baselines.total_num_steps}")
    print(f"  PPO steps: {baseline_config.habitat_baselines.rl.ppo.num_steps}")
    
    # Create trainer
    print("\n[Creating trainer...]\n")
    
    trainer = PPOTrainer(baseline_config)
    
    print("\n✓ Trainer created")
    print("\n[Starting training...]\n")
    
    # Train
    trainer.train()
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
