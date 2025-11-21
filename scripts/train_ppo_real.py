#!/usr/bin/env python3
"""
PPO Training with Stable-Baselines3
"""
import os
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.expanduser("~/habitat_data/habitat-lab/habitat-lab"))
sys.path.insert(0, os.path.expanduser("~/habitat_rl_sorting/src"))

from gymnasium_wrapper import ColorSortingGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


def train_ppo(
    config_path: str,
    total_timesteps: int = 100000,
    log_dir: str = "./logs",
    model_save_path: str = "./models"
):
    """Train PPO agent"""
    print("=" * 80)
    print("Training PPO on Color Sorting Task")
    print("=" * 80)
    
    # Create directories
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    
    print("\n[1/4] Creating environment...")
    env = ColorSortingGymEnv(
        config_path=config_path,
        render_mode="rgb_array",
        max_episode_steps=500
    )
    env = Monitor(env, log_dir)
    print("✓ Environment created")
    
    print("\n[2/4] Initializing PPO agent...")
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cuda",
    )
    print("✓ PPO agent initialized")
    
    print("\n[3/4] Setting up callbacks...")
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_save_path,
        name_prefix="ppo_color_sorting"
    )
    print("✓ Callbacks configured")
    
    print(f"\n[4/4] Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_path = os.path.join(model_save_path, "ppo_color_sorting_final")
    model.save(final_path)
    print(f"\n✓ Model saved to {final_path}")
    
    env.close()
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, 
                       default="~/habitat_rl_sorting/configs/color_sorting.yaml")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--log-dir", type=str, default="~/habitat_rl_sorting/logs")
    parser.add_argument("--model-dir", type=str, default="~/habitat_rl_sorting/models")
    
    args = parser.parse_args()
    
    train_ppo(
        config_path=os.path.expanduser(args.config),
        total_timesteps=args.timesteps,
        log_dir=os.path.expanduser(args.log_dir),
        model_save_path=os.path.expanduser(args.model_dir)
    )
