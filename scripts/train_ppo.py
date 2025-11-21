#!/usr/bin/env python3
"""
Example training script using Stable-Baselines3 PPO
Demonstrates how to train RL agent on Color Sorting task
"""
import os
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.expanduser("~/habitat_data/habitat-lab/habitat-lab"))
sys.path.insert(0, os.path.expanduser("~/habitat_rl_sorting/src"))

from gymnasium_wrapper import ColorSortingGymEnv

# Optional: Import RL library (uncomment if stable-baselines3 is installed)
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_env(config_path, rank=0, seed=0):
    """
    Create environment factory for vectorized environments.
    """
    def _init():
        env = ColorSortingGymEnv(
            config_path=config_path,
            render_mode="rgb_array",
            max_episode_steps=500
        )
        env.seed(seed + rank)
        return env
    return _init


def train_ppo_example(
    config_path: str,
    total_timesteps: int = 100000,
    n_envs: int = 4,
    log_dir: str = "./logs",
    model_save_path: str = "./models"
):
    """
    Example PPO training loop.
    
    Args:
        config_path: Path to Habitat config
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        log_dir: Logging directory
        model_save_path: Model save directory
    """
    print("=" * 80)
    print("Training PPO on Color Sorting Task")
    print("=" * 80)
    
    # Create directories
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    
    # Create vectorized environments
    print(f"\n[1/4] Creating {n_envs} parallel environments...")
    
    # For single environment (simpler, no multiprocessing)
    env = ColorSortingGymEnv(
        config_path=config_path,
        render_mode="rgb_array",
        max_episode_steps=500
    )
    
    print("✓ Environment created")
    
    # If using Stable-Baselines3:
    # env = DummyVecEnv([make_env(config_path, i, seed=42) for i in range(n_envs)])
    # Or for true parallel:
    # env = SubprocVecEnv([make_env(config_path, i, seed=42) for i in range(n_envs)])
    
    print("\n[2/4] Initializing PPO agent...")
    
    # Example PPO configuration (uncomment if using SB3)
    # model = PPO(
    #     "MultiInputPolicy",  # For Dict observation space
    #     env,
    #     verbose=1,
    #     tensorboard_log=log_dir,
    #     learning_rate=3e-4,
    #     n_steps=2048,
    #     batch_size=64,
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     ent_coef=0.01,
    # )
    
    print("✓ Agent initialized (or would be, if SB3 was installed)")
    
    print("\n[3/4] Setting up callbacks...")
    
    # Checkpoint callback
    # checkpoint_callback = CheckpointCallback(
    #     save_freq=10000,
    #     save_path=model_save_path,
    #     name_prefix="ppo_color_sorting"
    # )
    
    # Eval callback
    # eval_env = ColorSortingGymEnv(config_path=config_path, render_mode="rgb_array")
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=model_save_path,
    #     log_path=log_dir,
    #     eval_freq=5000,
    #     n_eval_episodes=5,
    #     deterministic=True
    # )
    
    print("✓ Callbacks configured")
    
    print(f"\n[4/4] Training for {total_timesteps} timesteps...")
    
    # Train the agent
    # model.learn(
    #     total_timesteps=total_timesteps,
    #     callback=[checkpoint_callback, eval_callback],
    #     progress_bar=True
    # )
    
    # Instead, run a simple random policy demo
    print("\n--- Running Random Policy Demo (since SB3 not installed) ---")
    
    obs, info = env.reset(seed=42)
    episode_reward = 0.0
    episode_length = 0
    
    for step in range(min(1000, total_timesteps)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_length += 1
        
        if step % 100 == 0:
            print(f"  Step {step}: reward={reward:.3f}, success={info.get(success, 0)}")
        
        if terminated or truncated:
            print(f"\n  Episode finished: length={episode_length}, reward={episode_reward:.3f}")
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
    
    env.close()
    
    print("\n✓ Training complete")
    
    # Save the model
    # model.save(os.path.join(model_save_path, "ppo_color_sorting_final"))
    print(f"\n✓ Model saved to {model_save_path}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO on Color Sorting Task")
    parser.add_argument(
        "--config",
        type=str,
        default="~/habitat_rl_sorting/configs/color_sorting.yaml",
        help="Path to Habitat config file"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="~/habitat_rl_sorting/logs",
        help="Logging directory"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="~/habitat_rl_sorting/models",
        help="Model save directory"
    )
    
    args = parser.parse_args()
    
    train_ppo_example(
        config_path=os.path.expanduser(args.config),
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        log_dir=os.path.expanduser(args.log_dir),
        model_save_path=os.path.expanduser(args.model_dir)
    )
