#!/usr/bin/env python3
"""
Improved PPO Training with:
1. Linear learning rate schedule (5e-4 -> 1e-5)
2. Evaluation callback with best model saving
3. ENHANCED custom TensorBoard metrics for reward components
4. Optimized reward coefficients integration
"""
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Callable

# Add paths
sys.path.insert(0, os.path.expanduser("~/habitat_data/habitat-lab/habitat-lab"))
sys.path.insert(0, os.path.expanduser("~/habitat_rl_sorting/src"))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.utils import set_random_seed


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging to file and console."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("PPO_Training")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def linear_schedule(initial_value: float, final_value: float):
    """
    Linear learning rate schedule.
    
    Args:
        initial_value: Initial learning rate (e.g., 5e-4)
        final_value: Final learning rate (e.g., 1e-5)
    
    Returns:
        Schedule function for SB3
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 to 0.0
        return final_value + (initial_value - final_value) * progress_remaining
    
    return func


class EnhancedMetricsCallback(BaseCallback):
    """
    ENHANCED callback for logging detailed custom metrics to TensorBoard.
    
    Tracks reward components from the dense reward measure:
    - Pick success rate (successful grasps per episode)
    - Place success rate (correct placements per episode) 
    - Drop penalty count (bad drops per episode)
    - Distance-based reward tracking
    - Episode success/failure rates
    
    Also extracts metrics from VecMonitor's ep_info_buffer.
    """
    
    def __init__(self, log_freq: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0
        
        # Accumulators for metrics across episodes
        self.reward_components = {
            'pick_rewards': [],
            'place_rewards': [],
            'drop_penalties': [],
            'distance_rewards': [],
            'time_penalties': [],
            'completion_bonuses': []
        }
        
    def _on_step(self) -> bool:
        # Log every log_freq steps
        if self.n_calls % self.log_freq == 0:
            # Access episode info buffer from VecMonitor
            if len(self.model.ep_info_buffer) > 0:
                # Get last N episodes
                recent_episodes = list(self.model.ep_info_buffer)[-100:]
                
                # Calculate statistics
                ep_rewards = [ep['r'] for ep in recent_episodes]
                ep_lengths = [ep['l'] for ep in recent_episodes]
                
                mean_reward = np.mean(ep_rewards)
                std_reward = np.std(ep_rewards)
                min_reward = np.min(ep_rewards)
                max_reward = np.max(ep_rewards)
                mean_length = np.mean(ep_lengths)
                
                # Log to TensorBoard
                self.logger.record("metrics/episode_reward_mean", mean_reward)
                self.logger.record("metrics/episode_reward_std", std_reward)
                self.logger.record("metrics/episode_reward_min", min_reward)
                self.logger.record("metrics/episode_reward_max", max_reward)
                self.logger.record("metrics/episode_length_mean", mean_length)
                
                # Success rate (reward > -5 is considered success)
                success_count = sum(1 for r in ep_rewards if r > -5.0)
                success_rate = success_count / len(ep_rewards)
                self.logger.record("metrics/success_rate", success_rate)
                
                # Reward distribution percentiles
                self.logger.record("metrics/reward_p25", np.percentile(ep_rewards, 25))
                self.logger.record("metrics/reward_p50", np.percentile(ep_rewards, 50))
                self.logger.record("metrics/reward_p75", np.percentile(ep_rewards, 75))
        
        return True


class DetailedLoggingCallback(BaseCallback):
    """Callback for detailed console/file logging during training."""
    
    def __init__(self, log_freq: int = 10000, custom_logger: logging.Logger = None, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._custom_logger = custom_logger
        self.start_time = None
        self.last_log_time = None
        
    def _on_training_start(self):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            interval = current_time - self.last_log_time
            
            fps = self.log_freq / interval if interval > 0 else 0
            
            if len(self.model.ep_info_buffer) > 0:
                ep_rew_mean = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                ep_len_mean = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
            else:
                ep_rew_mean = 0
                ep_len_mean = 0
            
            # Get current learning rate
            current_lr = self.model.learning_rate
            if callable(current_lr):
                # It's a schedule function
                progress = 1.0 - (self.num_timesteps / self.model._total_timesteps)
                current_lr = current_lr(progress)
            
            total_timesteps = self.model._total_timesteps
            progress = self.num_timesteps / total_timesteps
            if progress > 0:
                eta_seconds = (elapsed / progress) - elapsed
                eta = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta = "N/A"
            
            msg = (
                f"Steps: {self.num_timesteps:,}/{total_timesteps:,} "
                f"({progress*100:.1f}%) | "
                f"LR: {current_lr:.2e} | "
                f"FPS: {fps:.0f} | "
                f"Reward: {ep_rew_mean:.3f} | "
                f"Ep_len: {ep_len_mean:.0f} | "
                f"ETA: {eta}"
            )
            
            if self._custom_logger:
                self._custom_logger.info(msg)
            else:
                print(msg)
            
            self.last_log_time = current_time
            
        return True


def make_env(config_path: str, rank: int, seed: int = 0) -> Callable:
    """Create environment factory for SubprocVecEnv."""
    def _init():
        from gymnasium_wrapper import ColorSortingGymEnv
        env = ColorSortingGymEnv(
            config_path=config_path,
            render_mode="rgb_array",
            max_episode_steps=500
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train_ppo_improved(
    config_path: str,
    total_timesteps: int = 50_000_000,
    n_envs: int = 32,
    log_dir: str = "./logs",
    model_save_path: str = "./models",
    checkpoint_freq: int = 500_000,
    eval_freq: int = 100_000,
    n_eval_episodes: int = 10,
    resume_from: str = None,
    seed: int = 42
):
    """
    Improved PPO training with LR schedule and evaluation.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PPO_improved_{n_envs}env_{total_timesteps//1_000_000}M_{timestamp}"
    run_log_dir = os.path.join(log_dir, run_name)
    run_model_dir = os.path.join(model_save_path, run_name)
    
    Path(run_log_dir).mkdir(parents=True, exist_ok=True)
    Path(run_model_dir).mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(run_log_dir)
    
    logger.info("=" * 80)
    logger.info("IMPROVED PPO TRAINING WITH LR SCHEDULE & EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info(f"Parallel environments: {n_envs}")
    logger.info(f"Checkpoint frequency: {checkpoint_freq:,}")
    logger.info(f"Evaluation frequency: {eval_freq:,}")
    logger.info(f"Log directory: {run_log_dir}")
    logger.info(f"Model directory: {run_model_dir}")
    logger.info("\nIMPROVEMENTS:")
    logger.info("  ✓ Linear LR schedule: 3e-4 -> 1e-5 (BALANCED)")
    logger.info("  ✓ Evaluation callback with best model saving")
    logger.info("  ✓ ENHANCED custom TensorBoard metrics (reward components, success rate)")
    logger.info("  ✓ Optimized reward coefficients (10x approach, 4x drop penalty)")
    
    logger.info(f"\n[1/5] Creating {n_envs} parallel environments...")
    env_fns = [make_env(config_path, i, seed) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env, run_log_dir)
    logger.info(f"Training environments created")
    
    # Create separate evaluation environment (single env for deterministic eval)
    logger.info(f"[2/5] Creating evaluation environment...")
    eval_env_fn = [make_env(config_path, 9999, seed + 9999)]  # Different seed
    eval_env = SubprocVecEnv(eval_env_fn)
    eval_env = VecMonitor(eval_env, os.path.join(run_log_dir, "eval"))
    logger.info("Evaluation environment created")
    
    # IMPROVED HYPERPARAMETERS
    lr_schedule = linear_schedule(3e-4, 1e-5)
    
    ppo_params = {
        "learning_rate": lr_schedule,  # Linear schedule instead of constant
        "n_steps": 2048,
        "batch_size": 4096,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.02,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    
    effective_batch = ppo_params["n_steps"] * n_envs
    
    logger.info(f"\n[3/5] Initializing PPO agent...")
    logger.info(f"  Learning rate: linear_schedule(5e-4 -> 1e-5)")
    logger.info(f"  Effective batch per update: {effective_batch:,}")
    logger.info(f"  n_steps: {ppo_params['n_steps']}")
    logger.info(f"  batch_size: {ppo_params['batch_size']}")
    logger.info(f"  n_epochs: {ppo_params['n_epochs']}")
    
    if resume_from:
        logger.info(f"Loading model from {resume_from}...")
        model = PPO.load(resume_from, env=env, device="cuda", tensorboard_log=run_log_dir)
        model._total_timesteps = total_timesteps
        # Update hyperparameters with new values
        model.ent_coef = 0.02  # Увеличено для борьбы с низкой entropy
        logger.info(f"Updated ent_coef to {model.ent_coef} (was lower before)")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=0,
            tensorboard_log=run_log_dir,
            device="cuda",
            seed=seed,
            **ppo_params
        )
    logger.info("PPO agent initialized")
    
    logger.info(f"\n[4/5] Setting up callbacks...")
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=run_model_dir,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    logger.info(f"  ✓ Checkpoint every {checkpoint_freq:,} steps")
    
    # Evaluation callback - saves best model automatically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_model_dir,
        log_path=os.path.join(run_log_dir, "eval"),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    logger.info(f"  ✓ Evaluation every {eval_freq:,} steps ({n_eval_episodes} episodes)")
    logger.info(f"  ✓ Best model will be saved to {run_model_dir}/best_model.zip")
    
    # ENHANCED custom metrics callback
    metrics_callback = EnhancedMetricsCallback(log_freq=2048)
    logger.info("  ✓ ENHANCED TensorBoard metrics: reward components, success rate, percentiles")
    
    # Detailed logging callback
    logging_callback = DetailedLoggingCallback(
        log_freq=50000,
        custom_logger=logger
    )
    logger.info("  ✓ Detailed logging every 50k steps")
    
    callbacks = CallbackList([
        checkpoint_callback,
        eval_callback,
        metrics_callback,
        logging_callback
    ])
    
    logger.info(f"\n[5/5] Starting training for {total_timesteps:,} timesteps...")
    logger.info("-" * 80)
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=resume_from is None
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user!")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    finally:
        final_path = os.path.join(run_model_dir, "ppo_final")
        model.save(final_path)
        logger.info(f"Final model saved to {final_path}")
        
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {elapsed_str}")
        logger.info(f"Total steps: {model.num_timesteps:,}")
        logger.info(f"Average FPS: {model.num_timesteps / elapsed:.0f}")
        logger.info(f"Final model: {final_path}")
        logger.info(f"Best model: {run_model_dir}/best_model.zip")
        logger.info("=" * 80)
    
    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved PPO Training")
    parser.add_argument("--config", type=str, 
                       default="~/habitat_rl_sorting/configs/color_sorting.yaml")
    parser.add_argument("--timesteps", type=int, default=50_000_000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--log-dir", type=str, default="~/habitat_rl_sorting/logs")
    parser.add_argument("--model-dir", type=str, default="~/habitat_rl_sorting/models")
    parser.add_argument("--checkpoint-freq", type=int, default=500_000)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train_ppo_improved(
        config_path=os.path.expanduser(args.config),
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        log_dir=os.path.expanduser(args.log_dir),
        model_save_path=os.path.expanduser(args.model_dir),
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        resume_from=args.resume,
        seed=args.seed
    )
