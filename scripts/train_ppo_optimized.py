#\!/usr/bin/env python3
"""
Optimized PPO Training with Parallel Environments
Designed for long training runs (10M-100M steps) with maximum GPU/CPU utilization
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


class DetailedLoggingCallback(BaseCallback):
    """Callback for detailed logging during training."""
    
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


def train_ppo_optimized(
    config_path: str,
    total_timesteps: int = 20_000_000,
    n_envs: int = 20,
    log_dir: str = "./logs",
    model_save_path: str = "./models",
    checkpoint_freq: int = 500_000,
    resume_from: str = None,
    seed: int = 42
):
    """
    Optimized PPO training for maximum GPU/CPU utilization.
    
    Target: 80% GPU, 80% CPU utilization
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PPO_dense_{n_envs}env_{total_timesteps//1_000_000}M_{timestamp}"
    run_log_dir = os.path.join(log_dir, run_name)
    run_model_dir = os.path.join(model_save_path, run_name)
    
    Path(run_log_dir).mkdir(parents=True, exist_ok=True)
    Path(run_model_dir).mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(run_log_dir)
    
    logger.info("=" * 80)
    logger.info("MAXIMUM UTILIZATION PPO TRAINING")
    logger.info("=" * 80)
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info(f"Parallel environments: {n_envs}")
    logger.info(f"Checkpoint frequency: {checkpoint_freq:,}")
    logger.info(f"Log directory: {run_log_dir}")
    logger.info(f"Model directory: {run_model_dir}")
    
    logger.info(f"\n[1/4] Creating {n_envs} parallel environments...")
    env_fns = [make_env(config_path, i, seed) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env, run_log_dir)
    logger.info(f"Environments created successfully")
    
    # MAXIMUM UTILIZATION HYPERPARAMETERS
    # Optimized for A100 80GB + 32 CPU cores + 160GB RAM
    ppo_params = {
        "learning_rate": 3e-4,
        "n_steps": 2048,               # INCREASED: More steps per env
        "batch_size": 4096,            # INCREASED: Bigger batches for GPU
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    
    effective_batch = ppo_params["n_steps"] * n_envs
    updates_per_iter = effective_batch // ppo_params["batch_size"]
    
    logger.info(f"\n[2/4] Initializing PPO agent...")
    logger.info(f"  Effective batch per update: {effective_batch:,}")
    logger.info(f"  Mini-batches per iteration: {updates_per_iter}")
    for key, value in ppo_params.items():
        logger.info(f"  {key}: {value}")
    
    if resume_from:
        logger.info(f"Loading model from {resume_from}...")
        model = PPO.load(resume_from, env=env, device="cuda", tensorboard_log=run_log_dir)
        model._total_timesteps = total_timesteps
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
    
    logger.info(f"\n[3/4] Setting up callbacks...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=run_model_dir,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    
    logging_callback = DetailedLoggingCallback(
        log_freq=50000,
        custom_logger=logger
    )
    
    callbacks = CallbackList([checkpoint_callback, logging_callback])
    logger.info("Callbacks configured")
    
    logger.info(f"\n[4/4] Starting training for {total_timesteps:,} timesteps...")
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
        logger.warning("Training interrupted by user\!")
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
        logger.info("=" * 80)
    
    env.close()
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Maximum Utilization PPO Training")
    parser.add_argument("--config", type=str, 
                       default="~/habitat_rl_sorting/configs/color_sorting.yaml")
    parser.add_argument("--timesteps", type=int, default=20_000_000)
    parser.add_argument("--n-envs", type=int, default=20,
                       help="Number of parallel environments (default: 20 for 80%% CPU)")
    parser.add_argument("--log-dir", type=str, default="~/habitat_rl_sorting/logs")
    parser.add_argument("--model-dir", type=str, default="~/habitat_rl_sorting/models")
    parser.add_argument("--checkpoint-freq", type=int, default=500_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train_ppo_optimized(
        config_path=os.path.expanduser(args.config),
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        log_dir=os.path.expanduser(args.log_dir),
        model_save_path=os.path.expanduser(args.model_dir),
        checkpoint_freq=args.checkpoint_freq,
        resume_from=args.resume,
        seed=args.seed
    )
