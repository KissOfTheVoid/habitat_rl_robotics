"""
Gymnasium Wrapper for Habitat Color Sorting Environment
Provides clean interface for RL training
"""
import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Tuple
import os
import sys

# Add habitat-lab to path
from omegaconf import OmegaConf
sys.path.insert(0, os.path.expanduser("~/habitat_data/habitat-lab/habitat-lab"))

from habitat import get_config
from habitat.core.env import Env as HabitatEnv
# Import custom classes to trigger registration
import color_sorting_task
import color_sorting_sensors



class ColorSortingGymEnv(gym.Env):
    """
    Gymnasium wrapper for Habitat Color Sorting Task.
    
    Simplified action space (no base movement):
    - arm_action: 7D joint control
    - grip_action: 1D gripper control
    
    Total: 8D continuous action space
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config_path: str,
        render_mode: Optional[str] = "rgb_array",
        max_episode_steps: int = 500,
        override_options: Optional[list] = None,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        
        # Load Habitat config
        self.habitat_config = get_config(config_path, overrides=override_options or [])
        
        # Create Habitat environment
        self._env = HabitatEnv(config=self.habitat_config)
        
        # Define observation space
        self.observation_space = self._create_observation_space()
        
        # Define action space - 8D (no base)
        self.action_space = self._create_action_space()
        
        # Episode tracking
        self._episode_step = 0
        self._episode_reward = 0.0
        
    def _create_observation_space(self) -> gym.Space:
        """Create Gymnasium observation space from Habitat observations."""
        obs = self._env.reset()
        
        spaces_dict = {}
        
        # RGB camera
        if "head_rgb" in obs:
            rgb_shape = obs["head_rgb"].shape
            spaces_dict["rgb"] = gym.spaces.Box(
                low=0, high=255, shape=rgb_shape, dtype=np.uint8
            )
        
        # State vector
        state_components = []
        
        if "joint" in obs:
            state_components.append(obs["joint"].flatten())
        
        if "is_holding" in obs:
            state_components.append(obs["is_holding"].flatten())
        
        if "object_color_sensor" in obs:
            state_components.append(obs["object_color_sensor"].flatten())
        
        if "color_zone_sensor" in obs:
            state_components.append(obs["color_zone_sensor"].flatten())
        
        if state_components:
            state_vector = np.concatenate(state_components)
            spaces_dict["state"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=state_vector.shape, dtype=np.float32
            )
        
        return gym.spaces.Dict(spaces_dict)
    
    def _create_action_space(self) -> gym.Space:
        """
        Create 8D action space (no base movement).
        
        [0-6]: arm_action (7 DOF)
        [7]: grip_action (1D)
        """
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )
    
    def _flatten_action(self, action: np.ndarray) -> Dict:
        """Convert 8D action to Habitat format (no base)."""
        return {
            "action": ("arm_action",),  # Only arm action
            "action_args": {
                "arm_action": action[:7].astype(np.float32),
                "grip_action": action[7:8].astype(np.float32),
                # No base_vel - robot stays stationary
            }
        }
    
    def _process_observation(self, habitat_obs: Dict) -> Dict[str, np.ndarray]:
        """Convert Habitat observation to Gymnasium observation."""
        obs = {}
        
        if "head_rgb" in habitat_obs:
            obs["rgb"] = habitat_obs["head_rgb"].astype(np.uint8)
        
        state_components = []
        
        if "joint" in habitat_obs:
            state_components.append(habitat_obs["joint"].flatten())
        
        if "is_holding" in habitat_obs:
            state_components.append(habitat_obs["is_holding"].flatten())
        
        if "object_color_sensor" in habitat_obs:
            state_components.append(habitat_obs["object_color_sensor"].flatten())
        
        if "color_zone_sensor" in habitat_obs:
            state_components.append(habitat_obs["color_zone_sensor"].flatten())
        
        if state_components:
            obs["state"] = np.concatenate(state_components).astype(np.float32)
        
        return obs
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment for new episode."""
        if seed is not None:
            self._env.seed(seed)
            np.random.seed(seed)
        
        habitat_obs = self._env.reset()
        observation = self._process_observation(habitat_obs)
        
        self._episode_step = 0
        self._episode_reward = 0.0
        
        info = {
            "episode_step": self._episode_step,
            "episode_reward": self._episode_reward,
        }
        
        return observation, info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        habitat_action = self._flatten_action(action)
        
        observations = self._env.step(habitat_action)
        
        metrics = self._env.task.measurements.get_metrics()
        reward = metrics.get(self.habitat_config.habitat.task.reward_measure, 0.0)
        success = metrics.get(self.habitat_config.habitat.task.success_measure, 0)
        
        self._episode_step += 1
        self._episode_reward += reward
        
        terminated = bool(success) and self.habitat_config.habitat.task.end_on_success
        truncated = self._episode_step >= self.max_episode_steps
        
        observation = self._process_observation(observations)
        
        info = {
            'episode_reward': self._episode_reward,
            'episode_length': self._episode_step,
        }
        info.update(metrics)
        
        if hasattr(self._env, "episode_over"):
            terminated = terminated or self._env.episode_over
        
        return observation, reward, terminated, truncated, info
        
    def render(self) -> Optional[np.ndarray]:
        """Render environment."""
        if self.render_mode == "rgb_array":
            obs = self._env.sim.get_sensor_observations()
            if "head_rgb" in obs:
                return obs["head_rgb"].astype(np.uint8)
        return None
    
    def close(self):
        """Close environment."""
        self._env.close()
    
    def seed(self, seed: int):
        """Set random seed."""
        self._env.seed(seed)
        np.random.seed(seed)
