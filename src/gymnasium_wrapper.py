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
    
    Provides standard Gymnasium interface:
    - reset() -> observation, info
    - step(action) -> observation, reward, terminated, truncated, info
    - render() -> rgb_array
    - close()
    
    Observation Space:
        Dict with keys:
        - 'rgb': Box(0, 255, (H, W, 3), uint8) - Raw RGB camera feed
        - 'state': Box(-inf, inf, (N,), float32) - Robot state (joints, positions, etc.)
    
    Action Space:
        Box(-1, 1, (M,), float32) - Continuous control of robot joints + gripper
    
    Reward:
        Float value based on task performance (see ColorSortingReward measure)
    
    Episode Termination:
        - terminated: All objects successfully sorted
        - truncated: Max episode steps reached
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config_path: str,
        render_mode: Optional[str] = "rgb_array",
        max_episode_steps: int = 500,
        override_options: Optional[list] = None,
    ):
        """
        Initialize Gymnasium wrapper.
        
        Args:
            config_path: Path to Habitat config YAML file
            render_mode: Rendering mode ('rgb_array' for headless)
            max_episode_steps: Maximum steps per episode
            override_options: List of config overrides
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        
        
        # Load Habitat config
        self.habitat_config = get_config(config_path, overrides=override_options or [])
        
        
        # Create Habitat environment
        self._env = HabitatEnv(config=self.habitat_config)
        
        # Define observation space
        self.observation_space = self._create_observation_space()
        
        # Define action space
        self.action_space = self._create_action_space()
        
        # Episode tracking
        self._episode_step = 0
        self._episode_reward = 0.0
        
    def _create_observation_space(self) -> gym.Space:
        """
        Create Gymnasium observation space from Habitat observations.
        """
        # Get sample observation to determine shapes
        obs = self._env.reset()
        
        spaces_dict = {}
        
        # RGB camera (raw, no preprocessing)
        if "head_rgb" in obs:
            rgb_shape = obs["head_rgb"].shape
            spaces_dict["rgb"] = gym.spaces.Box(
                low=0, high=255, shape=rgb_shape, dtype=np.uint8
            )
        
        # State vector (joints, object positions, zone positions, etc.)
        state_components = []
        
        # Joint positions
        if "joint" in obs:
            state_components.append(obs["joint"].flatten())
        
        # Gripper state
        if "is_holding" in obs:
            state_components.append(obs["is_holding"].flatten())
        
        # Object positions and color IDs
        if "object_color_sensor" in obs:
            state_components.append(obs["object_color_sensor"].flatten())
        
        # Zone positions and color IDs
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
        Create flattened Box action space for SB3 compatibility.
        
        Original Habitat action: Dict of Dicts
        Flattened: Box(10,) = [arm(7) + grip(1) + base(2)]
        """
        # Flatten: arm_action(7) + grip_action(1) + base_vel(2) = 10D
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32
        )
    
    def _flatten_action(self, action: np.ndarray) -> Dict:
        """Convert flattened action to Habitat format with proper structure."""
        # action is (10,) array: [arm(7), grip(1), base(2)]
        return {
            "action": ("arm_action", "base_velocity"),  # Multiple actions
            "action_args": {
                "arm_action": action[:7].astype(np.float32),
                "grip_action": action[7:8].astype(np.float32),
                "base_vel": action[8:10].astype(np.float32) * 20.0  # Scale to [-20, 20]
            }
        }
    def _process_observation(self, habitat_obs: Dict) -> Dict[str, np.ndarray]:
        """
        Convert Habitat observation to Gymnasium observation.
        """
        obs = {}
        
        # RGB camera
        if "head_rgb" in habitat_obs:
            obs["rgb"] = habitat_obs["head_rgb"].astype(np.uint8)
        
        # State vector
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
    
    def _process_action(self, action: np.ndarray) -> Dict:
        """
        Convert Gymnasium action to Habitat action format.
        """
        # If action space is Dict, need to unpack
        habitat_action_space = self._env.action_space
        
        if isinstance(habitat_action_space, gym.spaces.Dict):
            # Unpack flattened action
            action_dict = {}
            offset = 0
            
            for key in sorted(habitat_action_space.spaces.keys()):
                space = habitat_action_space.spaces[key]
                if isinstance(space, gym.spaces.Box):
                    dims = int(np.prod(space.shape))
                    action_dict[key] = action[offset:offset+dims].reshape(space.shape)
                    offset += dims
            
            return action_dict
        else:
            return action
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment for new episode.
        
        Returns:
            observation: Dict with 'rgb' and 'state' keys
            info: Dict with episode info
        """
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
        """Execute one step using Habitat's step method."""
        # Convert to Habitat action format  
        habitat_action = self._flatten_action(action)
        
        # Use Habitat's standard step method
        observations = self._env.step(habitat_action)
        
        # Get measurements and metrics
        metrics = self._env.task.measurements.get_metrics()
        reward = metrics.get(self.habitat_config.habitat.task.reward_measure, 0.0)
        success = metrics.get(self.habitat_config.habitat.task.success_measure, 0)
        
        self._episode_step += 1
        self._episode_reward += reward
        
        # Termination
        terminated = bool(success) and self.habitat_config.habitat.task.end_on_success
        truncated = self._episode_step >= self.max_episode_steps
        
        # Process observation
        observation = self._process_observation(observations)
        
        # Info dict
        info = {
            'episode_reward': self._episode_reward,
            'episode_length': self._episode_step,
        }
        info.update(metrics)
        
        # Check if Habitat ended the episode
        if hasattr(self._env, "episode_over"):
            terminated = terminated or self._env.episode_over
        
        return observation, reward, terminated, truncated, info

        
    def render(self) -> Optional[np.ndarray]:
        """
        Render environment.
        
        Returns:
            RGB array if render_mode is 'rgb_array', else None
        """
        if self.render_mode == "rgb_array":
            # Return current RGB observation
            obs = self._env.sim.get_sensor_observations()
            if "head_rgb" in obs:
                return obs["head_rgb"].astype(np.uint8)
        return None
    
    def close(self):
        """Close environment and clean up resources."""
        self._env.close()
    
    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._env.seed(seed)
        np.random.seed(seed)
