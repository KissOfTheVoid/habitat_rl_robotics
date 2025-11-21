"""
Color Sorting Task for Habitat-Lab
Extends RearrangeTask for multi-object color-based sorting
"""
import numpy as np
from typing import Any, Dict, List

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import rearrange_logger


@registry.register_task(name="ColorSortingTask-v0")
class ColorSortingTaskV1(RearrangeTask):
    """
    Color Sorting Task: Robot must sort multiple objects into color-matched zones.
    
    Task Description:
    - Multiple objects with different colors spawn on a table
    - Each color has a designated target zone
    - Robot must pick and place objects in matching zones
    - Episode succeeds when all objects are correctly sorted
    
    Observation Space:
    - RGB camera feed (raw, no preprocessing)
    - Joint positions
    - Gripper state
    - Object positions and color IDs
    - Zone positions and color IDs
    
    Action Space:
    - Continuous control of robot joints + gripper
    
    Reward Shaping:
    - +100 for first successful placement in correct zone
    - +10 per step object stays in correct zone
    - -0.5 * distance for objects not in zones
    - -0.01 per step (time penalty)
    - -10 for placing in wrong zone
    """
    
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            should_place_articulated_agent=True,
            **kwargs,
        )
        
        # Configuration from task config
        self._num_objects = self._config.get("num_objects", 3)
        self._success_distance_threshold = self._config.get("success_distance_threshold", 0.1)
        self._spawn_region = self._config.get("spawn_region", {
            "x_range": [-0.3, 0.3],
            "y_range": [0.8, 1.0],
            "z_range": [-0.2, 0.2]
        })
        
        # Track which objects have been successfully placed
        self._objects_correctly_placed = {}
        self._objects_first_time_placed = set()
        
        # Color zones configuration
        self._color_zones = self._config.get("color_zones", {
            "red": {"position": [-0.5, 0.8, 0.0], "radius": 0.1, "color_id": 0},
            "green": {"position": [0.0, 0.8, 0.0], "radius": 0.1, "color_id": 1},
            "blue": {"position": [0.5, 0.8, 0.0], "radius": 0.1, "color_id": 2}
        })
        
        rearrange_logger.info(f"ColorSortingTask initialized with {self._num_objects} objects")
    
    def reset(self, episode: Episode):
        """
        Reset the task for a new episode.
        Spawns objects randomly and resets tracking variables.
        """
        observations = super().reset(episode)
        
        # Reset tracking
        self._objects_correctly_placed = {}
        self._objects_first_time_placed = set()
        
        # Spawn objects in random positions within spawn region
        self._spawn_objects()
        
        return observations
    
    def _spawn_objects(self):
        """
        Spawn objects randomly in the designated spawn region.
        Each object gets a random color assignment.
        """
        if not hasattr(self, "_sim") or self._sim is None:
            rearrange_logger.warning("Simulator not initialized, skipping object spawn")
            return
        
        # Get object manager from simulator
        rigid_obj_mgr = self._sim.get_rigid_object_manager()
        
        # Clear existing objects
        for obj_id in list(rigid_obj_mgr.get_object_handles()):
            if "object_" in obj_id:
                rigid_obj_mgr.remove_object_by_handle(obj_id)
        
        spawn_region = self._spawn_region
        np.random.seed(self._config.get("seed", None))
        
        for i in range(self._num_objects):
            # Random position in spawn region
            x = np.random.uniform(spawn_region["x_range"][0], spawn_region["x_range"][1])
            y = np.random.uniform(spawn_region["y_range"][0], spawn_region["y_range"][1])
            z = np.random.uniform(spawn_region["z_range"][0], spawn_region["z_range"][1])
            
            # Assign color (cycle through available colors)
            color_idx = i % len(self._color_zones)
            
            rearrange_logger.debug(f"Spawning object_{i} at ({x:.2f}, {y:.2f}, {z:.2f}) with color_id={color_idx}")
    
    def _get_object_to_zone_distances(self) -> Dict[str, float]:
        """
        Calculate distances from each object to its target zone.
        Returns dict mapping object_id to distance.
        """
        distances = {}
        
        if not hasattr(self, "_sim") or self._sim is None:
            return distances
        
        rigid_obj_mgr = self._sim.get_rigid_object_manager()
        
        for obj_handle in rigid_obj_mgr.get_object_handles():
            if "object_" not in obj_handle:
                continue
            
            obj = rigid_obj_mgr.get_object_by_handle(obj_handle)
            obj_pos = obj.translation
            
            # Get object color ID (should be stored in object metadata)
            obj_color_id = self._get_object_color_id(obj_handle)
            
            # Find matching zone
            target_zone = self._get_zone_by_color_id(obj_color_id)
            if target_zone is None:
                continue
            
            zone_pos = np.array(target_zone["position"])
            distance = np.linalg.norm(obj_pos - zone_pos)
            distances[obj_handle] = distance
        
        return distances
    
    def _get_object_color_id(self, obj_handle: str) -> int:
        """Extract color ID from object handle or metadata."""
        # Parse from handle: object_0 -> color_id 0, object_1 -> color_id 1, etc.
        try:
            obj_idx = int(obj_handle.split("_")[-1])
            return obj_idx % len(self._color_zones)
        except:
            return 0
    
    def _get_zone_by_color_id(self, color_id: int) -> Dict:
        """Get zone configuration by color ID."""
        for zone_name, zone_config in self._color_zones.items():
            if zone_config["color_id"] == color_id:
                return zone_config
        return None
    
    def _check_objects_in_zones(self) -> Dict[str, bool]:
        """
        Check which objects are correctly placed in their zones.
        Returns dict mapping object_id to boolean (in correct zone or not).
        """
        placements = {}
        distances = self._get_object_to_zone_distances()
        
        for obj_handle, distance in distances.items():
            is_correct = distance < self._success_distance_threshold
            placements[obj_handle] = is_correct
            
            # Track first-time placements
            if is_correct and obj_handle not in self._objects_first_time_placed:
                self._objects_first_time_placed.add(obj_handle)
                rearrange_logger.info(f"{obj_handle} placed correctly for the first time!")
        
        self._objects_correctly_placed = placements
        return placements
    
    def step(self, action, episode):
        """
        Execute one step of the task.
        Action already in correct format from wrapper: {"action": ..., "action_args": ...}
        """
        # Call parent step (RearrangeTask handles action processing)
        obs = super().step(action, episode)
        
        # Update our custom task state
        self._check_objects_in_zones()
        
        return obs
    def get_info(self, observations) -> Dict[str, Any]:
        """
        Return info dict with metrics for RL.
        """
        info = super().get_info(observations)
        
        distances = self._get_object_to_zone_distances()
        placements = self._objects_correctly_placed
        
        info.update({
            "success": int(all(placements.values())) if placements else 0,
            "zone_matches": list(placements.values()) if placements else [],
            "distances": list(distances.values()) if distances else [],
            "num_correctly_placed": sum(placements.values()) if placements else 0,
            "num_total_objects": self._num_objects,
        })
        
        return info
