"""
Color Sorting Task for Habitat-Lab
"""
import numpy as np
from typing import Any, Dict
import magnum as mn

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import rearrange_logger


@registry.register_task(name="ColorSortingTask-v0")
class ColorSortingTaskV1(RearrangeTask):
    
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            should_place_articulated_agent=True,
            **kwargs,
        )
        
        self._num_objects = self._config.get("num_objects", 1)
        self._success_distance_threshold = self._config.get("success_distance_threshold", 0.1)
        self._spawn_region = self._config.get("spawn_region", {
            "x_range": [-0.3, 0.3],
            "y_range": [1.3, 1.5],
            "z_range": [-0.2, 0.2]
        })
        
        self._objects_correctly_placed = {}
        self._objects_first_time_placed = set()
        self._spawned_objects = []
        
        self._color_zones = {
            "red": {"position": [-0.5, 0.8, 0.0], "radius": 0.1, "color_id": 0},
            "green": {"position": [0.0, 0.8, 0.0], "radius": 0.1, "color_id": 1},
            "blue": {"position": [0.5, 0.8, 0.0], "radius": 0.1, "color_id": 2}
        }
        
        rearrange_logger.info(f"ColorSortingTask initialized with {self._num_objects} objects")
    
    def reset(self, episode: Episode):
        observations = super().reset(episode)
        
        self._objects_correctly_placed = {}
        self._objects_first_time_placed = set()
        
        self._spawn_objects()
        
        return observations
    
    def _spawn_objects(self):
        """Spawn colored cube objects in front of robot within arm reach."""
        if not hasattr(self, "_sim") or self._sim is None:
            return
        
        rigid_obj_mgr = self._sim.get_rigid_object_manager()
        obj_attr_mgr = self._sim.get_object_template_manager()
        
        # Get robot end-effector position as reference
        try:
            agent = self._sim.articulated_agent
            ee_transform = agent.ee_transform()
            ee_pos = ee_transform.translation
            robot_pos = agent.base_pos
            rearrange_logger.info(f"Robot base: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f})")
            rearrange_logger.info(f"Robot EE: ({ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f})")
        except Exception as e:
            rearrange_logger.warning(f"Could not get robot position: {e}")
            return
        
        # Load red cube template
        red_cube_config = "/home/iskopyl/habitat_data/objects/red_cube.object_config.json"
        try:
            obj_attr_mgr.load_configs(red_cube_config)
        except Exception as e:
            rearrange_logger.warning(f"Could not load red_cube config: {e}")
        
        # Remove old objects
        for obj_handle in list(rigid_obj_mgr.get_object_handles()):
            if "object_" in obj_handle or "cube" in obj_handle.lower():
                rigid_obj_mgr.remove_object_by_handle(obj_handle)
        
        self._spawned_objects = []
        
        # Spawn cube in front of camera, visible and reachable by arm
        # Camera is above EE, so cube should be at EE height, in front of robot
        for i in range(self._num_objects):
            # Spawn in front of robot at a height between EE and camera
            offset_x = np.random.uniform(-0.1, 0.1)    # Slight left/right
            offset_y = np.random.uniform(0.1, 0.3)     # Slightly above EE (closer to camera)
            offset_z = np.random.uniform(-0.6, -0.4)   # Further in front for visibility
            
            abs_x = ee_pos[0] + offset_x
            abs_y = ee_pos[1] + offset_y
            abs_z = ee_pos[2] + offset_z
            
            try:
                obj = rigid_obj_mgr.add_object_by_template_handle(red_cube_config)
                if obj is None:
                    obj = rigid_obj_mgr.add_object_by_template_handle("cubeSolid")
                
                if obj is not None:
                    obj.translation = mn.Vector3(abs_x, abs_y, abs_z)
                    self._spawned_objects.append(obj.handle)
                    rearrange_logger.info(f"Spawned cube at ({abs_x:.2f}, {abs_y:.2f}, {abs_z:.2f})")
            except Exception as e:
                rearrange_logger.warning(f"Error spawning object: {e}")
        
        # Update target zone to be near robot too
        self._update_zone_positions()
    
    def _update_zone_positions(self):
        """Update zone positions relative to robot."""
        try:
            agent = self._sim.articulated_agent
            ee_pos = agent.ee_transform().translation
            # Place zone slightly to the side of EE
            self._color_zones = {
                "red": {"position": [ee_pos[0] + 0.3, ee_pos[1] - 0.1, ee_pos[2] - 0.3], "radius": 0.15, "color_id": 0},
            }
            rearrange_logger.info(f"Target zone: {self._color_zones['red']['position']}")
        except Exception as e:
            rearrange_logger.warning(f"Could not update zones: {e}")
    
    def _get_object_to_zone_distances(self) -> Dict[str, float]:
        distances = {}
        
        if not hasattr(self, "_sim") or self._sim is None:
            return distances
        
        rigid_obj_mgr = self._sim.get_rigid_object_manager()
        
        for idx, obj_handle in enumerate(self._spawned_objects):
            try:
                obj = rigid_obj_mgr.get_object_by_handle(obj_handle)
                if obj is None:
                    continue
                obj_pos = obj.translation
                obj_color_id = idx % len(self._color_zones)
                target_zone = self._get_zone_by_color_id(obj_color_id)
                
                if target_zone is None:
                    continue
                
                zone_pos = np.array(target_zone["position"])
                distance = np.linalg.norm(np.array([obj_pos.x, obj_pos.y, obj_pos.z]) - zone_pos)
                distances[obj_handle] = distance
            except:
                pass
        
        return distances
    
    def _get_zone_by_color_id(self, color_id: int) -> Dict:
        for zone_config in self._color_zones.values():
            if zone_config["color_id"] == color_id:
                return zone_config
        return None
    
    def _check_objects_in_zones(self) -> Dict[str, bool]:
        placements = {}
        distances = self._get_object_to_zone_distances()
        
        for obj_handle, distance in distances.items():
            is_correct = distance < self._success_distance_threshold
            placements[obj_handle] = is_correct
            
            if is_correct and obj_handle not in self._objects_first_time_placed:
                self._objects_first_time_placed.add(obj_handle)
        
        self._objects_correctly_placed = placements
        return placements
    
    def step(self, action, episode):
        obs = super().step(action, episode)
        self._check_objects_in_zones()
        return obs
    
    def get_info(self, observations) -> Dict[str, Any]:
        info = super().get_info(observations)
        placements = self._objects_correctly_placed
        
        info.update({
            "success": int(all(placements.values())) if placements else 0,
            "num_correctly_placed": sum(placements.values()) if placements else 0,
            "num_total_objects": self._num_objects,
        })
        
        return info
