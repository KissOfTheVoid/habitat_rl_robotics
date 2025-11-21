"""
Custom sensors and measurements for Color Sorting Task
"""
import numpy as np
from typing import Any

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator


@registry.register_sensor
class ColorZoneSensor(Sensor):
    cls_uuid: str = "color_zone_sensor"
    
    def __init__(self, sim: Simulator, config, *args, **kwargs):
        self._sim = sim
        super().__init__(config=config)
        
    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid
    
    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR
    
    def _get_observation_space(self, *args, **kwargs):
        from gym import spaces
        return spaces.Box(low=-np.inf, high=np.inf, shape=(3, 4), dtype=np.float32)
    
    def get_observation(self, observations, episode, *args, **kwargs):
        zone_data = [
            [-0.5, 0.8, 0.0, 0],
            [0.0, 0.8, 0.0, 1],
            [0.5, 0.8, 0.0, 2]
        ]
        return np.array(zone_data, dtype=np.float32)


@registry.register_sensor
class ObjectColorSensor(Sensor):
    cls_uuid: str = "object_color_sensor"
    
    def __init__(self, sim: Simulator, config, *args, **kwargs):
        self._sim = sim
        super().__init__(config=config)
    
    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid
    
    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR
    
    def _get_observation_space(self, *args, **kwargs):
        from gym import spaces
        return spaces.Box(low=-np.inf, high=np.inf, shape=(3, 4), dtype=np.float32)
    
    def get_observation(self, observations, episode, *args, **kwargs):
        object_data = []
        rigid_obj_mgr = self._sim.get_rigid_object_manager()
        
        for obj_handle in rigid_obj_mgr.get_object_handles():
            if "object_" not in obj_handle:
                continue
            obj = rigid_obj_mgr.get_object_by_handle(obj_handle)
            pos = obj.translation
            try:
                obj_idx = int(obj_handle.split("_")[-1])
                color_id = obj_idx % 3
            except:
                color_id = 0
            object_data.append([pos[0], pos[1], pos[2], color_id])
        
        while len(object_data) < 3:
            object_data.append([0, 0, 0, -1])
        
        return np.array(object_data[:3], dtype=np.float32)


@registry.register_measure
class ColorSortingSuccess(Measure):
    cls_uuid: str = "color_sorting_success"
    
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__()
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ColorSortingSuccess.cls_uuid
    
    def reset_metric(self, episode, task, *args, **kwargs):
        self._metric = 0
    
    def update_metric(self, episode, task, *args, **kwargs):
        if hasattr(task, '_objects_correctly_placed'):
            placements = task._objects_correctly_placed
            self._metric = 1 if (placements and all(placements.values())) else 0
        else:
            self._metric = 0


@registry.register_measure
class ColorSortingReward(Measure):
    cls_uuid: str = "color_sorting_reward"
    
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._prev_objects_placed = set()
        super().__init__()
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ColorSortingReward.cls_uuid
    
    def reset_metric(self, episode, task, *args, **kwargs):
        self._prev_objects_placed = set()
        self._metric = 0
    
    def update_metric(self, episode, task, *args, **kwargs):
        reward = -0.01
        if not hasattr(task, '_objects_correctly_placed'):
            self._metric = reward
            return
        placements = task._objects_correctly_placed
        for obj_handle, is_correct in placements.items():
            if is_correct:
                if obj_handle not in self._prev_objects_placed:
                    reward += 100.0
                    self._prev_objects_placed.add(obj_handle)
                reward += 10.0
        self._metric = reward


@registry.register_measure
class NumObjectsCorrectlyPlaced(Measure):
    cls_uuid: str = "num_objects_correctly_placed"
    
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__()
    
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NumObjectsCorrectlyPlaced.cls_uuid
    
    def reset_metric(self, episode, task, *args, **kwargs):
        self._metric = 0
    
    def update_metric(self, episode, task, *args, **kwargs):
        if hasattr(task, '_objects_correctly_placed'):
            placements = task._objects_correctly_placed
            self._metric = sum(placements.values()) if placements else 0
        else:
            self._metric = 0
