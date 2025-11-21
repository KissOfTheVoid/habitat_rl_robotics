"""
Structured Config for Color Sorting Task
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any
from omegaconf import MISSING

from habitat.config.default_structured_configs import (
    HabitatBaseConfig,
    MeasurementConfig,
    LabSensorConfig,
    TaskConfig
)


@dataclass
class ColorSortingRewardMeasurementConfig(MeasurementConfig):
    """Config for Color Sorting Reward measurement."""
    type: str = "ColorSortingReward"
    step_penalty: float = -0.01
    first_placement_reward: float = 100.0
    stay_in_zone_reward: float = 10.0
    distance_penalty_weight: float = -0.5
    collision_penalty: float = -10.0


@dataclass
class ColorSortingSuccessMeasurementConfig(MeasurementConfig):
    """Config for Color Sorting Success measurement."""
    type: str = "ColorSortingSuccess"


@dataclass
class NumObjectsCorrectlyPlacedMeasurementConfig(MeasurementConfig):
    """Config for NumObjectsCorrectlyPlaced measurement."""
    type: str = "NumObjectsCorrectlyPlaced"


@dataclass
class ColorZoneSensorConfig(LabSensorConfig):
    """Config for Color Zone Sensor."""
    type: str = "ColorZoneSensor"


@dataclass
class ObjectColorSensorConfig(LabSensorConfig):
    """Config for Object Color Sensor."""
    type: str = "ObjectColorSensor"


@dataclass
class ColorSortingTaskConfig(TaskConfig):
    """Config for Color Sorting Task."""
    type: str = "ColorSortingTask-v0"
    
    # Task-specific parameters
    num_objects: int = 3
    success_distance_threshold: float = 0.1
    
    spawn_region: Dict[str, List[float]] = field(default_factory=lambda: {
        "x_range": [-0.3, 0.3],
        "y_range": [0.8, 1.0],
        "z_range": [-0.2, 0.2]
    })
    
    color_zones: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "red": {"position": [-0.5, 0.8, 0.0], "radius": 0.1, "color_id": 0},
        "green": {"position": [0.0, 0.8, 0.0], "radius": 0.1, "color_id": 1},
        "blue": {"position": [0.5, 0.8, 0.0], "radius": 0.1, "color_id": 2}
    })
