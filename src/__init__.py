"""
Habitat RL Color Sorting Environment
Модульная система для обучения роборуки сортировке объектов по цвету
"""
import sys
import os

# Add habitat-lab to path
sys.path.insert(0, os.path.expanduser("~/habitat_data/habitat-lab/habitat-lab"))

# Register structured configs
from hydra.core.config_store import ConfigStore
from habitat.config.default_structured_configs import register_hydra_plugin

from .color_sorting_config import (
    ColorSortingRewardMeasurementConfig,
    ColorSortingSuccessMeasurementConfig,
    NumObjectsCorrectlyPlacedMeasurementConfig,
    ColorZoneSensorConfig,
    ObjectColorSensorConfig,
    ColorSortingTaskConfig
)

# Import to register with registry
from .color_sorting_task import ColorSortingTaskV1
from .color_sorting_sensors import (
    ColorZoneSensor,
    ObjectColorSensor,
    ColorSortingSuccess,
    ColorSortingReward,
    NumObjectsCorrectlyPlaced
)

# Register configs with Hydra
cs = ConfigStore.instance()

cs.store(
    group="habitat/task/measurements",
    name="color_sorting_reward",
    node=ColorSortingRewardMeasurementConfig
)

cs.store(
    group="habitat/task/measurements",
    name="color_sorting_success",
    node=ColorSortingSuccessMeasurementConfig
)

cs.store(
    group="habitat/task/measurements",
    name="num_objects_correctly_placed",
    node=NumObjectsCorrectlyPlacedMeasurementConfig
)

cs.store(
    group="habitat/task/lab_sensors",
    name="color_zone_sensor",
    node=ColorZoneSensorConfig
)

cs.store(
    group="habitat/task/lab_sensors",
    name="object_color_sensor",
    node=ObjectColorSensorConfig
)

__version__ = "1.0.0"
__all__ = [
    "ColorSortingTaskV1",
    "ColorZoneSensor",
    "ObjectColorSensor",
    "ColorSortingSuccess",
    "ColorSortingReward",
    "NumObjectsCorrectlyPlaced"
]
