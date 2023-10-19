from dataclasses import dataclass

from habitat.config.default_structured_configs import (
    LabSensorConfig,
)
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin

cs = ConfigStore.instance()


@dataclass
class ArmBinaryMaskSensorConfig(LabSensorConfig):
    type: str = "BinaryMaskSensor"


# -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------

cs.store(
    package="habitat.task.lab_sensors.arm_binary_mask_sensor",
    group="habitat/task/lab_sensors",
    name="arm_binary_mask_sensor",
    node=ArmBinaryMaskSensorConfig,
)


class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat",
            path="pkg://config/tasks/",
        )
