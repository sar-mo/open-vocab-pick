from dataclasses import dataclass

from habitat.config.default_structured_configs import LabSensorConfig, MeasurementConfig
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin

cs = ConfigStore.instance()


# sensors
@dataclass
class ArmBinaryMaskSensorConfig(LabSensorConfig):
    type: str = "BinaryMaskSensor"


# measures
@dataclass
class RearrangePickRewardV2MeasurementConfig(MeasurementConfig):
    type: str = "RearrangePickRewardV2"
    dist_reward: float = 2.0
    pick_reward: float = 2.0
    constraint_violate_pen: float = 1.0
    drop_pen: float = 0.5
    wrong_pick_pen: float = 0.5
    force_pen: float = 0.0001
    max_force_pen: float = 0.01
    force_end_pen: float = 1.0
    use_diff: bool = True
    drop_obj_should_end: bool = True
    wrong_pick_should_end: bool = True
    enable_vel_penality: float = -1.0
    using_gaze_grasp: bool = True
    gaze_ee_goal_dist: float = 0.4


@dataclass
class RearrangePickSuccessV2MeasurementConfig(MeasurementConfig):
    type: str = "RearrangePickSuccessV2"
    ee_resting_success_threshold: float = 0.15


# -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------

# sensors
cs.store(
    package="habitat.task.lab_sensors.arm_binary_mask_sensor",
    group="habitat/task/lab_sensors",
    name="arm_binary_mask_sensor",
    node=ArmBinaryMaskSensorConfig,
)

# measures
cs.store(
    package="habitat.task.measurements.pick_reward_v2",
    group="habitat/task/measurements",
    name="pick_reward_v2",
    node=RearrangePickRewardV2MeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.pick_success_v2",
    group="habitat/task/measurements",
    name="pick_success_v2",
    node=RearrangePickSuccessV2MeasurementConfig,
)


class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat",
            path="pkg://config/tasks/",
        )
