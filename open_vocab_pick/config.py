# from dataclasses import dataclass

# from habitat.config.default_structured_configs import (
#     CollisionsMeasurementConfig, HabitatConfig, LabSensorConfig,
#     MeasurementConfig, SimulatorConfig)
# from habitat_baselines.config.default_structured_configs import (
#     HabitatBaselinesRLConfig, PolicyConfig, RLConfig)
# from hydra.core.config_search_path import ConfigSearchPath
# from hydra.core.config_store import ConfigStore
# from hydra.plugins.search_path_plugin import SearchPathPlugin

# # cs = ConfigStore.instance()


# # @dataclass
# # class ImageGoalRotationSensorConfig(LabSensorConfig):
# #     type: str = "ImageGoalRotationSensor"
# #     sample_angle: bool = True


# # @dataclass
# # class ArmPanopticSensorConfig(HabitatSimSemanticSensorConfig):
# #     uuid: str = "articulated_agent_arm_panoptic"
# #     width: int = 256
# #     height: int = 256

# # # -----------------------------------------------------------------------------
# # # Register configs in the Hydra ConfigStore
# # # -----------------------------------------------------------------------------

# # cs.store(
# #     # package=f"habitat.task.lab_sensors.image_goal_rotation_sensor",
# #     group="habitat/task/lab_sensors",
# #     name="image_goal_rotation_sensor",
# #     node=ImageGoalRotationSensorConfig,
# # )


# # cs.store(
# #     group="habitat/simulator/sim_sensors",
# #     name="arm_panoptic_sensor",
# #     node=ArmPanopticSensorConfig,
# # )
# # # class HabitatConfigPlugin(SearchPathPlugin):
# # #     def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
# # #         search_path.append(
# # #             provider="habitat",
# # #             path="pkg://config/tasks/",
# # #         )
