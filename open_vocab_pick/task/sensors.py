from typing import TYPE_CHECKING, Any

from habitat.core.registry import registry
from habitat.core.simulator import (
    SemanticSensor,
    Sensor,
    SensorTypes,
    Simulator,
)
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from dataclasses import dataclass

from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig, HabitatConfig, LabSensorConfig,
    MeasurementConfig, SimulatorConfig)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesRLConfig, PolicyConfig, RLConfig)
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin


cv2 = try_cv2_import()


if TYPE_CHECKING:
    from omegaconf import DictConfig

from dataclasses import dataclass

from habitat.config.default_structured_configs import (
    HabitatSimSemanticSensorConfig,
)
from habitat.core.registry import registry
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

@registry.register_sensor
class BinaryMaskSensor(Sensor):
    r"""Sensor for observations which are used in Rearrange.
    SemanticSensor needs to be one of the Simulator sensors.
    This sensor return the semantic sensor image taken from the robot arm with a binary mask applied
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the Rearrange sensor.
    
    Returns:
        observations: the semantic sensor image with a binary mask applied
    """
    cls_uuid: str = "articulated_agent_arm_binary_mask"

    def __init__(
        self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        semantic_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, SemanticSensor)
        ]
        if len(semantic_sensor_uuids) != 1:
            raise ValueError(
                "Cannot create BinaryMaskSensor, there are no semantic sensors,"
                f" {len(semantic_sensor_uuids)} detected"
            )
        (self._semantic_sensor_uuid,) = semantic_sensor_uuids
        self._target_object_handles = None
        super().__init__(config=config)

        # Prints out the answer to life on init
        print("The answer to life is", 42)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC # check what this should be
    
    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._semantic_sensor_uuid
        ] # basically says the dimenions of the observations are the same as the panoptic sensor
    
    def _modify_semantic_ids(self):
        # Add the rigid object id for the semantic map
        rom = self._sim.get_rigid_object_manager()
        # object_handles = []
        for i, handle in enumerate(rom.get_object_handles()):
            obj = rom.get_object_by_handle(handle)
            if handle in self._target_object_handles:
                for node in obj.visual_scene_nodes:
                    node.semantic_id = (
                        obj.object_id + self._sim.habitat_config.object_ids_start
                    )
            else:
                for node in obj.visual_scene_nodes:
                    node.semantic_id = 0
        # print('object_handles', object_handles)


    # This is called whenever reset is called or an action is taken
    def get_observation(self, *args: Any, observations, episode: RearrangeEpisode, **kwargs: Any):
        # Approach 1: change semantic id of everything but the target object to 0, change semantic id of target object to 1, return the semantic observation
        self._target_object_handles = episode.targets.keys() # alternatively, you can also do idxs, _ = self._sim.get_targets()
        # print('target_handles', list(self._target_object_handles))
        # scene = self._sim.semantic_scene
        # self.print_scene_recur(scene)
        # print(instance_id_to_label_id)
        self._modify_semantic_ids()

        # print(self._sim.scene_obj_ids)
        sim_obs = self._sim.get_sensor_observations()
        updated_semantic_observations = self._sim._sensor_suite.get_observations(sim_obs)[self._semantic_sensor_uuid]
        # semantic_observations = observations[self._semantic_sensor_uuid]
        return updated_semantic_observations
    



cs = ConfigStore.instance()

@dataclass
class ArmBinaryMaskSensorConfig(LabSensorConfig):
    type: str = "BinaryMaskSensor"
    # width: int = 256
    # height: int = 256

cs.store(
    package="habitat.task.lab_sensors.arm_binary_mask_sensor",
    group="habitat/task/lab_sensors",
    name="arm_binary_mask_sensor",
    node=ArmBinaryMaskSensorConfig,
)