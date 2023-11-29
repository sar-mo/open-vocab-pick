from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from gym.spaces.box import Box
from habitat.core.registry import registry
from habitat.core.simulator import (
    Observations,
    SemanticSensor,
    Sensor,
    SensorTypes,
    Simulator,
    VisualObservation,
)
from gym import spaces
from habitat.core.utils import try_cv2_import
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
# from open_vocab_pick


import pickle


if TYPE_CHECKING:
    from omegaconf import DictConfig

cv2 = try_cv2_import()

# pickling is at least 10x faster than csv
def load_pickle(path):
    file = open(path, "rb")
    data = pickle.load(file)
    return data

def save_pickle(data, path):
    file = open(path, "wb")
    data = pickle.dump(data, file)

@registry.register_sensor
class ClipObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id, and we will generate the prompt corresponding to it
    so that it's usable by CLIP's text encoder.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "clip_objectgoal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.mapping = load_pickle(config.mapping)
        self.cache = load_pickle(config.cache)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        **kwargs: Any,
    ) -> Optional[int]:
        # only works with one target for now
        object_id = list(episode.targets.keys())[0]
        # remove everything after the last underscore, this is not good behavior, wont' generalize
        object_id = object_id[:object_id.rfind("_")]
        object_category = self.mapping[object_id]
        object_clip_embedding = self.cache[object_category]
        return object_clip_embedding


@registry.register_sensor
class BinaryMaskSensor(Sensor):
    r"""Sensor for observations which are used in Rearrange.
    SemanticSensor needs to be one of the Simulator sensors.
    This sensor return the semantic sensor image taken from the robot arm
    with a binary mask applied
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the Rearrange sensor.

    Returns:
        observations: the semantic sensor image with a binary mask applied
    """
    cls_uuid: str = "articulated_agent_arm_binary_mask"

    def __init__(self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any):
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
        self._target_object_handles: List[str] = []
        super().__init__(config=config)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.SEMANTIC  # check what this should be

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Box:
        # # Get the semantic sensor's observation space
        # semantic_space = self._sim.sensor_suite.observation_spaces.spaces[
        #     self._semantic_sensor_uuid
        # ]

        # # Create a new observation space with boolean values and the same shape
        # binary_space = spaces.Box(
        #     low=False,
        #     high=True,
        #     shape=semantic_space.shape,
        #     dtype=np.bool
        # )

        # return binary_space
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._semantic_sensor_uuid
        ]  # basically says the dimensions of the observations are the same as
        # the panoptic sensor

    def _modify_semantic_ids(self) -> None:
        # Change semantic id of everything but target objects to 0,
        # change semantic id of target objects to 1
        rom = self._sim.get_rigid_object_manager()
        for handle in rom.get_object_handles():
            obj = rom.get_object_by_handle(handle)
            if handle in self._target_object_handles:
                for node in obj.visual_scene_nodes:
                    node.semantic_id = (
                        obj.object_id + self._sim.habitat_config.object_ids_start
                    )  # to-do: figure out which semantic id will make it fully white
            else:
                for node in obj.visual_scene_nodes:
                    node.semantic_id = 0

    # This is called whenever reset is called or an action is taken
    def get_observation(
        self,
        *args: Any,
        observations: Observations,
        episode: RearrangeEpisode,
        **kwargs: Any,
    ) -> VisualObservation:
        self._target_object_handles = episode.targets.keys()
        self._modify_semantic_ids()

        # Get updated semantic sensor observations
        sim_obs = (
            self._sim.get_sensor_observations()
        )  # Fetch the simulator observations, all visual sensors.
        updated_semantic_observations = self._sim._sensor_suite.sensors[
            self._semantic_sensor_uuid
        ].get_observation(
            sim_obs
        )  # Post-process visual sensor observations

        # # Create a binary mask based on target object semantic IDs
        # # Initially set all pixels to False
        # binary_mask = np.zeros_like(updated_semantic_observations, dtype=np.bool)

        # # Set pixels corresponding to target objects to True
        # for handle in self._target_object_handles:
        #     semantic_id = handle + self._sim.habitat_config.object_ids_start
        #     binary_mask[updated_semantic_observations == semantic_id] = True

        # return binary_mask

        return updated_semantic_observations