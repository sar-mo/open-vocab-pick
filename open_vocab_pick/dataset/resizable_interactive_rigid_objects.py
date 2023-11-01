import json
import math
import multiprocessing
import os
from typing import Dict, List, Tuple

import git
import habitat_sim
import magnum as mn
import numpy as np
import objaverse
from habitat.core.simulator import (
    Observations,
    Simulator,
)
from habitat_sim.attributes_managers import ObjectAttributesManager
from habitat_sim.physics import ManagedRigidObject, RigidObjectManager

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
os.chdir(dir_path)
data_path = os.path.join(dir_path, "data")
output_directory = "/"  # @param {type:"string"}
output_path = os.path.join(dir_path, output_directory)
if not os.path.exists(output_path):
    os.mkdir(output_path)

# define some globals the first time we run.
if "sim" not in globals():
    global sim
    sim = None
    global obj_attr_mgr
    obj_attr_mgr = None
    global prim_attr_mgr
    obj_attr_mgr = None
    global stage_attr_mgr
    stage_attr_mgr = None
    global rigid_obj_mgr
    rigid_obj_mgr = None

# %%
# @title Define Configuration Utility Functions { display-mode: "form" }
# @markdown (double click to show code)

# @markdown This cell defines a number of utility functions used throughout the tutorial
# to make simulator reconstruction easy:
# @markdown - make_cfg
# @markdown - make_default_settings
# @markdown - make_simulator_from_settings


def make_cfg(settings: Dict[str, str]) -> habitat_sim.Configuration:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    # Optional; Specify the location of an existing scene dataset configuration
    # that describes the locations and configurations of all the assets to be used
    if "scene_dataset_config" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config"]

    # Note: all sensors must have the same resolution
    sensor_specs = []
    if settings["color_sensor_1st_person"]:
        color_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        color_sensor_1st_person_spec.uuid = "color_sensor_1st_person"
        color_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        color_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        color_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_1st_person_spec)
    if settings["depth_sensor_1st_person"]:
        depth_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_1st_person_spec.uuid = "depth_sensor_1st_person"
        depth_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        depth_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        depth_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_1st_person_spec)
    if settings["semantic_sensor_1st_person"]:
        semantic_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_1st_person_spec.uuid = "semantic_sensor_1st_person"
        semantic_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_1st_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        semantic_sensor_1st_person_spec.position = [
            0.0,
            settings["sensor_height"],
            0.0,
        ]
        semantic_sensor_1st_person_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        semantic_sensor_1st_person_spec.sensor_subtype = (
            habitat_sim.SensorSubType.PINHOLE
        )
        sensor_specs.append(semantic_sensor_1st_person_spec)
    if settings["color_sensor_3rd_person"]:
        color_sensor_3rd_person_spec = habitat_sim.CameraSensorSpec()
        color_sensor_3rd_person_spec.uuid = "color_sensor_3rd_person"
        color_sensor_3rd_person_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_3rd_person_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        color_sensor_3rd_person_spec.position = [
            0.0,
            settings["sensor_height"] + 0.2,
            0.2,
        ]
        color_sensor_3rd_person_spec.orientation = [-math.pi / 4, 0, 0]
        color_sensor_3rd_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_3rd_person_spec)

    # Here you can specify the amount of displacement in a forward action and turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def make_default_settings() -> Dict[str, str]:
    settings = {
        "width": 720,  # Spatial resolution of the observations
        "height": 544,
        # Scene path
        "scene": "./data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb",
        "scene_dataset_config": "./data/scene_datasets/mp3d_example/mp3d.scene_dataset_config.json",
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "sensor_pitch": -math.pi / 8.0,  # sensor pitch (x rotation in rads)
        "color_sensor_1st_person": True,  # RGB sensor
        "color_sensor_3rd_person": False,  # RGB sensor 3rd person
        "depth_sensor_1st_person": False,  # Depth sensor
        "semantic_sensor_1st_person": False,  # Semantic sensor
        "seed": 1,
        "enable_physics": True,  # enable dynamics simulation
    }
    return settings


def make_simulator_from_settings(sim_settings: Dict[str, str]) -> None:
    cfg = make_cfg(sim_settings)
    # clean-up the current simulator instance if it exists
    global sim
    global obj_attr_mgr
    global prim_attr_mgr
    global stage_attr_mgr
    global rigid_obj_mgr
    if sim != None:
        sim.close()
    # initialize the simulator
    sim = habitat_sim.Simulator(cfg)
    # Managers of various Attributes templates
    obj_attr_mgr = sim.get_object_template_manager()
    prim_attr_mgr = sim.get_asset_template_manager()
    stage_attr_mgr = sim.get_stage_template_manager()
    # Manager providing access to rigid objects
    rigid_obj_mgr = sim.get_rigid_object_manager()


def simulate(
    sim: Simulator, dt: float = 1.0, get_frames: bool = True
) -> Observations:
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())
    return observations


# Set an object transform relative to the agent state
def set_object_state_from_agent(
    sim,
    obj,
    offset=np.array([0, 2.0, -1.5]),
    orientation=mn.Quaternion(((0, 0, 0), 1)),
):
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    ob_translation = agent_transform.transform_point(offset)
    obj.translation = ob_translation
    obj.rotation = orientation

# %%
# @title Initialize Simulator and Load Scene { display-mode: "form" }

# convienience functions defined in Utility cell manage global variables
sim_settings = make_default_settings()
# set globals: sim,
make_simulator_from_settings(sim_settings)

# %%
lvis_annotations : Dict[str, List[str]] = objaverse.load_lvis_annotations()
lvis_annotations

# just get two categories for now
lvis_annotations = {k: v for k, v in lvis_annotations.items() \
                    if k in ["Band_Aid", "Bible"]}
lvis_annotations

# %%
# download LVIS annotations

def download_objects(uids: str) -> Dict:
    """
    Downloads and returns objects from objaverse based on the provided UIDs.

    Parameters:
        uids (str): UID of the object to download.

    Returns:
        Dict: Objects indexed by their UIDs.
    """
    processes = multiprocessing.cpu_count()
    return objaverse.load_objects(uids=uids, download_processes=processes)

for category, uids in lvis_annotations.items():
    object_descriptions = download_objects(uids)

# %%
introduce_surface = True  # @param{type:"boolean"}
rigid_obj_mgr.remove_all_objects()

def calculate_scaling_factor(
        obj_bb_size: Tuple[float, float, float],
        max_threshold: float,
        min_threshold: float
    ) -> float:
    """
    Calculate the scaling factor based on object bounding box size and thresholds.

    Parameters:
        obj_bb_size (Tuple[float, float, float]): The size of the object's bounding box.
        max_threshold (float): Maximum size threshold for scaling.
        min_threshold (float): Minimum size threshold for scaling.

    Returns:
        float: The scaling factor.
    """
    max_size = max(obj_bb_size)

    if max_size > max_threshold:
        return max_threshold / max_size
    elif max_size < min_threshold:
        return min_threshold / max_size
    else:
        return 1.0  # No scaling required

def adjust_object_scale(
        rigid_obj_mgr: RigidObjectManager,
        obj_attr_mgr: ObjectAttributesManager,
        obj: ManagedRigidObject,
        scale_factor: float
    ) -> None:
    """
    Adjusts the scale of the object and updates it in the simulation.

    Parameters:
        rigid_obj_mgr (RigidObjectManager): Manager responsible for rigid objects
            in simulation.
        obj_template_mgr: Manager for object templates.
        obj: The object instance.
        scale_factor (float): The scaling factor to apply.
    """
    # Remove the current object
    rigid_obj_mgr.remove_object_by_id(obj.object_id)

    # # Adjust and update the object's scale
    # obj_template.scale = mn.Vector3(scale_factor)
    # obj_template_id = obj_attr_mgr.register_template(obj_template)

    # # Add the adjusted object back to the simulation
    # rigid_obj_mgr.add_object_by_template_id(obj_template_id)

def generate_objects_info_file(
        sim: Simulator,
        lvis_annotations: Dict[str, List[str]],
    ) -> None:
    """
    Adds objects to the simulator and saves their scaling factors to a JSON file.

    Parameters:
        sim (Simulator): Habitat simulator instance.
        lvis_annotations : Dictionary of object categories and UIDs.
    """
    objaverse_objects_info = dict()

    for category, objects in lvis_annotations.items():
        for obj_uid in objects:
            print(f'uid: {obj_uid}')
            # Download and create a new object template
            obj_file_path = list(download_objects([obj_uid]).values())[0]
            obj_template = obj_attr_mgr.create_new_template(obj_file_path, False)
            obj_template_id = obj_attr_mgr.register_template(obj_template)

            # Add the object to the simulator
            obj = rigid_obj_mgr.add_object_by_template_id(obj_template_id)

            # Calculate and apply the scaling factor
            obj_bb_size = obj.root_scene_node.cumulative_bb.size()
            scale_factor = calculate_scaling_factor(obj_bb_size, 0.25, 0.1)

            adjust_object_scale(rigid_obj_mgr, obj_attr_mgr, obj, scale_factor)

            objaverse_objects_info[obj_uid] = {
                "category": category,
                "scale_factor": scale_factor
            }
    # Save scaling factors to a JSON file
    with open('objaverse_objects_info.json', 'w') as json_file:
        json.dump(objaverse_objects_info, json_file, indent=4)

generate_objects_info_file(sim, lvis_annotations)

# example_type = "resizable_objaverse_objects"
# observations = simulate(sim, dt=3.0)
# if make_video:
#     vut.make_video(
#         observations,
#         "color_sensor_1st_person",
#         "color",
#         output_path + example_type,
#         open_vid=show_video,
#     )
rigid_obj_mgr.remove_all_objects()