import magnum as mn
import numpy as np
from habitat.articulated_agents.robots.spot_robot import SpotRobot
from habitat.articulated_agents.robots.stretch_robot import StretchRobot
from habitat.core.simulator import (
    Simulator,
)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    cosine = np.clip(np.dot(v1, v2), -1.0, 1.0)
    object_angle = np.arccos(cosine)
    return object_angle


def get_camera_object_angle(sim: Simulator, obj_pos: np.ndarray) -> float:
    """Calculates angle between gripper line-of-sight and given global position."""

    # Get the camera transformation
    cam_T = get_camera_transform(sim)

    # Get object location in camera frame
    cam_obj_pos = cam_T.inverted().transform_point(obj_pos).normalized()

    # Get angle between (normalized) location and the vector that the camera should
    # look at
    obj_angle = angle_between(cam_obj_pos, mn.Vector3(0, 0, -1))
    # original method used obj_angle = cosine(cam_obj_pos, mn.Vector3(0, 1, 0)

    return obj_angle


def get_camera_transform(sim: Simulator) -> mn.Matrix4:
    agent = sim.get_agent_data(agent_idx=None).articulated_agent

    if isinstance(agent, SpotRobot):
        cam_info = agent.params.cameras["articulated_agent_arm_depth"]
    elif isinstance(agent, StretchRobot):
        cam_info = agent.params.cameras["robot_head"]
    else:
        raise NotImplementedError("This robot does not have GazeGraspAction.")

    # Get the camera's attached link
    link_trans = agent.sim_obj.get_link_scene_node(
        cam_info.attached_link_id
    ).transformation
    # Get the camera offset transformation
    offset_trans = mn.Matrix4.translation(cam_info.cam_offset_pos)
    cam_trans = link_trans @ offset_trans @ cam_info.relative_transform

    return cam_trans
