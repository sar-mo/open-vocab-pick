#!/usr/bin/env python3


from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from omegaconf import DictConfig


import numpy as np
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import (
    Observations,
    Simulator,
)
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.tasks.rearrange.rearrange_sensors import (
    EndEffectorToObjectDistance,
    ForceTerminate,
    RearrangeReward,
    RobotForce,
)
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import rearrange_logger

from open_vocab_pick.utils.utils import get_camera_object_angle


@registry.register_measure
class RearrangePickRewardV2(RearrangeReward):
    cls_uuid: str = "pick_reward_v2"

    def __init__(
        self,
        *args: Any,
        sim: Simulator,
        config: "DictConfig",
        task: RearrangeTask,
        **kwargs: Any,
    ) -> None:
        self.cur_dist = -1.0
        self._prev_picked = False
        self._metric = None
        self._last_pos: Optional[np.ndarray] = None
        self._last_rot: Optional[float] = None

        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return RearrangePickRewardV2.cls_uuid

    def reset_metric(
        self,
        *args: Any,
        episode: RearrangeEpisode,
        task: RearrangeTask,
        observations: Observations,
        **kwargs: Any,
    ) -> None:
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                EndEffectorToObjectDistance.cls_uuid,
                RobotForce.cls_uuid,
                ForceTerminate.cls_uuid,
            ],
        )
        self.cur_dist = -1.0
        self._prev_picked = self._sim.grasp_mgr.snap_idx is not None

        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def get_gaze_distance(self, ee_dist: float, target_obj_id: str) -> float:
        rom = self._sim.get_rigid_object_manager()
        obj_pos = rom.get_object_by_id(target_obj_id).translation
        object_angle = get_camera_object_angle(self._sim, obj_pos)
        # Spot gripper should be about 0.4 m away from the object
        dist_to_goal = object_angle + abs(ee_dist - self._config.gaze_ee_goal_dist)

        return dist_to_goal

    def update_metric(
        self,
        *args: Any,
        episode: RearrangeEpisode,
        task: RearrangeTask,
        observations: Observations,
        **kwargs: Any,
    ) -> None:
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )
        ee_to_object_distance = task.measurements.measures[
            EndEffectorToObjectDistance.cls_uuid
        ].get_metric()

        snapped_id = self._sim.grasp_mgr.snap_idx
        cur_picked = snapped_id is not None

        abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]

        if self._config.using_gaze_grasp:
            dist_to_goal = self.get_gaze_distance(
                ee_to_object_distance[str(task.targ_idx)], abs_targ_obj_idx
            )
        else:
            dist_to_goal = ee_to_object_distance[str(task.targ_idx)]

        # ------------

        # Penalize the base movement
        # Check if the robot moves or not
        move = 0.0
        if self._last_pos is not None and self._last_rot is not None:
            cur_pos = np.array(self._sim.articulated_agent.base_pos)
            last_rot = float(self._sim.articulated_agent.base_rot)
            if (
                np.linalg.norm(self._last_pos - cur_pos) >= 0.01
                or abs(self._last_rot - last_rot) >= 0.01
            ):
                move = 1.0

        # Get the control inputs
        lin_vel = np.array(
            self._task.actions["base_velocity"].base_vel_ctrl.linear_velocity
        )
        lin_vel_norm = np.linalg.norm(lin_vel)

        ang_vel = np.array(
            self._task.actions["base_velocity"].base_vel_ctrl.angular_velocity
        )
        ang_vel_norm = np.linalg.norm(ang_vel)

        if self._config.enable_vel_penality != -1:
            self._metric -= (
                move
                * self._config.enable_vel_penality
                * (lin_vel_norm**2 + ang_vel_norm**2)
            )

        # Update the last location of the robot
        self._last_pos = np.array(self._sim.articulated_agent.base_pos)
        self._last_rot = float(self._sim.articulated_agent.base_rot)

        # ------------

        did_pick = cur_picked and (not self._prev_picked)
        if did_pick:
            if snapped_id == abs_targ_obj_idx:
                self._metric += self._config.pick_reward
                # If we just transitioned to the next stage our current
                # distance is stale.
                self.cur_dist = -1
            else:
                # picked the wrong object
                self._metric -= self._config.wrong_pick_pen
                if self._config.wrong_pick_should_end:
                    rearrange_logger.debug("Grasped wrong object, ending episode.")
                    self._task.should_end = True
                self._prev_picked = cur_picked
                self.cur_dist = -1
                return

        if self._config.use_diff:
            if self.cur_dist < 0:
                dist_diff = 0.0
            else:
                dist_diff = self.cur_dist - dist_to_goal

            # Filter out the small fluctuations
            dist_diff = round(dist_diff, 3)
            self._metric += self._config.dist_reward * dist_diff
        else:
            self._metric -= self._config.dist_reward * dist_to_goal
        self.cur_dist = dist_to_goal

        if not cur_picked and self._prev_picked:
            # Dropped the object
            self._metric -= self._config.drop_pen
            if self._config.drop_obj_should_end:
                self._task.should_end = True
            self._prev_picked = cur_picked
            self.cur_dist = -1
            return

        self._prev_picked = cur_picked


@registry.register_measure
class RearrangePickSuccessV2(Measure):
    cls_uuid: str = "pick_success_v2"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ) -> None:
        self._sim = sim
        self._config = config
        # self._prev_ee_pos = None
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return RearrangePickSuccessV2.cls_uuid

    def reset_metric(
        self,
        *args: Any,
        episode: RearrangeEpisode,
        task: RearrangeTask,
        observations: Observations,
        **kwargs: Any,
    ) -> None:
        task.measurements.check_measure_dependencies(
            self.uuid, [EndEffectorToObjectDistance.cls_uuid]
        )
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(
        self,
        *args: Any,
        episode: RearrangeEpisode,
        task: RearrangeTask,
        observations: Observations,
        **kwargs: Any,
    ) -> None:
        # Is the agent holding the object and it's at the start?
        abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]

        # Check that we are holding the right object and the object is actually
        # being held.
        self._metric = (
            abs_targ_obj_idx == self._sim.grasp_mgr.snap_idx
            and not self._sim.grasp_mgr.is_violating_hold_constraint()
        )
