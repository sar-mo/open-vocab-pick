#!/bin/bash
#SBATCH --job-name=pick_jimmy_2
#SBATCH --chdir /srv/cvmlp-lab/flash1/smohanty61/habitat-lab
#SBATCH --output=/srv/cvmlp-lab/flash1/smohanty61/checkpoint/train/pick_jimmy.log
#SBATCH --error=/srv/cvmlp-lab/flash1/smohanty61/checkpoint/train/pick_jimmy.err
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 6
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --requeue
#SBATCH --partition short

srun /nethome/smohanty61/flash/miniconda/envs/clip_grip/bin/python -u -m habitat_baselines.run \
    --config-name=rearrange/rl_skill.yaml \
    benchmark/rearrange/skills=pick_spot \
    habitat.environment.max_episode_steps=1250 \
    habitat_baselines.evaluate=False \
    habitat_baselines.tensorboard_dir=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/train/tb_pick_jimmy_2 \
    habitat_baselines.video_dir=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/train/video_pick_jimmy_2 \
    habitat_baselines.checkpoint_folder=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/train/checkpoints_pick_jimmy_2 \
    habitat_baselines.writer_type=wb \
    habitat_baselines.wb.project_name=train_pick \
    habitat_baselines.wb.entity=sarmo \
    habitat_baselines.wb.run_name=pick_jimmy \
    habitat_baselines.num_checkpoints=50 \
    habitat.simulator.kinematic_mode=True \
    habitat.simulator.ac_freq_ratio=1 \
    habitat.task.measurements.force_terminate.max_accum_force=-1.0 \
    habitat.task.measurements.force_terminate.max_instant_force=-1.0 \
    habitat.task.actions.arm_action.delta_pos_limit=0.05 \
    habitat.task.actions.base_velocity.lin_speed=5.0 \
    habitat.task.actions.base_velocity.ang_speed=5.0 \
    habitat.task.measurements.pick_reward.force_pen=0.01 \
    habitat.task.measurements.pick_reward.max_force_pen=0.01 \
    habitat.task.measurements.pick_reward.force_end_pen=1.0 \
    habitat.task.measurements.collisions_terminate.max_scene_colls=100.0 \
    habitat_baselines.num_environments=24 \
    habitat_baselines.load_resume_state_config=False

# he had 24 envs
# # Command for eval
# # python -u -m habitat_baselines.run
# --config-name=rearrange/rl_skill.yaml
# habitat_baselines.evaluate=True
# habitat_baselines.tensorboard_dir=checkpoint/tb_pick_6_0
# habitat_baselines.video_dir=checkpoint/video_pick_6_0
# habitat_baselines.checkpoint_folder=checkpoint/checkpoints_pick_6_0
# habitat_baselines.eval_ckpt_path_dir=checkpoint/checkpoints_pick_6_0/ckpt.326.pth
# abitat_baselines.num_checkpoints=500
# habitat.simulator.kinematic_mode=True
# habitat.simulator.ac_freq_ratio=1
# habitat.task.measurements.force_terminate.max_accum_force=-1.0
# habitat.task.measurements.force_terminate.max_instant_force=-1.0
# habitat.task.actions.arm_action.delta_pos_limit=0.0125
# habitat.task.measurements.pick_reward.enable_vel_penality=1.0

    # habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz \

    # habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largetrain_37s_37kepi_2obj.json.gz \
    # habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz
    # data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz

    # python -u -m habitat_baselines.run
    # --config-name=rearrange/rl_skill.yaml
    # habitat_baselines.evaluate=False
    # habitat_baselines.tensorboard_dir=results/tmp/tb
    # habitat_baselines.checkpoint_folder=results/tmp/checkpoints
    # habitat_baselines.num_checkpoints=500
    # habitat.simulator.kinematic_mode=True
    # habitat.simulator.ac_freq_ratio=1
    # habitat.task.measurements.force_terminate.max_accum_force=-1.0
    #  habitat.task.measurements.force_terminate.max_instant_force=-1.0
    #  habitat.task.actions.arm_action.delta_pos_limit=0.0125
    #  habitat.task.measurements.pick_reward.enable_vel_penality=0.0
    #  habitat_baselines.num_environments=2
    #  habitat.task.measurements.pick_reward.force_pen=0.01
    #  habitat.task.measurements.pick_reward.max_force_pen=0.01
    #  habitat.task.measurements.pick_reward.force_end_pen=1.0
    #  habitat.task.measurements.collisions_terminate.max_scene_colls=100.0


​
# Command for training
# python -u -m habitat_baselines.run --config-name=rearrange/rl_skill.yaml habitat_baselines.evaluate=False habitat_baselines.tensorboard_dir=checkpoint/tb_debug_v1 habitat_baselines.video_dir=checkpoint/video_debug_v1 habitat_baselines.checkpoint_folder=checkpoint/checkpoints_debug_v1 habitat_baselines.eval_ckpt_path_dir=checkpoint/checkpoints_debug_v1 habitat_baselines.num_checkpoints=500 habitat.simulator.kinematic_mode=True habitat.simulator.ac_freq_ratio=1 habitat.task.measurements.force_terminate.max_accum_force=-1.0 habitat.task.measurements.force_terminate.max_instant_force=-1.0 habitat.task.actions.arm_action.delta_pos_limit=0.0125 habitat.task.measurements.pick_reward.enable_vel_penality=0.0 habitat_baselines.num_environments=2 benchmark/rearrange=pick_spot_joint_rest
​
# Command for training in FP
# python -u -m habitat_baselines.run
# --config-name=rearrange/rl_skill.yaml
# habitat_baselines.evaluate=False
# habitat_baselines.tensorboard_dir=checkpoint/tb_debug_v1
# habitat_baselines.video_dir=checkpoint/video_debug_v1
# habitat_baselines.checkpoint_folder=checkpoint/checkpoints_debug_v1
# habitat_baselines.eval_ckpt_path_dir=checkpoint/checkpoints_debug_v1
# habitat_baselines.num_checkpoints=500
# habitat.simulator.kinematic_mode=True
# habitat.simulator.ac_freq_ratio=1
# habitat.task.measurements.force_terminate.max_accum_force=-1.0
# habitat.task.measurements.force_terminate.max_instant_force=-1.0
# habitat.task.actions.arm_action.delta_pos_limit=0.0125
# habitat.task.measurements.pick_reward.enable_vel_penality=0.0
# habitat_baselines.num_environments=2
# benchmark/rearrange=pick_spot_joint_rest
# habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz


# python -u -m habitat_baselines.run \
# --config-name=rearrange/rl_skill.yaml \
# habitat_baselines.evaluate=False \
# habitat_baselines.tensorboard_dir=checkpoint/tb_debug_v1 \
# habitat_baselines.video_dir=checkpoint/video_debug_v1 \
# habitat_baselines.checkpoint_folder=checkpoint/checkpoints_debug_v1 \
# habitat_baselines.eval_ckpt_path_dir=checkpoint/checkpoints_debug_v1 \
# habitat_baselines.num_checkpoints=500 \
# habitat.simulator.kinematic_mode=True \
# habitat.simulator.ac_freq_ratio=1 \
# habitat.task.measurements.force_terminate.max_accum_force=-1.0 \
# habitat.task.measurements.force_terminate.max_instant_force=-1.0 \
# habitat.task.actions.arm_action.delta_pos_limit=0.0125 \
# habitat.task.measurements.pick_reward.enable_vel_penality=0.0 \
# habitat_baselines.num_environments=2 \
# benchmark/rearrange=pick_spot_joint_rest \
# habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz \
​
# Command for training in replica with the new collisions
# python -u -m habitat_baselines.run
# --config-name=rearrange/rl_skill.yaml \
# habitat_baselines.evaluate=False \
# habitat_baselines.tensorboard_dir=checkpoint/tb_debug_v2 \
# habitat_baselines.video_dir=checkpoint/video_debug_v2 \
# habitat_baselines.checkpoint_folder=checkpoint/checkpoints_debug_v2 \
# habitat_baselines.eval_ckpt_path_dir=checkpoint/checkpoints_debug_v2 \
# habitat_baselines.num_checkpoints=500 habitat.simulator.kinematic_mode=True \
# habitat.simulator.ac_freq_ratio=1 \
# habitat.task.measurements.force_terminate.max_accum_force=-1.0 \
# habitat.task.measurements.force_terminate.max_instant_force=-1.0 \
# habitat.task.actions.arm_action.delta_pos_limit=0.0125 \
# habitat.task.measurements.pick_reward.enable_vel_penality=0.0 \
# habitat_baselines.num_environments=2 \
# habitat.task.measurements.pick_reward.force_pen=0.01 \
# habitat.task.measurements.pick_reward.max_force_pen=0.01 \
# habitat.task.measurements.pick_reward.force_end_pen=1.0
# habitat.task.measurements.collisions_terminate.max_scene_colls=100.0
​
​
