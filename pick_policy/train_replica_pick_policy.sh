#!/bin/bash
#SBATCH --job-name=pick_replica_cad_train_2
#SBATCH --chdir /srv/cvmlp-lab/flash1/smohanty61/habitat-lab
#SBATCH --output=/srv/cvmlp-lab/flash1/smohanty61/checkpoint/train/pick_2.log
#SBATCH --error=/srv/cvmlp-lab/flash1/smohanty61/checkpoint/train/pick_2.err
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 6
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --requeue
#SBATCH --partition short
#SBATCH --constraint a40

srun /nethome/smohanty61/flash/miniconda/envs/clip_grip/bin/python -u -m habitat_baselines.run \
    --config-name=rearrange/rl_skill.yaml \
    benchmark/rearrange=pick_spot \
    habitat.environment.max_episode_steps=1250 \
    habitat_baselines.evaluate=False \
    habitat_baselines.tensorboard_dir=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/train/tb_pick_replica_2 \
    habitat_baselines.checkpoint_folder=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/train/checkpoints_pick_replica_2 \
    habitat_baselines.num_checkpoints=500 \
    habitat.simulator.kinematic_mode=True \
    habitat.simulator.ac_freq_ratio=1 \
    habitat.task.measurements.force_terminate.max_accum_force=-1.0 \
    habitat.task.measurements.force_terminate.max_instant_force=-1.0 \
    habitat.task.actions.arm_action.delta_pos_limit=0.05 \
    habitat.task.measurements.pick_reward.enable_vel_penality=-1.0 \
    habitat.task.actions.base_velocity.lin_speed=5.0 \
    habitat.task.actions.base_velocity.ang_speed=5.0 \
    habitat.task.measurements.pick_reward.force_pen=0.01 \
    habitat.task.measurements.pick_reward.max_force_pen=0.01 \
    habitat.task.measurements.pick_reward.force_end_pen=1.0 \
    habitat.task.measurements.collisions_terminate.max_scene_colls=100.0 \
    habitat_baselines.num_environments=21 \
    habitat_baselines.load_resume_state_config=False