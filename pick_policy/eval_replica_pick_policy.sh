#!/bin/bash
#SBATCH --job-name=pick_replica_cad_eval_2
#SBATCH --chdir /srv/cvmlp-lab/flash1/smohanty61/habitat-lab
#SBATCH --output=/srv/cvmlp-lab/flash1/smohanty61/checkpoint/eval/pick_2.log
#SBATCH --error=/srv/cvmlp-lab/flash1/smohanty61/checkpoint//eval/pick_2.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 6
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --requeue
#SBATCH --partition short
#SBATCH --constraint a40

srun /nethome/smohanty61/flash/miniconda/envs/clip_grip/bin/python -u -m habitat_baselines.run \
    --config-name=rearrange/rl_skill.yaml \
    habitat.environment.max_episode_steps=1250 \
    habitat_baselines.evaluate=True \
    habitat_baselines.tensorboard_dir=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/eval/tb_pick_replica_2 \
    habitat_baselines.checkpoint_folder=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/eval/checkpoints_pick_replica_2 \
    habitat_baselines.eval_ckpt_path_dir=/srv/flash1/smohanty61/results/pick_policy/train/checkpoints_pick_replica_2/ckpt.65.pth \
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
    habitat_baselines.load_resume_state_config=False \
    habitat_baselines.eval.video_option='["disk"]' \
    habitat_baselines.video_dir=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/eval/video_pick_replica_2 \
    habitat_baselines.test_episode_count=5

# Note that when using habitat_baselines.load_resume_state_config=True, you cannot set habitat_baselines.num_environments to be other value, otherwise, there will be an issue
# Since there were only 21 scenes, I think I cannot have more than 21 num_environments.

# # add these lines when trying to generate a video
# habitat_baselines.eval.video_option='["disk"]'
# habitat_baselines.video_dir="path/to/your/video/dir"
# # and this when testing only a few episodes
# habitat_baselines.test_episode_count={NUMBER_OF_EPISODES}

# # you can also train the dataset in the fp scenes using 
# benchmark/rearrange=pick_spot_joint_rest
# habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz