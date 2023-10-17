#!/bin/bash
#SBATCH --job-name=pick_binary_mask_eval
#SBATCH --chdir /srv/cvmlp-lab/flash1/smohanty61/habitat-lab
#SBATCH --output=/srv/cvmlp-lab/flash1/smohanty61/checkpoint/eval/pick_binary_mask.log
#SBATCH --error=/srv/cvmlp-lab/flash1/smohanty61/checkpoint/eval/pick_binary_mask.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 6
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --requeue
#SBATCH --partition short
#SBATCH --constraint a40

srun /nethome/smohanty61/flash/miniconda/envs/clip_grip/bin/python -u -m open_vocab_pick.run \
    --config-name='config/experiments/rl_skill_updated_sensors' \
    habitat_baselines.evaluate=True \
    habitat_baselines.tensorboard_dir=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/eval/tb_pick_binary_mask \
    habitat_baselines.checkpoint_folder=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/eval/checkpoints_pick_binary_mask \
    habitat_baselines.num_environments=21 \
    habitat_baselines.load_resume_state_config=False \
    habitat_baselines.eval.video_option='["disk"]' \
    habitat_baselines.video_dir=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/eval/video_pick_binary_mask \
    habitat_baselines.eval_ckpt_path_dir=/srv/flash1/smohanty61/results/pick_policy/train/checkpoints_pick_binary_mask \
    habitat_baselines.test_episode_count=20

# habitat_baselines.num_environments=20 consumes ~ 42 MiB with 2 gpu a40
