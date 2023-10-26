#!/bin/bash
#SBATCH --job-name=pick_teacher_VER_2_eval
#SBATCH --chdir /srv/cvmlp-lab/flash1/smohanty61/habitat-lab
#SBATCH --output=/srv/cvmlp-lab/flash1/smohanty61/checkpoint/eval/pick_teacher_arm_rgb_removed.log
#SBATCH --error=/srv/cvmlp-lab/flash1/smohanty61/checkpoint/eval/pick_teacher_arm_rgb_removed.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 6
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --requeue
#SBATCH --partition short

srun /nethome/smohanty61/flash/miniconda/envs/clip_grip/bin/python -u -m open_vocab_pick.run \
    --config-name='config/experiments/rl_skill_teacher_policy' \
    habitat_baselines.evaluate=True \
    habitat_baselines.tensorboard_dir=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/eval/tb_pick_teacher_arm_rgb_removed \
    habitat_baselines.checkpoint_folder=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/eval/checkpoints_pick_teacher_arm_rgb_removed \
    habitat_baselines.num_environments=15 \
    habitat_baselines.load_resume_state_config=False \
    habitat_baselines.eval_ckpt_path_dir=/srv/flash1/smohanty61/results/pick_policy/train/checkpoints_pick_teacher_VER/ckpt.16.pth
    # habitat_baselines.video_dir=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/eval/video_pick_teacher_VER \
    # habitat_baselines.test_episode_count=5
    # habitat_baselines.eval.video_option='["disk"]'
