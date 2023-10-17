#!/bin/bash
#SBATCH --job-name=pick_debug_train
#SBATCH --chdir /srv/cvmlp-lab/flash1/smohanty61/habitat-lab
#SBATCH --output=/srv/cvmlp-lab/flash1/smohanty61/checkpoint/train/pick_debug.log
#SBATCH --error=/srv/cvmlp-lab/flash1/smohanty61/checkpoint/train/pick_debug.err
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 6
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --requeue
#SBATCH --partition short
#SBATCH --constraint a40
#SBATCH --exclude calculon,alexa,cortana,bmo,t1000,vicki,baymax

srun /nethome/smohanty61/flash/miniconda/envs/clip_grip/bin/python -u -m open_vocab_pick.run \
    --config-name='config/experiments/rl_skill_updated_sensors' \
    habitat_baselines.evaluate=False \
    habitat_baselines.tensorboard_dir=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/train/tb_pick_debug \
    habitat_baselines.checkpoint_folder=/srv/cvmlp-lab/flash1/smohanty61/results/pick_policy/train/checkpoints_pick_debug \
    habitat_baselines.num_environments=21 \
    habitat_baselines.load_resume_state_config=False

# habitat_baselines.num_environments=20 consumes ~ 42 MiB with 2 gpu a40
