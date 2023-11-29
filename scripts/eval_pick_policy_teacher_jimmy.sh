#!/bin/bash
#SBATCH --job-name=pick_teacher_jimmy
#SBATCH --output=/srv/cvmlp-lab/flash1/smohanty61/checkpoint/eval/tb_pick_teacher_jimmy.log
#SBATCH --error=/srv/cvmlp-lab/flash1/smohanty61/checkpoint/eval/tb_pick_teacher_jimmy.err
#SBATCH --chdir /srv/cvmlp-lab/flash1/smohanty61/habitat-lab
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 10
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --partition=short

countcollpen=0.05
maxcountcolls=100
runid=1
srun /nethome/smohanty61/flash/miniconda/envs/clip_grip/bin/python -u -m habitat_baselines.run \
    --config-name=rearrange/rl_skill.yaml  \
    benchmark/rearrange/skills=pick_spot \
    habitat_baselines.num_checkpoints=50 \
    habitat_baselines.total_num_steps=1.0e8 \
    habitat_baselines.num_environments=5 \
    habitat_baselines.tensorboard_dir=/srv/cvmlp-lab/flash1/smohanty61/results/eval/tb_pick_teacher_jimmy \
    habitat_baselines.checkpoint_folder=/srv/cvmlp-lab/flash1/smohanty61/results/eval/checkpoints_pick_teacher_jimmy \
    habitat_baselines.writer_type=wb \
    habitat_baselines.wb.project_name=gaze_grasping \
    habitat_baselines.wb.entity=sarmo \
    habitat_baselines.wb.run_name=pick_teacher_jimmy_eval \
    hydra.job.name=mg3 \
    habitat.task.actions.base_velocity_non_cylinder.longitudinal_lin_speed=2.0 \
    habitat.task.actions.base_velocity_non_cylinder.ang_speed=2.0  \
    habitat.task.actions.base_velocity_non_cylinder.allow_dyn_slide=False \
    habitat.task.actions.arm_action.delta_pos_limit=0.01667 \
    habitat.task.measurements.pick_reward.dist_reward=20.0 \
    habitat.task.measurements.pick_reward.wrong_pick_pen=5.0 \
    habitat.task.measurements.pick_reward.count_coll_pen=$countcollpen \
    habitat.task.measurements.pick_reward.max_count_colls=$maxcountcolls \
    habitat.task.measurements.pick_reward.count_coll_end_pen=5 \
    habitat.task.success_reward=10.0 \
    habitat.task.slack_reward=-0.01 \
    habitat.environment.max_episode_steps=1500 \
    habitat.simulator.kinematic_mode=True \
    habitat.simulator.ac_freq_ratio=4 \
    habitat.simulator.ctrl_freq=120 \
    habitat_baselines.evaluate=True \
    habitat_baselines.eval.video_option='["disk"]' \
    habitat_baselines.test_episode_count=20 \
    habitat_baselines.eval_ckpt_path_dir=/srv/cvmlp-lab/flash1/smohanty61/results/train/checkpoints_pick_teacher_jimmy/ckpt.2.pth \
    habitat_baselines.video_dir=/srv/cvmlp-lab/flash1/smohanty61/results/eval/video_pick_teacher_jimmy \
    habitat_baselines.load_resume_state_config=False

## 33000 VRAM
# habitat_baselines.eval.video_option: [] \