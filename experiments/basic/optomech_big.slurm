#!/bin/bash
#SBATCH --job-name=optomech           # Job name
#SBATCH --output=output.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --partition=standard          # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --account=MHPCC96650DAS       # 
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --time=96:00:00                # Maximum runtime (D-HH:MM:SS)

#Load necessary modules (if needed)
#module load module_name

#Your job commands go here
#For example:
#python my_script.py

# 1: Baseline
cd ./visuomoto_deep_optics/
poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--exp_name=baseline \
--env-id optomech-v1 \
--seed=88887 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=256 \
--observation_mode=image_action \
--command_secondaries \
--model_wind_diff_motion \
--init_differential_motion \
--incremental_control \
--command_tip_tilt \
--save_model \
--model_save_interval=1000 \
--num_envs=1 \
--tau=0.001 \
--exploration_noise=0.3 \
--decay_rate=0.005 \
--reward_function=negastrehl \
--buffer_size=1_000_000 \
--learning_starts=10_000 \
--policy_frequency=1 \
--batch_size=128 \
--max_episode_steps=100 \
--actor_type=custom \
--critic_type=custom \
--actor_fc_scale=512 \
--qnetwork_fc_scale=512 \
--qnetwork_channel_scale=64 \
--actor_channel_scale=64 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \


# 2: Later Start
cd ./visuomoto_deep_optics/
poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--exp_name=later_start \
--env-id optomech-v1 \
--seed=88887 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=256 \
--observation_mode=image_action \
--command_secondaries \
--model_wind_diff_motion \
--init_differential_motion \
--incremental_control \
--command_tip_tilt \
--save_model \
--model_save_interval=1000 \
--num_envs=1 \
--tau=0.001 \
--exploration_noise=0.3 \
--decay_rate=0.005 \
--reward_function=negastrehl \
--buffer_size=1_000_000 \
--learning_starts=1_000_000 \
--policy_frequency=1 \
--batch_size=128 \
--max_episode_steps=100 \
--actor_type=custom \
--critic_type=custom \
--actor_fc_scale=512 \
--qnetwork_fc_scale=512 \
--qnetwork_channel_scale=64 \
--actor_channel_scale=64 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \


# 3: Smaller model.
cd ./visuomoto_deep_optics/
poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--exp_name=smaller_model \
--env-id optomech-v1 \
--seed=88887 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=256 \
--observation_mode=image_action \
--command_secondaries \
--model_wind_diff_motion \
--init_differential_motion \
--incremental_control \
--command_tip_tilt \
--save_model \
--model_save_interval=1000 \
--num_envs=1 \
--tau=0.001 \
--exploration_noise=0.3 \
--decay_rate=0.005 \
--reward_function=negastrehl \
--buffer_size=1_000_000 \
--learning_starts=10_000 \
--policy_frequency=1 \
--batch_size=128 \
--max_episode_steps=100 \
--actor_type=custom \
--critic_type=custom \
--actor_fc_scale=128 \
--qnetwork_fc_scale=128 \
--qnetwork_channel_scale=16 \
--actor_channel_scale=16 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \

# 4: Bigger model.
cd ./visuomoto_deep_optics/
poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--exp_name=bigger_model \
--env-id optomech-v1 \
--seed=88887 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=256 \
--observation_mode=image_action \
--command_secondaries \
--model_wind_diff_motion \
--init_differential_motion \
--incremental_control \
--command_tip_tilt \
--save_model \
--model_save_interval=1000 \
--num_envs=1 \
--tau=0.001 \
--exploration_noise=0.3 \
--decay_rate=0.005 \
--reward_function=negastrehl \
--buffer_size=1_000_000 \
--learning_starts=10_000 \
--policy_frequency=1 \
--batch_size=128 \
--max_episode_steps=100 \
--actor_type=custom \
--critic_type=custom \
--actor_fc_scale=1024 \
--qnetwork_fc_scale=1024 \
--qnetwork_channel_scale=64 \
--actor_channel_scale=64 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \



# 4: Bigger model.
cd ./visuomoto_deep_optics/
poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--exp_name=baseline \
--env-id optomech-v1 \
--seed=88887 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=256 \
--observation_mode=image_action \
--command_secondaries \
--model_wind_diff_motion \
--init_differential_motion \
--incremental_control \
--command_tip_tilt \
--save_model \
--model_save_interval=1000 \
--num_envs=1 \
--tau=0.001 \
--exploration_noise=0.3 \
--decay_rate=0.005 \
--reward_function=negastrehl \
--buffer_size=1_000_000 \
--learning_starts=10_000 \
--policy_frequency=1 \
--batch_size=128 \
--max_episode_steps=100 \
--actor_type=custom \
--critic_type=custom \
--actor_fc_scale=512 \
--qnetwork_fc_scale=512 \
--qnetwork_channel_scale=64 \
--actor_channel_scale=64 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \


# Slower decay.
cd ./visuomoto_deep_optics/
poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--exp_name=slower_decay \
--env-id optomech-v1 \
--seed=88887 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=256 \
--observation_mode=image_action \
--command_secondaries \
--model_wind_diff_motion \
--init_differential_motion \
--incremental_control \
--command_tip_tilt \
--save_model \
--model_save_interval=1000 \
--num_envs=1 \
--tau=0.001 \
--exploration_noise=0.5 \
--decay_rate=0.00005 \
--reward_function=negastrehl \
--buffer_size=1_000_000 \
--learning_starts=10_000 \
--policy_frequency=1 \
--batch_size=128 \
--max_episode_steps=100 \
--actor_type=custom \
--critic_type=custom \
--actor_fc_scale=512 \
--qnetwork_fc_scale=512 \
--qnetwork_channel_scale=64 \
--actor_channel_scale=64 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \

# Faster training
cd ./visuomoto_deep_optics/
poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--exp_name=faster_training \
--env-id optomech-v1 \
--seed=88887 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=256 \
--observation_mode=image_action \
--command_secondaries \
--model_wind_diff_motion \
--init_differential_motion \
--incremental_control \
--command_tip_tilt \
--save_model \
--model_save_interval=1000 \
--num_envs=1 \
--tau=0.01 \
--exploration_noise=0.3 \
--decay_rate=0.005 \
--reward_function=negastrehl \
--buffer_size=1_000_000 \
--learning_starts=10_000 \
--policy_frequency=1 \
--batch_size=256 \
--max_episode_steps=100 \
--actor_type=custom \
--critic_type=custom \
--actor_fc_scale=256 \
--qnetwork_fc_scale=256 \
--qnetwork_channel_scale=64 \
--actor_channel_scale=64 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \


# Reduced policy frequency
cd ./visuomoto_deep_optics/
poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--exp_name=reduced_policy_frequency \
--env-id optomech-v1 \
--seed=88887 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=256 \
--observation_mode=image_action \
--command_secondaries \
--model_wind_diff_motion \
--init_differential_motion \
--incremental_control \
--command_tip_tilt \
--save_model \
--model_save_interval=1000 \
--num_envs=1 \
--tau=0.001 \
--exploration_noise=0.3 \
--decay_rate=0.005 \
--reward_function=negastrehl \
--buffer_size=1_000_000 \
--learning_starts=10_000 \
--policy_frequency=4 \
--batch_size=128 \
--max_episode_steps=100 \
--actor_type=custom \
--critic_type=custom \
--actor_fc_scale=512 \
--qnetwork_fc_scale=512 \
--qnetwork_channel_scale=64 \
--actor_channel_scale=64 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \

#Optionally, you can include cleanup commands here (e.g., after the job finishes)
#For example:
#rm some_temp_file.txt