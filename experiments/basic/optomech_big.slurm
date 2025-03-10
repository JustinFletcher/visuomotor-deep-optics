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

cd ./visuomoto_deep_optics/

# 1: Baseline
poetry run python ./optomech/ddpg_lstm.py \
--env-id optomech-v1 \
--seed=88887 \
--gpu_list=0 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=64 \
--observation_mode=image_only \
--command_secondaries \
--incremental_control \
--model_wind_diff_motion \
--init_differential_motion \
--save_model \
--model_save_interval=10000 \
--num_envs=1 \
--tau=0.0001 \
--exploration_noise=1.0 \
--decay_rate=0.001 \
--reward_function=negastrehl \
--action_scale=0.01 \
--reward_scale=10.0 \
--buffer_size=1_000_000 \
--learning_starts=100_000 \
--prelearning_sample=scales \
--policy_frequency=1 \
--batch_size=32 \
--max_episode_steps=200 \
--actor_type=impala \
--critic_type=impala \
--actor_fc_scale=256 \
--qnetwork_fc_scale=256 \
--qnetwork_channel_scale=16 \
--actor_channel_scale=16 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \
&


# 2: Later Start
poetry run python ./optomech/ddpg_lstm.py \
--env-id optomech-v1 \
--seed=88887 \
--gpu_list=1 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=64 \
--observation_mode=image_only \
--command_secondaries \
--incremental_control \
--model_wind_diff_motion \
--init_differential_motion \
--save_model \
--model_save_interval=10000 \
--num_envs=1 \
--tau=0.0001 \
--exploration_noise=1.0 \
--decay_rate=0.001 \
--reward_function=negastrehl \
--action_scale=0.001 \
--reward_scale=10.0 \
--buffer_size=1_000_000 \
--learning_starts=100_000 \
--prelearning_sample=scales \
--policy_frequency=1 \
--batch_size=32 \
--max_episode_steps=200 \
--actor_type=impala \
--critic_type=impala \
--actor_fc_scale=256 \
--qnetwork_fc_scale=256 \
--qnetwork_channel_scale=16 \
--actor_channel_scale=16 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \
&



# 3: Smaller model.
poetry run python ./optomech/ddpg_lstm.py \
--env-id optomech-v1 \
--seed=88887 \
--gpu_list=2 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=64 \
--observation_mode=image_only \
--command_secondaries \
--incremental_control \
--model_wind_diff_motion \
--init_differential_motion \
--save_model \
--model_save_interval=10000 \
--num_envs=1 \
--tau=0.0001 \
--exploration_noise=1.0 \
--decay_rate=0.001 \
--reward_function=negastrehl \
--action_scale=0.0001 \
--reward_scale=10.0 \
--buffer_size=1_000_000 \
--learning_starts=100_000 \
--prelearning_sample=scales \
--policy_frequency=1 \
--batch_size=32 \
--max_episode_steps=200 \
--actor_type=impala \
--critic_type=impala \
--actor_fc_scale=256 \
--qnetwork_fc_scale=256 \
--qnetwork_channel_scale=16 \
--actor_channel_scale=16 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \
&


# 4: Bigger model.
poetry run python ./optomech/ddpg_lstm.py \
--env-id optomech-v1 \
--seed=88887 \
--gpu_list=3 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=64 \
--observation_mode=image_only \
--command_secondaries \
--incremental_control \
--model_wind_diff_motion \
--init_differential_motion \
--save_model \
--model_save_interval=10000 \
--num_envs=1 \
--tau=0.0001 \
--exploration_noise=1.0 \
--decay_rate=0.001 \
--reward_function=negastrehl \
--action_scale=0.00001 \
--reward_scale=10.0 \
--buffer_size=1_000_000 \
--learning_starts=100_000 \
--prelearning_sample=scales \
--policy_frequency=1 \
--batch_size=32 \
--max_episode_steps=200 \
--actor_type=impala \
--critic_type=impala \
--actor_fc_scale=256 \
--qnetwork_fc_scale=256 \
--qnetwork_channel_scale=16 \
--actor_channel_scale=16 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \
&




# 4: Bigger model.
poetry run python ./optomech/ddpg_lstm.py \
--env-id optomech-v1 \
--seed=88887 \
--gpu_list=4 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=64 \
--observation_mode=image_only \
--command_secondaries \
--incremental_control \
--model_wind_diff_motion \
--init_differential_motion \
--save_model \
--model_save_interval=10000 \
--num_envs=1 \
--tau=0.0001 \
--exploration_noise=1.0 \
--decay_rate=0.001 \
--reward_function=negastrehl \
--action_scale=0.000001 \
--reward_scale=10.0 \
--buffer_size=1_000_000 \
--learning_starts=100_000 \
--prelearning_sample=scales \
--policy_frequency=1 \
--batch_size=32 \
--max_episode_steps=200 \
--actor_type=impala \
--critic_type=impala \
--actor_fc_scale=256 \
--qnetwork_fc_scale=256 \
--qnetwork_channel_scale=16 \
--actor_channel_scale=16 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \
&



# Slower decay.
poetry run python ./optomech/ddpg_lstm.py \
--env-id optomech-v1 \
--seed=88887 \
--gpu_list=5 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=64 \
--observation_mode=image_only \
--command_secondaries \
--incremental_control \
--model_wind_diff_motion \
--init_differential_motion \
--save_model \
--model_save_interval=10000 \
--num_envs=1 \
--tau=0.0001 \
--exploration_noise=1.0 \
--decay_rate=0.001 \
--reward_function=negastrehl \
--action_scale=0.0000001 \
--reward_scale=10.0 \
--buffer_size=1_000_000 \
--learning_starts=100_000 \
--prelearning_sample=scales \
--policy_frequency=1 \
--batch_size=32 \
--max_episode_steps=200 \
--actor_type=impala \
--critic_type=impala \
--actor_fc_scale=256 \
--qnetwork_fc_scale=256 \
--qnetwork_channel_scale=16 \
--actor_channel_scale=16 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \
&


# Faster training
poetry run python ./optomech/ddpg_lstm.py \
--env-id optomech-v1 \
--seed=88887 \
--gpu_list=6 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=64 \
--observation_mode=image_only \
--command_secondaries \
--incremental_control \
--model_wind_diff_motion \
--init_differential_motion \
--save_model \
--model_save_interval=10000 \
--num_envs=1 \
--tau=0.0001 \
--exploration_noise=1.0 \
--decay_rate=0.001 \
--reward_function=negastrehl \
--action_scale=0.00000001 \
--reward_scale=10.0 \
--buffer_size=1_000_000 \
--learning_starts=100_000 \
--prelearning_sample=scales \
--policy_frequency=1 \
--batch_size=32 \
--max_episode_steps=200 \
--actor_type=impala \
--critic_type=impala \
--actor_fc_scale=256 \
--qnetwork_fc_scale=256 \
--qnetwork_channel_scale=16 \
--actor_channel_scale=16 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \
&



# Reduced policy frequency
poetry run python ./optomech/ddpg_lstm.py \
--env-id optomech-v1 \
--seed=88887 \
--gpu_list=7 \
--object_type=single \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=nanoelf \
--focal_plane_image_size_pixels=64 \
--observation_mode=image_only \
--command_secondaries \
--incremental_control \
--model_wind_diff_motion \
--init_differential_motion \
--save_model \
--model_save_interval=10000 \
--num_envs=1 \
--tau=0.0001 \
--exploration_noise=1.0 \
--decay_rate=0.00000001 \
--reward_function=negastrehl \
--action_scale=0.0001 \
--reward_scale=10.0 \
--buffer_size=1_000_000 \
--learning_starts=100_000 \
--prelearning_sample=scales \
--policy_frequency=1 \
--batch_size=32 \
--max_episode_steps=200 \
--actor_type=impala \
--critic_type=impala \
--actor_fc_scale=256 \
--qnetwork_fc_scale=256 \
--qnetwork_channel_scale=16 \
--actor_channel_scale=16 \
--actor_learning_rate=1e-4 \
--critic_learning_rate=3e-4 \
&

#Optionally, you can include cleanup commands here (e.g., after the job finishes)
#For example:
#rm some_temp_file.txt