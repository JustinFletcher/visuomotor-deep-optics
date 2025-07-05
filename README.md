# optomech: a Gym Environment for Visuomotor Optomechanical Control Agent Training

This repository implements a model of a dynamic, partially controllable optomechanical system. 

[//]: # (This repository implements an active deep optics approach, in which an agent model is trained to control an optomechanical system to mazimize episodic reward from an environment that, in turn, contains a task model. )


<!-- ## Install

These build instructions are meant for a standard Linux install and assume that Anaconda is already installed. This README was built using Ubuntu 18.04 and `Anaconda3-5.2.0-Linux-x86_64`. Begin by creating a new environment to isolate this project.

```
# create `dl` conda environment
conda create -n dl python=3.10 pip
conda activate dl
```

We build on the dl-schema template by Matthew Phelps, found [here](https://github.com/phelps-matthew/dl-schema.git). The following instructions will install its dependencies.

```
# install torch and dependencies, assumes cuda version >= 11.0
pip install torch==2.0.0 torchvision==0.15.1
pip install mlflow==2.2.2 pyrallis==0.3.1 
pip install pandas tqdm pillow matplotlib 

# install hyperparameter search dependencies
pip install ray[tune] hyperopt

# install dl-schema repo
cd dl-schema
pip install -e .

# return to the top-level directory
cd ..
```

## Usage
* Download and extract the MNIST dataset
```python
cd dl_schema
python create_mnist_dataset.py
cd ..
```
* Train small CNN model (ResNet-18)
```python
python train.py
```
* View train configuration options
```python
python train.py --help
```
* Train from yaml configuration, with CLI override
```python
python train.py --config_path configs/resnet.yaml --lr 0.001 --gpus [7]
```
* Start mlflow ui to visualize results
```
# navgiate to dl_schema root directory containing `mlruns`
mlflow ui
# to set host and port
mlflow ui --host 0.0.0.0 --port 8080
```
* Serialize dataclass train config to yaml, outputting `configs/train_cfg.yaml`
```python
python cfg.py
```

* Resume a prior run using the run_name as a hook
```
TODO
```

## Hyperparameter Experiments
* Use ray tune to perform multi-gpu hyperparameter search
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python tune.py --exp_name hyper_search
```

```
# return to the top-level directory
cd dl-schema
```

# Building the Environment

## Gymnasium

Install Gymnasium and run the example script to verify installation. You should see a sequence of observation vectors printed to the console.
```
pip install gymnasium
python gym-example/gym_example.py
```

## HCIPy
Install HCIPy and its test dependendencies. Run the test suite to verify installation. 
```
pip install hcipy
```

## DeepOpticsGym-v0

You have now completed and verfied installation of all external dependencies. Next, validate build and install the DeepOpticsGym-v0 environment used for this work. 

```
python optomech/run_optomech_via_gym.py
```

If successful, this test should run the environment open-loop. To see a live visualization, run:

```
python optomech/run_optomech_via_gym.py --render --record_env_state_info

```

You should see the latest science frame and partial PSF as debug_output.png in the repo top-level directory. To see running times and add an atmosphere, invoke:

```
python optomech/run_optomech_via_gym.py --report_time --num_atmosphere_layers=2
```

## Adding an agent.

There is an place for an agent in `run_optomech_via_gym.py`. Please don't directly add one to that script. Instead, copy the file and customize it to your needs. Remember to propogate any environment function signature changes back to the original `run_optomech_via_gym.py` before submitting a pull request. If you wanted to be a huge help, you could even write unit tests!

## Rendering an episode.

Rendering is deliberately decoupled from epsiode simulation. To render an environment, you must both record and write environment state information. These are flag-configurable variables so that users can optionally disable them to minimize step time during model training.

'''
python optomech/run_optomech_via_gym.py --record_env_state_info --write_env_state_info --report_time --num_atmosphere_layers=2
'''

Once you have a saved episode, you can render it using the following command by substituting the directory of your choice for the UUID shown below. If mulitple episode save directories are present in the provided directory, the `render_history.py` will choose the most recently modified one by default, as a convenience to the developer. Note that this is a little buggy cross-OS, so YMMV.

'''
python optomech/render_history.py --episode_info_dir=./tmp/a3161c4f-00ed-4fbe-841e-bc4f42c810f2 
'''

## Running an AO test.

You can exercise the functionality of the AO-based reward function independently from an agent as a debugging step. For calm atmospheres, simple scenes, and negligable differential sub-aperture motion, the AO loop should close without agent commands. 

The command below will initialize and run an episode in which a scene comprising a single non-resolved target is imaged without an atmosphere by a perfectly stable optomechanical system that is initialized with a small random deformation in a deformable mirror placed along the optical path.

```
python optomech/run_optomech_via_gym.py --record_env_state_info --write_env_state_info --report_time --action_type=none --object_type=single --randomize_dm --ao_interval_ms=1.0 --control_interval_ms=2.0 --frame_interval_ms=4.0 --decision_interval_ms=8.0 --num_steps=4 --num_atmosphere_layers=1 --aperture_type=elf
```

The following command will render the prior episode. The render should show a speckle pattern that gradually converges to a sharp PSF, along with a view of the DM surface, which should flatten out.

```
python optomech/render_history.py --episode_info_dir=/Users/fletcher/research/visuomotor-deep-optics/tmp/ --render_mode=dm
```

Now that we know the AO is working, we can increase the scenario complexity until it fails, thereby establishing the task to be addressed by an autonomous agent. The following command will construct a scenario in which the differential motion of the sub-apertures is realistic, which prevents AO loop closure. Here, differential motion is a forward-controllable proxy for phase wrap - as differential motion increases, so does the maximium phase difference across a wavefront.

```
python optomech/run_optomech_via_gym.py --record_env_state_info --write_env_state_info --report_time --action_type=none --object_type=single --randomize_dm --ao_interval_ms=1.0 --control_interval_ms=2.0 --frame_interval_ms=4.0 --decision_interval_ms=8.0 --num_steps=4 --num_atmosphere_layers=1 --aperture_type=elf --simulate_differential_motion
```

In the example below, the post-correction surface of the segments are shown. Due to the random walk of the natural differential motion they vary across time and prevent AO loop closure.

```
python optomech/render_history.py --episode_info_dir=/Users/fletcher/research/visuomotor-deep-optics/tmp/ --render_mode=diffmotion
```

By running this experiment, we see that AO correction begins to fail under conditions of relatively modest differential motion. This poses the problem to be solved: we must produce an agent that mitigates natural differential motion to improve wavefront stability and allow the AO loop to close. Since AO loop closure is our first objective, we introduce a reward function that is a proxy for that objective. The next command will run a few steps in an environment with this reward. Note that the agent here is a null agent, it takes no action.

```
python optomech/run_optomech_via_gym.py --record_env_state_info --write_env_state_info --report_time --action_type=none --object_type=single --randomize_dm --ao_interval_ms=1.0 --control_interval_ms=2.0 --frame_interval_ms=4.0 --decision_interval_ms=8.0 --num_steps=4 --num_atmosphere_layers=1 --aperture_type=elf --simulate_differential_motion --reward_function=ao_rms_slope
```

Now we have all we need to visualize the problem from our agent's point of view. The next command renders the environment as the agent would see it. Each step is rendered individually, beginning with the focal plane observations that inform the agent. The agents actions are shown as tensors encoding control surface commands, and several environment and performance values are plotted. 

```
python optomech/render_history.py --episode_info_dir=/Users/fletcher/research/visuomotor-deep-optics/tmp/ --render_mode=agent_view
```

Note that the agent here is a null agent - it takes no action. To visualize a randomly acting agent, rerun the experiment and switch `action_type=none` to `action_type=random`.

Our task is to build an agent that expands range of correctable uncompensated differential motion. As baseline is increased, uncompensated differential motion will also increase. So, we can also formulate our task in terms of baseline: to successfully field a telescope of a desired baseline, we must first develop an agent that is capable of correcting the uncompensated differential motion implied by that baseline. Baseline is, in turn, dictated by the target of observation. We can characterize targets in terms of the angular resolution and contrast needed to adequetly image the scene. In this work, we will focus only on angular resolution, as contrast requires simultaneous supression of the starlight. [Idea: we could use the max uncompesated differential motion as reward. It's readily availible in simulation, but could be characterized and then induced on a real system by introducing random commands during the training period for the embodied agent.] -->

# Training An Optomechanical Control Agent

## Checkout

Before training an agent in our custom optomechanical environment, we should
first train RL agents in another environment within the same development
context. For this, we use ClearRL.


First, get a minimal, stable conda distribution from
https://docs.anaconda.com/miniconda/. The commands below will install 3.10.

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

Then activate conda.

```
source ~/miniconda3/bin/activate
```

And add conda to all of the shells in your home directory with this.

```
conda init --all
```

Now install Poetry, updating the python version to be whatever version
miniconda gave you, which should be 3.10 if you followed these instructions 
exactly.

```
curl -sSL https://install.python-poetry.org | python3.10 -
```

You may need to edit your shell configuration file (the install will tell you),
which you can do by identifying your shell with `echo $0` and editing the shell
configuration file for that shell. Instructions vary by shell, but are on
StackOverflow. For `zsh` (macOS default), this is at `~/.zshrc` which you can
create if it doesn't exist using `touch ~/.zshrc` and open with
`open ~/.zshrc`, after which you can paste
 `export PATH="/Users/yourusernamehere/.local/bin:$PATH"` and restart your
 shell. Then `poetry --version` should work, confirming the installation.

Now clone ClearRL with the following invocation.

```
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
poetry install
```

You may also need to ensure your version of pytorch is compatble with your
preferred accelerator. For instance, A100s require you to use a 2.0 release of
pytorch. If the following command to runs an experiment with CleanRl through
poetry fails, update the pyproject.toml file in cleanrl to require a newer
version (e.g. 2.5.0) of pytorch.

```
cd cleanrl
poetry run python cleanrl/ppo.py --seed 1 --env-id CartPole-v0 --total-timesteps 50000 --num_envs=4
```

This should run. To view the results, open a new terminal, change directories
to the log, and view it in tensorboard.

```
cd cleanrl/
poetry run tensorboard --logdir runs --bind_all
```

Now we can move to an environment that is more representative of our own:
Atari. to install the Atari enviroment, issue:

```
poetry install -E atari
```

With Atari installed, we can use SAC to train a visuomotor model by envoking
the following command.

```
poetry run python cleanrl/sac_atari.py --env-id BreakoutNoFrameskip-v4
```

Again, view training progress with TensorBoard to ensure training proceeds as
expected.

```
cd cleanrl/
tensorboard --logdir runs
```

At this point, we've now configured our environment to support both and 
optomechanical visuomotor control task and a deep reinforcement learning 
training for a visuomotor model. 

## Optomechanical Environment

Now that we are reasonably certain that we have a realiable DRL framework in 
which to work, we can introduce the optomech environment.

### No Aberration with Secondary Control

Now, we can move on to training our model. To begin, we simply ensure that the 
model training proceedure works for a trivial problem instance. In the command
below, we train a simple visuomotor agent via DDPG. The problem instance is has
no differential motion, atmosphere, or AO control. As such, the agent can
achieve the maximum reward by simply producing a zero-valued action. This, 
however, is non-trivial to achieve in practice, becuase of the high dynamic 
range of the control task. After running the command below, we find that 
training fails.

```
poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--env-id optomech-v1 \
--object_type single \
--command_secondaries \
--ao_interval_ms 1.0 \
--control_interval_ms 10.0 \
--frame_interval_ms 20.0 \
--decision_interval_ms 40.0 \
--num_atmosphere_layers 0 \
--aperture_type elf \
--focal_plane_image_size_pixels 256 \
--seed 88887 \
--tau 0.004 \
--exploration_noise 0.001 \
--reward_function negastrehl \
--learning_starts 25000 \
--policy_frequency 2 \
--batch_size 32 \
--max_episode_steps 1000
```

Any randomly choose action is very likely to  yield a poor reward. As such,
exploration noise and pre-training sampling produces an impoverished replay 
buffer, lacking the diversity necessary to
enable effective training of the critic. This property is common to all high
dynamic range control tasks. Rather than fine-tuning the optimization or model
parameters to the scale of the task, which may not be known for any given task,
we introduce a scale-sampling experience pre-population routine, in which 
pre-training experience is sampled from a normal distribution. At the start of 
each pre-training episode, the standard deviation of the distribution is 
choosen with uniform probability from a set of scale values. 

```
poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--env-id optomech-v1 \
--save_model \
--model_save_interval=10000 \
--object_type=single \
--command_secondaries \
--ao_interval_ms=8.0 \
--control_interval_ms=8.0 \
--frame_interval_ms=8.0 \
--decision_interval_ms=8.0 \
--num_atmosphere_layers=0 \
--aperture_type=elf \
--focal_plane_image_size_pixels=256 \
--seed=88887 \
--tau=0.004 \
--exploration_noise=0.001 \
--reward_function negastrehl \
--learning_starts=25000 \
--policy_frequency 2 \
--batch_size=128 \
--max_episode_steps=1000 \
--prelearning_sample=scales
```

Now, the training routine produces a stable, high-performance model. On a 
high-performance system, we can increase the number of environment replicas
and scale of the model to improve training performance.

```
poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--env-id optomech-v1 \
--save_model \
--model_save_interval=10000 \
--object_type=single \
--command_secondaries \
--ao_interval_ms=8.0 \
--control_interval_ms=8.0 \
--frame_interval_ms=8.0 \
--decision_interval_ms=8.0 \
--num_atmosphere_layers=0 \
--aperture_type=elf \
--focal_plane_image_size_pixels=256 \
--seed=88887 \
--tau=0.004 \
--exploration_noise=0.001 \
--reward_function negastrehl \
--learning_starts=100000 \
--policy_frequency 4 \
--batch_size=512 \
--max_episode_steps=125 \
--prelearning_sample=scales \
--num_envs=64 \
--async_env \
--low_dim_actor \
--low_dim_qnetwork \
--actor_fc_scale=1024 \
--qnetwork_fc_scale=1024
```

Without aberration, the model can maximize performance by simply predicting an
action vector that is exactly zero. Thus, we can verify that the model is 
performing in a way that reflects a high average episodic reward without 
viewing the environment. However, once aberrations are introduced, we need to 
visually inspect on-policy rollouts to verify success. This can be done by 
periodically saving the model then rendering on-policy rollouts. To save the 
actor model, simply add the `save_model` flag to any run, like so:

```
poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--env-id optomech-v1 \
--save_model \
--model_save_interval=10000 \
--command_secondaries \
--object_type=single \
--ao_interval_ms=8.0 \
--control_interval_ms=8.0 \
--frame_interval_ms=8.0 \
--decision_interval_ms=8.0 \
--num_atmosphere_layers=0 \
--aperture_type=elf \
--focal_plane_image_size_pixels=256 \
--seed=88887 \
--tau=0.004 \
--exploration_noise=0.001 \
--reward_function=negastrehl \
--learning_starts=25000 \
--policy_frequency 2 \
--batch_size=128 \
--max_episode_steps=1000 \
--prelearning_sample=scales \
--model_save_interval=10000 \
--save_model
```

To rollout the saved policy, use the following command and insert the path to 
the policy model that you wish to view.

```
poetry run python ./optomech/rollout.py \
--model_path=./runs/optomech-v1__ddpg_continuous_action_optomech__88887__1737932591/eval_ddpg_continuous_action_optomech_0/ddpg_continuous_action_optomech_0_policy.pt \
--env_vars_path=./runs/optomech-v1__ddpg_continuous_action_optomech__88887__1737932591/args.json \
--write_env_state_info=True \
--record_env_state_info=True
```

Once this is complete, you have a rollout with all the environment data saved.
To view it, run:

```
poetry run python optomech/render_history.py \
--render_mode=agent_view \
--episode_info_dir=./tmp/dataset_example/2914d3d5-e08b-476d-b1af-00372e9236e3
```

Finally, it may be useful to generate static, off-policy datasets to evaluate 
some approaches. The following command will generate off-policy samples choosen
entirely at random. Either a path to a saved environment variable JSON or 
keywords must be provided. Note that you are making the rollout off policy 
simply by omitting a `model_path`.

```
poetry run python ./optomech/rollout.py \
--env_vars_path=./runs/optomech-v1__ddpg_continuous_action_optomech__88887__1738187650/args.json \
--write_env_state_info=True \
--record_env_state_info=True \
--dataset \
--eval_save_path=./tmp/dataset_example \
--num_episodes=1
```

To increase the size of the dataset, increase the 
number of episodes.  Setting `--write_env_state_info` to `False` will greatly
increase the sample rate, at the cost of removing detailed rollout logging for
visualization of training data.

```
poetry run python ./optomech/rollout.py \
--env_vars_path=./runs/optomech-v1__ddpg_continuous_action_optomech__88887__1737932591/args.json  \
--write_env_state_info=False \
--record_env_state_info=True \
--dataset \
--eval_save_path=./tmp/dataset_example \
--num_episodes=100
```

### Static Aberration with Secondary Control

We have now successfully trained a policy that will performs well in the 
a trivial problem instance in which no aberation is present. To add a static
aberrations to an episode, set the environment flag `init_differential_motion` 
to `True`.

```
poetry run python ./optomech/rollout.py \
--env_vars_path=./runs/optomech-v1__ddpg_continuous_action_optomech__88887__1737932591/args.json \
--write_env_state_info=True \
--record_env_state_info=True \
--eval_save_path=./tmp/static/ \
--num_episodes=1 \
--init_differential_motion=True \
--prelearning_sample=zeros  \
--simulate_differential_motion=False \
--model_wind_diff_motion=True \
--model_gravity_diff_motion=False \
--model_temp_diff_motion=False
```

We can render this rollout with the following command. 

```
poetry run python optomech/render_history.py \
--render_mode=agent_view \
--episode_info_dir=./tmp/static/
```

This render displays the expected instrumental scenario: there are static
aberrations present in the primary mirrors, which are constant over a single 
episode (becuase `simulate_differential_motion` is `False`) but vary between
episodes. We can also introduce AO to this scenario to show that AO partially,
but not entirely, corrects for the aberrations.

```
poetry run python ./optomech/rollout.py \
--env_vars_path=./runs/optomech-v1__ddpg_continuous_action_optomech__88887__1738187650/args.json \
--write_env_state_info=True \
--record_env_state_info=True \
--dataset \
--eval_save_path=./tmp/dataset_example \
--num_episodes=4 \
--model_wind_diff_motion=True \
--model_gravity_diff_motion=False \
--model_temp_diff_motion=False \
--prelearning_sample=zeros \
--init_differential_motion=True \
--ao_loop_active=True
```

To learn correction policies for the static abberation scenario, we may train a
model, as follows.

```
poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--env-id optomech-v1 \
--object_type=single \
--command_secondaries \
--ao_interval_ms=1.0 \
--control_interval_ms=1.0 \
--frame_interval_ms=1.0 \
--decision_interval_ms=1.0 \
--num_atmosphere_layers=0 \
--aperture_type=elf \
--focal_plane_image_size_pixels=256 \
--seed=88887 \
--tau=0.004 \
--exploration_noise=0.001 \
--reward_function=negastrehl \
--learning_starts=25000 \
--policy_frequency 2 \
--batch_size=128 \
--max_episode_steps=1000 \
--prelearning_sample=scales \
--save_model \
--model_save_interval=10000 \
--init_differential_motion \
--model_wind_diff_motion \
--num_envs=64 \
--async_env \
--actor_fc_scale=256 \
--qnetwork_fc_scale=256 \
--qnetwork_channel_scale=256 \
--actor_channel_scale=256 



--simulate_differential_motion=False \
--model_gravity_diff_motion=False \
--model_temp_diff_motion=False \
```

At any point in training, we can select a saved model and perform a rollout on 
the same environment (though a different episode) to evaluate performance.

#### Static Wind Aberration Dataset Generation

To produce a dataset for the small aberration scenario, simply run the model 
training script, which will write out an environment metadata JSON file. Use 
this file to produce a dataset via an off-policy rollout. An example follows,
though you made need to create some empty directories if you wish to use the
command unchanged.

```
poetry run python ./optomech/rollout.py \
--env_vars_path=./datasets/wind_diff_motion_random_action/args_wind_init_motion.json \
--write_env_state_info=True \
--record_env_state_info=True \
--dataset \
--eval_save_path=./datasets/wind_diff_motion_random_action/ \
--num_episodes=1 \
```

We recommend visually verifying the desired behavor of the dataset before 
proceeding. This can be done via:

```
poetry run python optomech/render_history.py \
--render_mode=agent_view \
--episode_info_dir=./datasets/wind_diff_motion_random_action/
```

If successful, you should see a randomly acting agent. Becuase the action scale
for the secondaries is much larger than the wind aberrations (under defualt 
settings) those aberations will not be visible. To see them, you can run the
same experiment with the rollout sample (i.e., action) policy set to `zeros`.


```
poetry run python ./optomech/rollout.py \
--env_vars_path=./datasets/wind_diff_motion_random_action/args_wind_init_motion.json \
--write_env_state_info=True \
--record_env_state_info=True \
--dataset \
--eval_save_path=./datasets/wind_diff_motion_random_action/ \
--num_episodes=1 \
--prelearning_sample=zeros \
```

This, viewed as before, will reveal the small perturbations due to wind. 

```
poetry run python optomech/render_history.py \
--render_mode=agent_view \
--episode_info_dir=./datasets/wind_diff_motion_random_action/ \
```

Together, these two examples illustrate that the articluations necessary to
correct for the modeled wind-driven optomechanical aberrations are well 
within the range accessible to the secondaries. All that is needed is a policy
that performs the necessary corrections. 

In addition to the random action sampling strategy used above, it may also be
helpful to sample actions at multiple scales. This may be done as follows:


```
poetry run python ./optomech/rollout.py \
--env_vars_path=./datasets/wind_diff_motion_scales_action/args_wind_init_motion.json \
--write_env_state_info=True \
--record_env_state_info=True \
--dataset \
--eval_save_path=./datasets/wind_diff_motion_scales_action/ \
--num_episodes=1 \
--prelearning_sample=scales \
--scale_reset_interval=10 \
--
```

Visualizing this, we can see that every 10 steps a new scale is choose.

```
poetry run python optomech/render_history.py \
--render_mode=agent_view \
--episode_info_dir=./datasets/wind_diff_motion_scales_action/ \
```

### Wind and Gravity Aberrations with Secondary and Optomechanical Correction

```
poetry run python ./optomech/rollout.py \
--env_vars_path=/Users/fletcher/research/visuomotor-deep-optics/runs/optomech-v1__ddpg_continuous_action_optomech__88887__1737236515/args.json \
--write_env_state_info=True \
--record_env_state_info=True \
--eval_save_path=./tmp/debug \
--num_episodes=1 \
--init_differential_motion=True \
--prelearning_sample=zeros \
--simulate_differential_motion=True \
--model_wind_diff_motion=True \
--model_gravity_diff_motion=False \
--model_temp_diff_motion=False
```

### Wind, Gravity, and Atmosphere with Optomechanica, Secondary, and AO


### Wind, Gravity, and Atmosphere with Optomechanica, Secondary, and DM Corrections


### Policy Ensembles as an Observing Strategy

Here, we take up the challenge posed in landman2021selfoptimizing to construct
reward functions that operate directly on the focal plane, removing the need 
for a dedicated wavefront sensor. We propose a vortex stability function that 
rewards periods of sustained optical vorticies within a selected region of the 
focal plane. When a scene is configured such that a dim object is present
within that angular region the contrast relative to a central source will be 
enhanced relative to a perfectly corrected image of the same scene.


### On-sky Validation

We use the __ to collect on-sky, on-policy samples corresponding to a trained 
policy ensemble that includes only deformable mirror control. At present, no existing optomechanical system The policy used for sample generation is trained using an optomech environmeng that also

## Scratch space


poetry run python ./optomech/ddpg_continuous_action_optomech.py \
--env-id optomech-v1 \
--save_model \
--model_save_interval=1000 \
--object_type=single \
--ao_interval_ms=8.0 \
--control_interval_ms=8.0 \
--frame_interval_ms=8.0 \
--decision_interval_ms=8.0 \
--num_atmosphere_layers=0 \
--aperture_type=elf \
--focal_plane_image_size_pixels=256 \
--seed=88887 \
--tau=0.004 \
--exploration_noise=0.001 \
--reward_function=negastrehl \
--learning_starts=5000 \
--policy_frequency 2 \
--batch_size=128 \
--max_episode_steps=1000 \
--prelearning_sample=scales \
--model_save_interval=10000 \
--save_model

poetry run python ./optomech/rollout.py \
--env_vars_path=./runs/optomech-v1__ddpg_continuous_action_optomech__88887__1738438004/args.json \
--write_env_state_info=True \
--ao_loop_active=True \
--record_env_state_info=True \
--dataset \
--eval_save_path=./rollouts/ \
--num_episodes=1 \
--command_dm=True \
--num_atmosphere_layers=0 \


poetry run python ./optomech/rollout.py \
--env_vars_path=./runs/sample/args.json \
--write_env_state_info=True \
--record_env_state_info=True \
--dataset \
--eval_save_path=./rollouts/ \
--num_episodes=1 \
--num_atmosphere_layers=0 \
--focal_plane_image_size_pixels=256 \
--aperture_type=nanoelf


poetry run python optomech/render_history.py \
--render_mode=agent_view \
--episode_info_dir=./rollouts/

## NanoElf Experiment Area

Goal: Demonstrate coherence from a static aberration, then stabilize an optical
vortex.

/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL; while :; do clear; 
rsync -chavzP --delete fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/runs /Users/fletcher/research/visuomotor-deep-optics/runs; sleep 5; done


/usr/local/krb5/bin/pkinit  fletch@HPCMP.HPC.MIL;


rsync -rltv --progress --human-readable --delete --iconv=utf-8,utf-8-mac -e 'ssh -T -c aes128-gcm@openssh.com -o Compression=no -x' fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/runs /Users/fletcher/research/visuomotor-deep-optics/runs
; sleep 5; done


rsync -aHAXxv --numeric-ids --delete --progress -e "ssh -T -c arcfour -o Compression=no -x" fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/runs /Users/fletcher/research/visuomotor-deep-optics/runs
; sleep 5; done


poetry run python ./optomech/ddpg_lstm.py \
    --env-id optomech-v1 \
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
    --init_differential_motion \
    --model_wind_diff_motion \
    --save_model \
    --model_save_interval=1_000 \
    --num_envs=1 \
    --batch_size=32 \
    --tau=0.0001 \
    --policy_frequency=2 \
    --l2_reg=1e-3 \
    --reward_function=strehl \
    --reward_scale=1.0 \
    --max_grad_norm=5.0 \
    --buffer_size=1_000_000 \
    --learning_starts=10_000 \
    --actor_training_delay=5_000 \
    --max_episode_steps=100 \
    --actor_type=impala \
    --critic_type=impala \
    --actor_learning_rate=1e-5 \
    --critic_learning_rate=1e-4 \
    --qnetwork_channel_scale=16 \
    --actor_channel_scale=16 \
    --prelearning_sample=scales \
    --actor_fc_scale=256 \
    --qnetwork_fc_scale=256 \
    --action_scale=1.0 \
    --exploration_noise=0.1 \
    --decay_rate=0.998 \
    --target_smoothing \
    --writer_interval=1 \



rm -rf ./rollouts/* y

poetry run python ./optomech/sa.py \
    --eval_save_path=./rollouts/ \
    --env-id optomech-v1 \
    --object_type=single \
    --ao_interval_ms=5.0 \
    --control_interval_ms=5.0 \
    --frame_interval_ms=5.0 \
    --decision_interval_ms=5.0 \
    --num_atmosphere_layers=0 \
    --aperture_type=elf \
    --focal_plane_image_size_pixels=256 \
    --observation_mode=image_only \
    --command_secondaries \
    --init_differential_motion \
    --model_wind_diff_motion \
    --num_envs=1 \
    --reward_function=align \
    --max_episode_steps=300_000 \
    --record_env_state_info \
    --write_env_state_info \
    --write_state_interval=1 \

poetry run python optomech/render_history.py \
    --render_mode=agent_view \
    --episode_info_dir=./rollouts/ \
    --render_interval=1 \


Clear everything, build your replay buffers, merge and filter them, and train a model on them.

Next steps
- [ ] Migrate to Coral
- [ ] Manually tune multi_sa up to it's max size
- [ ] Launch a big multi_sa job
- [ ] Add direct controls for the number of steps in the training phases
- [ ] Launch a big training job.


rm -rf ./rollouts/*
rm -rf ./runs/*

poetry run python optomech/multi_sa.py \
    --eval_save_path=./rollouts/ \
    --env-id optomech-v1 \
    --object_type=single \
    --ao_interval_ms=5.0 \
    --control_interval_ms=5.0 \
    --frame_interval_ms=5.0 \
    --decision_interval_ms=5.0 \
    --num_atmosphere_layers=0 \
    --aperture_type=elf \
    --focal_plane_image_size_pixels=256 \
    --observation_mode=image_only \
    --command_secondaries \
    --init_differential_motion \
    --model_wind_diff_motion \
    --num_envs=1 \
    --reward_function=align \
    --max_episode_steps=10_000 \
    --record_env_state_info \
    --write_env_state_info \
    --write_state_interval=1 

poetry run python ./optomech/merge_replay_buffers.py \
    --parent_dir=rollouts \
    --chunk_size=100 \
    --filter_disadvantageous \
    --explore_to_exploit_ratio=10 \

poetry run python ./optomech/ddpg_lstm.py \
    --env-id optomech-v1 \
    --object_type=single \
    --ao_interval_ms=5.0 \
    --control_interval_ms=5.0 \
    --frame_interval_ms=5.0 \
    --decision_interval_ms=5.0 \
    --num_atmosphere_layers=0 \
    --aperture_type=elf \
    --focal_plane_image_size_pixels=256 \
    --observation_mode=image_only \
    --command_secondaries \
    --init_differential_motion \
    --model_wind_diff_motion \
    --num_envs=1 \
    --reward_function=align \
    --batch_size=16 \
    --tau=0.0001 \
    --policy_frequency=2 \
    --reward_scale=3.0 \
    --max_grad_norm=100_000 \
    --buffer_size=1_000_000 \
    --replay_buffer_load_path=./rollouts/combined_replay_buffer.pt \
    --learning_starts=1 \
    --experience_sampling_delay=2_000 \
    --actor_training_delay=1_000 \
    --max_episode_steps=1_00 \
    --actor_type=impala \
    --critic_type=impala \
    --actor_learning_rate=1e-3 \
    --critic_learning_rate=1e-3 \
    --qnetwork_channel_scale=16 \
    --actor_channel_scale=16 \
    --actor_fc_scale=256 \
    --qnetwork_fc_scale=256 \
    --action_scale=1.0 \
    --exploration_noise=0.1 \
    --decay_rate=0.998 \
    --writer_interval=10 \




poetry run python optomech/render_history.py \
    --render_mode=agent_view \
    --episode_info_dir=./rollouts/ \
    --render_interval=1 \





poetry run python ./optomech/rollout.py \
--model_path=./runs/optomech-v1__optomech-nanoelf-smallimpala-pf1-noise0.001-delay100k-50kstart__1174__1744245057/eval_optomech-nanoelf-smallimpala-pf1-noise0.001-delay100k-50kstart_2900000/optomech-nanoelf-smallimpala-pf1-noise0.001-delay100k-50kstart_2900000_policy.pt \
--env_vars_path=./runs/optomech-v1__optomech-nanoelf-smallimpala-pf1-noise0.001-delay100k-50kstart__1174__1744245057/args.json \
--write_env_state_info=True \
--record_env_state_info=True \
--eval_save_path=./rollouts/ \
--num_episodes=1 \
--exploration_noise=1e-4 \
--seed=746464


poetry run python optomech/render_history.py \
--render_mode=agent_view \
--episode_info_dir=./rollouts/


### Now add initial incoherence

The command below will create a training environment in which a small amount of
piston error is present thoughout the run, and must be corrected. This command 
has bee verified to yield a performant policy in about 400K iterations.



- [ ] Add feature to ensure that saturated commands do not result in alignment.
## Log
- 200k steps, 0.01 explore noise, 0.004 tau, 64 sizes, 3e-4 lr, policy 2, sort of learned something, I think.
```

# Remote rollout

## Start interactive job
srun --time=08:00:00 --mem=1764642 --account=MHPCC96650DAS --gres=gpu:A100-PCIe:8 --partition=standard --pty bash -i

## Clear rollouts
rm -rf ./rollouts/*

## Perform rollout
poetry run python ./optomech/rollout.py \
--model_path=./runs/optomech-v1__nanoelfplus-as1.0-noise1.0-decay0.995__2154__1743277832/eval_nanoelfplus-as1.0-noise1.0-decay0.995_3100000/nanoelfplus-as1.0-noise1.0-decay0.995_3100000_policy.pt \
--env_vars_path=./runs/optomech-v1__nanoelfplus-as1.0-noise1.0-decay0.995__2154__1743277832/args.json \
--write_env_state_info=True \
--record_env_state_info=True \
--eval_save_path=./rollouts/ \
--num_episodes=4 \
--exploration_noise=0.00001 \
--seed=746464

## Rollout zeros locally
poetry run python ./optomech/rollout.py \
--env_vars_path=./runs/optomech-v1__nanoelfplus-as1.0-noise1.0-decay0.995__2154__1743277832/args.json \
--write_env_state_info=True \
--record_env_state_info=True \
--eval_save_path=./rollouts/ \
--prelearning_sample=zeros \
--num_episodes=1 \
--exploration_noise=0.00001 \
--seed=746464

## Rollout random locally
poetry run python ./optomech/rollout.py \
--env_vars_path=./runs/optomech-v1__nanoelfplus-as1.0-noise1.0-decay0.995__2154__1743277832/args.json \
--write_env_state_info=True \
--record_env_state_info=True \
--eval_save_path=./rollouts/ \
--num_episodes=1 \
--exploration_noise=0.00001 \
--seed=746464


## Pull rollout back
/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL;
rsync -chavzP --stats fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/rollouts /Users/fletcher/research/visuomotor-deep-optics/rollouts/remote

## Render rollouts

```
poetry run python optomech/render_history.py \
--render_mode=agent_view \
--episode_info_dir=./rollouts/62df093b-ae11-4424-b556-a69cf79e09ef

poetry run python optomech/render_history.py \
--render_mode=agent_view \
--episode_info_dir=./rollouts/53d7c4a7-0daa-4e0f-807b-1923537692e0


poetry run python optomech/render_history.py \
--render_mode=agent_view \
--episode_info_dir=./



poetry run python optomech/render_history.py \
--render_mode=agent_view \
--episode_info_dir=./
```



## Major Features to be Added

Below is a list of the features that still need to be done. If you complete one of these, first, thank you! When complete, please add an invocation to the readme above demonstrating success, mark it complete below with your initials and date, and submit a pull request for your feature branch back to dev. I'll pull it to main once it's reconciled with other changes. I've tagged the locations
in the repo at which some of these features should be implemented with "Major Feature" in the relevant TODO.

- [X] Environment save functionality.
- [X] A decent environment rendering functionality.
- [ ] Flag-selectable wavelength and bandwidth to model chromaticity. (100nm bandwidth @ ~10 sample)
- [X] A SHWFS closed-loop AO model to populate the step reward.
- [ ] Parallelization of each wavelenth and sub-step of the environment step.
- [ ] Atmosphere phase screen caching to speed up the simulation step.
- [X] Piston/tip/tilt secondaries for the ELF system.
- [X] A mock tensegrity model to map control commands to low-order aberations; update action shape.
- [ ] A physically-informed tensegrity model to map control commands to low-order aberations. (low priority)
- [X] Add differential motion directly to the segments in the optical system model.
- [ ] Add a direct exoplanet imaging astrophysical scene model to the object plane generator.
- [X] Add a reward and training render view.
- [X] Properly fork cleanrl and build a dedicated poetry config for optomech
- [X]  rename to optomech.
- [X] Refactor action space to be tuple of tuples of boxes.
- [X] Build action space vectorization and devectorization functions.
- [X] Migrate all action space vectorization into flag-selectable env feature.
- [ ] Replace tuple-based action space encoding with new Dict based encoding.
- [ ] Refactor AO interval construct to simulation granularity.
- [ ] Add command lag... oh wait I did...
- [ ] Add M1 control, in addition to tensioner control.

## Known Issues

- High-level optical computations aren't producing the correct values.
- Differential motion initialization depends on commands.


