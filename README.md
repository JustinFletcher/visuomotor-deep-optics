# Visuomotor Deep Optics 

This repository implements a model of a dynamic, partially controllable optomechanical system. 

[//]: # (This repository implements an active deep optics approach, in which an agent model is trained to control an optomechanical system to mazimize episodic reward from an environment that, in turn, contains a task model. )


## Install

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
python deep-optics-gym/run_dasie_via_gym.py
```

If successful, this test should run the environment open-loop. To see a live visualization, run:

```
python deep-optics-gym/run_dasie_via_gym.py --render --record_env_state_info

```

You should see the latest science frame and partial PSF as debug_output.png in the repo top-level directory. To see running times and add an atmosphere, invoke:

```
python deep-optics-gym/run_dasie_via_gym.py --report_time --num_atmosphere_layers=2
```

## Adding an agent.

There is an place for an agent in `run_dasie_via_gym.py`. Please don't directly add one to that script. Instead, copy the file and customize it to your needs. Remember to propogate any environment function signature changes back to the original `run_dasie_via_gym.py` before submitting a pull request. If you wanted to be a huge help, you could even write unit tests!

## Rendering an episode.

Rendering is deliberately decoupled from epsiode simulation. To render an environment, you must both record and write environment state information. These are flag-configurable variables so that users can optionally disable them to minimize step time during model training.

'''
python deep-optics-gym/run_dasie_via_gym.py --record_env_state_info --write_env_state_info --report_time --num_atmosphere_layers=2
'''

Once you have a saved episode, you can render it using the following command by substituting the directory of your choice for the UUID shown below. If mulitple episode save directories are present in the provided directory, the `render_history.py` will choose the most recently modified one by default, as a convenience to the developer. Note that this is a little buggy cross-OS, so YMMV.

'''
python deep-optics-gym/render_history.py --episode_info_dir=./tmp/a3161c4f-00ed-4fbe-841e-bc4f42c810f2 
'''

## Running an AO test.

You can exercise the functionality of the AO-based reward function independently from an agent as a debugging step. For calm atmospheres, simple scenes, and negligable differential sub-aperture motion, the AO loop should close without agent commands. 

The command below will initialize and run an episode in which a scene comprising a single non-resolved target is imaged without an atmosphere by a perfectly stable optomechanical system that is initialized with a small random deformation in a deformable mirror placed along the optical path.

```
python deep-optics-gym/run_dasie_via_gym.py --record_env_state_info --write_env_state_info --report_time --action_type=none --object_type=single --randomize_dm --ao_interval_ms=1.0 --control_interval_ms=2.0 --frame_interval_ms=4.0 --decision_interval_ms=8.0 --num_steps=4 --num_atmosphere_layers=1 --aperture_type=elf
```

The following command will render the prior episode. The render should show a speckle pattern that gradually converges to a sharp PSF, along with a view of the DM surface, which should flatten out.

```
python deep-optics-gym/render_history.py --episode_info_dir=/Users/fletcher/research/visuomotor-deep-optics/tmp/ --render_mode=dm
```

Now that we know the AO is working, we can increase the scenario complexity until it fails, thereby establishing the task to be addressed by an autonomous agent. The following command will construct a scenario in which the differential motion of the sub-apertures is realistic, which prevents AO loop closure. Here, differential motion is a forward-controllable proxy for phase wrap - as differential motion increases, so does the maximium phase difference across a wavefront.

```
python deep-optics-gym/run_dasie_via_gym.py --record_env_state_info --write_env_state_info --report_time --action_type=none --object_type=single --randomize_dm --ao_interval_ms=1.0 --control_interval_ms=2.0 --frame_interval_ms=4.0 --decision_interval_ms=8.0 --num_steps=4 --num_atmosphere_layers=1 --aperture_type=elf --simulate_differential_motion
```

In the example below, the post-correction surface of the segments are shown. Due to the random walk of the natural differential motion they vary across time and prevent AO loop closure.

```
python deep-optics-gym/render_history.py --episode_info_dir=/Users/fletcher/research/visuomotor-deep-optics/tmp/ --render_mode=diffmotion
```

By running this experiment, we see that AO correction begins to fail under conditions of relatively modest differential motion. This poses the problem to be solved: we must produce an agent that mitigates natural differential motion to improve wavefront stability and allow the AO loop to close. Since AO loop closure is our first objective, we introduce a reward function that is a proxy for that objective. The next command will run a few steps in an environment with this reward. Note that the agent here is a null agent, it takes no action.

```
python deep-optics-gym/run_dasie_via_gym.py --record_env_state_info --write_env_state_info --report_time --action_type=none --object_type=single --randomize_dm --ao_interval_ms=1.0 --control_interval_ms=2.0 --frame_interval_ms=4.0 --decision_interval_ms=8.0 --num_steps=4 --num_atmosphere_layers=1 --aperture_type=elf --simulate_differential_motion --reward_function=ao_rms_slope
```

Now we have all we need to visualize the problem from our agent's point of view. The next command renders the environment as the agent would see it. Each step is rendered individually, beginning with the focal plane observations that inform the agent. The agents actions are shown as tensors encoding control surface commands, and several environment and performance values are plotted. 

```
python deep-optics-gym/render_history.py --episode_info_dir=/Users/fletcher/research/visuomotor-deep-optics/tmp/ --render_mode=agent_view
```

Note that the agent here is a null agent - it takes no action. To visualize a randomly acting agent, rerun the experiment and switch `action_type=none` to `action_type=random`.

Our task is to build an agent that expands range of correctable uncompensated differential motion. As baseline is increased, uncompensated differential motion will also increase. So, we can also formulate our task in terms of baseline: to successfully field a telescope of a desired baseline, we must first develop an agent that is capable of correcting the uncompensated differential motion implied by that baseline. Baseline is, in turn, dictated by the target of observation. We can characterize targets in terms of the angular resolution and contrast needed to adequetly image the scene. In this work, we will focus only on angular resolution, as contrast requires simultaneous supression of the starlight. [Idea: we could use the max uncompesated differential motion as reward. It's readily availible in simulation, but could be characterized and then induced on a real system by introducing random commands during the training period for the embodied agent.]

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

### No Aberration

Now, we can move on to training our model. To begin, we simply ensure that the 
model training proceedure works for a trivial problem instance. In the command
below, we train a simple visuomotor agent via DDPG. The problem instance is has
no differential motion, atmosphere, or AO control. As such, the agent can
achieve the maximum reward by simply producing a zero-valued action. This, 
however, is non-trivial to achieve in practice, becuase of the high dynamic 
range of the control task. After running the command below, we find that 
training fails.

```
poetry run python ./deep-optics-gym/ddpg_continuous_action_optomech.py --env-id DASIE-v1 --object_type=single --ao_interval_ms=8.0 --control_interval_ms=8.0 --frame_interval_ms=8.0 --decision_interval_ms=8.0 --num_atmosphere_layers=0 --aperture_type=elf --focal_plane_image_size_pixels=256 --seed=88887 --tau=0.004 --exploration_noise=0.001 --reward_function negastrehl --learning_starts=25000 --policy_frequency 2 --batch_size=128 --max_episode_steps=200
```


Any randomly choose action is virtually certain to 
yield a poor reward. As such, exploration noise and pre-training sampling 
produce an impoverished replay buffer, lacking the diversity necessary to
enable effective training of the critic. This property is common to all high
dynamic range control tasks. Rather than fine-tuning the optimization or model
parameters to the scale of the task, which may not be known for any given task,
we introduce a scale-sampling experience pre-population routine, in which 
pre-training experience is sampled from a normal distribution. At the start of 
each pre-training episode, the standard deviation of the distribution is 
choosen with uniform probability from a set of scale values. 

```
poetry run python ./deep-optics-gym/ddpg_continuous_action_optomech.py --env-id DASIE-v1 --object_type=single --ao_interval_ms=8.0 --control_interval_ms=8.0 --frame_interval_ms=8.0 --decision_interval_ms=8.0 --num_atmosphere_layers=0 --aperture_type=elf --focal_plane_image_size_pixels=256 --seed=88887 --tau=0.004 --exploration_noise=0.001 --reward_function negastrehl --learning_starts=25000 --policy_frequency 2 --batch_size=128 --max_episode_steps=1000 --prelearning_sample=scales
```

Now, the training routine produces a stable, high-performance model. On a 
high-performance system, we can increase the number of environment replicas
and scale of the model to improve training performance.

```
poetry run python ./deep-optics-gym/ddpg_continuous_action_optomech.py --env-id DASIE-v1 --object_type=single --ao_interval_ms=8.0 --control_interval_ms=8.0 --frame_interval_ms=8.0 --decision_interval_ms=8.0 --num_atmosphere_layers=0 --aperture_type=elf --focal_plane_image_size_pixels=256 --seed=88887 --tau=0.004 --exploration_noise=0.001 --reward_function negastrehl --learning_starts=100000 --policy_frequency 4 --batch_size=512 --max_episode_steps=125 --prelearning_sample=scales --num_envs=64 --async_env --low_dim_actor --low_dim_qnetwork --actor_fc_scale=1024 --qnetwork_fc_scale=1024
```

Without aberration, the model can maximize performance by simply predicting an
action vector that is exactly zero. Thus, we can verify that the model is 
performing in a way that reflects a high average episodic reward without 
viewing the environment. However, once aberrations are introduced, we need to 
visually inspect on-policy rollouts to verify success. This can be done by 
periodically saving the model then rendering on-policy rollouts. To save the 
actor model, simply add the `save_model` flag to any run, like so:

```
poetry run python ./deep-optics-gym/ddpg_continuous_action_optomech.py --env-id DASIE-v1 --object_type=single --ao_interval_ms=8.0 --control_interval_ms=8.0 --frame_interval_ms=8.0 --decision_interval_ms=8.0 --num_atmosphere_layers=0 --aperture_type=elf --focal_plane_image_size_pixels=256 --seed=88887 --tau=0.004 --exploration_noise=0.001 --reward_function=negastrehl --learning_starts=25000 --policy_frequency 2 --batch_size=128 --max_episode_steps=1000 --prelearning_sample=scales --model_save_interval=10000 --save_model
```

To rollout the saved policy, use the following command and insert the path to 
the policy model that you wish to view.

```
poetry run python ./deep-optics-gym/rollout.py --model_path=/Users/fletcher/research/visuomotor-deep-optics/runs/DASIE-v1__ddpg_continuous_action_optomech__88887__1737236515/eval_ddpg_continuous_action_optomech_800/ddpg_continuous_action_optomech_800_policy.pt --env_vars_path=/Users/fletcher/research/visuomotor-deep-optics/runs/DASIE-v1__ddpg_continuous_action_optomech__88887__1737236515/args.json --write_env_state_info=True --record_env_state_info=True
```

Once this is complete, you have a rollout with all the environment data saved.
To view it, run:

```
poetry run python deep-optics-gym/render_history.py --render_mode=agent_view --episode_info_dir=/Users/fletcher/research/visuomotor-deep-optics/runs/DASIE-v1__ddpg_continuous_action_optomech__88887__1737236515/eval_ddpg_continuous_action_optomech_800/0bc2cd6e-4d1e-4b93-aa7a-c3557ace1945
```

Finally, it may be useful to generate static, off-policy datasets to evaluate 
some approaches. The following command will generate off-policy samples choosen
entirely at random. Either a path to a saved environment variable JSON or 
keywords must be provided. 

```
poetry run python ./deep-optics-gym/rollout.py --env_vars_path=/Users/fletcher/research/visuomotor-deep-optics/runs/DASIE-v1__ddpg_continuous_action_optomech__88887__1737236515/args.json --write_env_state_info=True --record_env_state_info=True --dataset  --eval_save_path=./tmp/dataset_example --num_episodes=4
```

To increase the size of the dataset, increase the 
number of episodes.  Setting `--write_env_state_info` to `False` will greatly
increase the sample rate, at the cost of removing detailed rollout logging for
visualization of training data.

```
poetry run python ./deep-optics-gym/rollout.py --env_vars_path=/Users/fletcher/research/visuomotor-deep-optics/runs/DASIE-v1__ddpg_continuous_action_optomech__88887__1737236515/args.json --write_env_state_info=False --record_env_state_info=True --dataset  --eval_save_path=./tmp/dataset_example --num_episodes=100
```

### Static Aberration

We have now successfully trained a policy that will performs well in the 
a trivial problem instance in which no aberation is present. To add a static
aberrations to an episode, set the environment flag `init_differential_motion` 
to `True`.

```
poetry run python ./deep-optics-gym/rollout.py --env_vars_path=/Users/fletcher/research/visuomotor-deep-optics/runs/DASIE-v1__ddpg_continuous_action_optomech__88887__1737236515/args.json --write_env_state_info=True --record_env_state_info=True --eval_save_path=./tmp/debug --num_episodes=1 --init_differential_motion=True --prelearning_sample=zeros  --simulate_differential_motion=False --model_wind_diff_motion=True --model_gravity_diff_motion=False --model_temp_diff_motion=False
```

We can render this rollout with the following command. 

```
poetry run python deep-optics-gym/render_history.py --render_mode=agent_view --episode_info_dir=./tmp/debug/
```

This render displays the expected instrumental scenario: there are static
aberrations present in the primary mirrors, which are constant over a single 
episode (becuase `simulate_differential_motion` is `False`) but vary between
episodes. We can also introduce AO to this scenario to show that AO apartially,
but not entirely, corrects for the aberrations.

```
poetry run python ./deep-optics-gym/rollout.py --env_vars_path=/Users/fletcher/research/visuomotor-deep-optics/runs/DASIE-v1__ddpg_continuous_action_optomech__88887__1737236515/args.json --write_env_state_info=True --record_env_state_info=True --dataset  --eval_save_path=./tmp/dataset_example --num_episodes=4 --model_wind_diff_motion=True --model_gravity_diff_motion=False --model_temp_diff_motion=False --prelearning_sample=zeros --init_differential_motion=True --ao_loop_active=True
```

To learn correction policies for the static abberation scenario, we may train a
model, as follows.

```
poetry run python ./deep-optics-gym/ddpg_continuous_action_optomech.py \
--env-id DASIE-v1 \
--object_type=single \
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
--model_save_interval=10000 \
--save_model \
--simulate_differential_motion=False \
--model_wind_diff_motion=True \
--model_gravity_diff_motion=False \
--model_temp_diff_motion=False \
--init_differential_motion=True \
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
poetry run python ./deep-optics-gym/rollout.py \
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
poetry run python deep-optics-gym/render_history.py \
--render_mode=agent_view \
--episode_info_dir=./datasets/wind_diff_motion_random_action/
```

If successful, you should see a randomly acting agent. Becuase the action scale
for the secondaries is much larger than the wind aberrations (under defualt 
settings) those aberations will not be visible. To see them, you can run the
same experiment with the rollout sample (i.e., action) policy set to `zeros`.


```
poetry run python ./deep-optics-gym/rollout.py \
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
poetry run python deep-optics-gym/render_history.py \
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
poetry run python ./deep-optics-gym/rollout.py \
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
poetry run python deep-optics-gym/render_history.py \
--render_mode=agent_view \
--episode_info_dir=./datasets/wind_diff_motion_scales_action/ \
```
### Small Dynamic Aberration


```
poetry run python ./deep-optics-gym/rollout.py --env_vars_path=/Users/fletcher/research/visuomotor-deep-optics/runs/DASIE-v1__ddpg_continuous_action_optomech__88887__1737236515/args.json --write_env_state_info=True --record_env_state_info=True --eval_save_path=./tmp/debug --num_episodes=1 --init_differential_motion=True --prelearning_sample=zeros  --simulate_differential_motion=True --model_wind_diff_motion=True --model_gravity_diff_motion=False --model_temp_diff_motion=False
```


### Realistic Dynamic Aberration


### Realistic Dynamic Aberration and AO


### Realistic Dynamic Aberraction and AO for Directed Contrast


### Policy Ensembles as an Observing Strategy


## Major Features to be Added

Below is a list of the features that still need to be done. If you complete one of these, first, thank you! When complete, please add an invocation to the readme above demonstrating success, mark it complete below with your initials and date, and submit a pull request for your feature branch back to dev. I'll pull it to main once it's reconciled with other changes. I've tagged the locations
in the repo at which some of these features should be implemented with "Major Feature" in the relevant TODO.

- [X] Environment save functionality.
- [X] A decent environment rendering functionality.
- [ ] Flag-selectable wavelengths to model chromaticity.
- [X] A SHWFS closed-loop AO model to populate the step reward.
- [ ] Parallelization of each wavelenth and sub-step of the environment step.
- [ ] Atmosphere phase screen caching to speed up the simulation step.
- [X] Piston/tip/tilt secondaries for the ELF system.
- [X] A mock tensegrity model to map control commands to low-order aberations; update action shape.
- [ ] A physically-informed tensegrity model to map control commands to low-order aberations.
- [X] Add differential motion directly to the segments in the optical system model.
- [ ] Add a direct exoplanet imaging astrophysical scene model to the object plane generator.
- [X] Add a reward and training render view.
- [X] Properly fork cleanrl and build a dedicated poetry config for optomech
- [ ] rename to optomech.
- [X] Refactor action space to be tuple of tuples of boxes.
- [X] Build action space vectorization and devectorization functions.
- [X] Migrate all action space vectorization into flag-selectable env feature.
- [ ] Replace tuple-based action space encoding with new Dict based encoding.

## Known Issues

None


