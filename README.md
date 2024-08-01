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
- [ ] Add a reward and training render view.
## Known Issues


