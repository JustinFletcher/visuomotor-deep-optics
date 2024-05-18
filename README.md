# Visuomotor Deep Optics

This repository implements an active deep optics approach, in which an agent model is trained to control an optomechanical system to mazimize episodic reward from an environment that, in turn, contains a task model. 

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
```

The following command will render the prior episode. The render should show a speckle pattern that gradually converges to a sharp PSF.

```
```

Now that we know the AO is working, we can increase the scenario complexity until it fails, thereby establishing the task to be addressed by an autonomous agent. The following command will construct a scenario in which the differential motion of the sub-apertures is realistic, which prevents AO loop closure. Here, differential motion is a forward-controllable proxy for phase wrap - as differential motion increases, so does the maximium phase difference across a wavefront.

```
```

By sweeping over the parameter that controls the distribution of sub-aperture differential motion, we can plot the falloff in attainable Strehl as a function of mean differntial motion. The command below will run a script that performs this experiment.

```
```

By running this experiment, we see that AO correction begins to fail under conditions of relatively modest differential motion. Our task is to build an agent that expands range of correctable uncompensated differential motion. As baseline is increased, uncompensated differential motion will also increase. So, we can also formulate our task in terms of baseline: to successfully field a telescope of a desired baseline, we must first develop and agent that capable of correcting the uncompensated differential motion implied by that baseline. Baseline is, in turn, dictated by the target of observation. We can characterize targets in terms of the angular resolution and contrast needed to adequetly image the scene. In this work, we will focus only on angular resolution, as contrast requires simultaneous supression of the starlight. [Idea: we coulad actually use the max uncompenated differential motion as reward. It's readily availible in simulation, but could be characterized and then induced on a real system by introducing random commands during the training period for the embodied agent.]

## Major Features to be Added

Below is a list of the features that still need to be done. If you complete one of these, first, thank you! When complete, please add an invocation to the readme above demonstrating success, mark it complete below with your initials and date, and submit a pull request for your feature branch back to dev. I'll pull it to main once it's reconciled with other changes. I've tagged the locations
in the repo at which some of these features should be implemented with "Major Feature" in the relevant TODO.

- [ ] Environment save functionality.
- [X] A decent environment rendering functionality.
- [ ] Flag-selectable wavelengths to model chromaticity.
- [ ] A SHWFS closed-loop AO model to populate the step reward.
- [ ] Parallelization of each wavelenth and sub-step of the environment step.
- [ ] Atmosphere phase screen caching to speed up the simulation step.
- [ ] Piston/tip/tilt secondaries for the ELF system.
- [ ] A tensegrity model to map control commands to low-order aberations; update action shape.
- [ ] Add low-order aberations directly to the segments in the optical system model.

## Known Issues

- There is a very sus correlation between the pupil-focal propogation speed and the atmosphere propogation speed. To see this, run `python deep-optics-gym/run_dasie_via_gym.py --report_time --record_env_state_info --num_atmosphere_layers=2` and inspect the logs to see these two speeds. They vary together, but there's no reason I can find why they should. This suggests that something is being recomputed, which may yield unphysical behavour. The best debug step is to save and then render the atmosphere history of an episode and see if it's continuous.
