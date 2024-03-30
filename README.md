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


## Major Features to be Added

Below is a list of the features that still need to be done. If you complete one of these, first, thank you! When complete, please add an invocation to the readme above demonstrating success, mark it complete below with your initials and date, and submit a pull request for your feature branch back to dev. I'll pull it to main once it's reconciled with other changes. I've tagged the locations
in the repo at which some of these features should be implemented with "Major Feature" in the relevant TODO.

- [ ] Environment save functionality.
- [ ] A decent environment rendering functionality.
- [ ] Flag-selectable wavelengths to model chromaticity.
- [ ] A SHWFS closed-loop AO model to populate the step reward.
- [ ] Parallelization of each wavelenth and sub-step of the environment step.
- [ ] Atmosphere phase screen caching to speed up the simulation step.
- [ ] Piston/tip/tilt secondaries for the ELF system.
- [ ] A tensegrity model to map control commands to low-order aberations; update action shape.
- [ ] Add low-order aberations directly to the segments in the optical system model.

## Known Issues

- There is a very sus correlation between the pupil-focal propogation speed and the atmosphere propogation speed. To see this, run `python deep-optics-gym/run_dasie_via_gym.py --report_time --record_env_state_info --num_atmosphere_layers=2` and inspect the logs to see these two speeds. They vary together, but there's no reason I can find why they should. This suggests that something is being recomputed, which may yield unphysical behavour. The best debug step is to save and then render the atmosphere history of an episode and see if it's continuous.
