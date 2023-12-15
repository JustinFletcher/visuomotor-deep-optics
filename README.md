# Visuomotor Deep Optics

This repository implements an active deep optics approach, in which an agent model is trained to control an optomechanical system to mazimize episodic reward from an environment that, in turn, contains a task model. We train the

## Install

These build instructions are meant for a standard Linux install and assume that Anaconda is already installed. This README was built using Ubuntu 18.04 and `Anaconda3-5.2.0-Linux-x86_64`. Begin by creating a new environment to isolate this project.

```
# create `dl` conda environment
conda create -n dl python=3.10 pip
conda activate dl
```

We build on the dl-schema template by Matthew Phelps, found [here](https://github.com/phelps-matthew/dl-schema.git). A fork of the dl-schema repository is included here. The following instructions will install its dependencies.

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

If successful, this test should run the environment open-loop with live visualization.

## Model Training


