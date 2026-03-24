# Visuomotor Deep Optics

Recurrent PPO for closed-loop alignment of segmented-aperture telescopes, trained on a GPU-accelerated HCIPy-based optical simulation (OptomechEnv).

## Installation

### Prerequisites

- Python 3.10.x (`>=3.10,<3.11`)
- [Poetry](https://python-poetry.org/)
- Git

### Quick start (local / laptop)

```bash
git clone https://github.com/JustinFletcher/visuomotor-deep-optics.git
cd visuomotor-deep-optics
git checkout dev
poetry install
```

### HPC from-source install

On minimal HPC nodes (e.g. MHPCC) you may need to build Python 3.10 and its C dependencies from source. Download all tarballs on a login node first — home directories are typically shared across compute nodes.

**1. Build C libraries to `$HOME/local`:**

libffi (required for `_ctypes`):
```bash
curl -LO https://github.com/libffi/libffi/releases/download/v3.4.6/libffi-3.4.6.tar.gz
tar -xzf libffi-3.4.6.tar.gz && cd libffi-3.4.6
./configure --prefix=$HOME/local && make && make install
cd ..
```

bzip2 (required for `_bz2`):
```bash
curl -LO https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz
tar -xzf bzip2-1.0.8.tar.gz && cd bzip2-1.0.8
make -f Makefile-libbz2_so
make install PREFIX=$HOME/local
cp libbz2.so* $HOME/local/lib/
cd ..
```

xz / liblzma (required for `_lzma`):
```bash
curl -LO https://github.com/tukaani-project/xz/releases/download/v5.4.6/xz-5.4.6.tar.gz
tar -xzf xz-5.4.6.tar.gz && cd xz-5.4.6
./configure --prefix=$HOME/local && make && make install
cd ..
```

**2. Build Python 3.10:**
```bash
curl -LO https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
tar -xzf Python-3.10.14.tgz && cd Python-3.10.14
LDFLAGS="-L$HOME/local/lib -L$HOME/local/lib64" \
CPPFLAGS="-I$HOME/local/include" \
PKG_CONFIG_PATH="$HOME/local/lib/pkgconfig:$HOME/local/lib64/pkgconfig" \
  ./configure --prefix=$HOME/local --enable-optimizations
make -j$(nproc) && make install
export PATH=$HOME/local/bin:$PATH
echo 'export PATH=$HOME/local/bin:$PATH' >> ~/.bashrc
cd ..
```

Verify:
```bash
$HOME/local/bin/python3.10 -c "import _ctypes, _bz2, _lzma; print('All OK')"
```

**3. Install Poetry and project:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH=$HOME/.local/bin:$PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
poetry env use $HOME/local/bin/python3.10
poetry install
```

Alternatively, if a module is available:
```bash
module load python/3.10
poetry install
```

## Project Structure

```
visuomotor-deep-optics/
├── optomech/optomech/             # OptomechEnv Gymnasium environments
│   ├── optomech.py                #   V1 — original HCIPy env
│   ├── optomech_v2.py             #   V2 — clean refactor, configurable kwargs
│   ├── optomech_v3.py             #   V3 — CPU-optimized
│   ├── optomech_v4.py             #   V4 — GPU-accelerated (MFT extraction)
│   └── optomech_v5.py             #   V5 — batched GPU VectorEnv (~4000 SPS on H100)
├── train/ppo/                     # Recurrent PPO training infrastructure
│   ├── train_ppo_optomech.py      #   Shared training core (CLI, PPO loop, eval)
│   ├── train_ppo_nanoelf_piston.py#   Piston-only experiment (2 DOF)
│   ├── train_ppo_nanoelf_ptt.py   #   Piston + tip/tilt (6 DOF)
│   ├── train_ppo_nanoelf_ptt_curriculum.py  # PTT with progressive TT ramp
│   ├── train_ppo_nanoelf_ptt_nocurr.py      # PTT, full TT from step 0
│   ├── train_ppo_nanoelf_ptt_nocurr_hbanneal.py  # Zero TT, HB annealing
│   ├── ppo_recurrent.py           #   LSTM-based PPO implementation
│   ├── ppo_models.py              #   ConvNet / ResNet18 encoder + LSTM + actor-critic
│   ├── rollout.py                 #   Evaluation rollouts with GIF output
│   ├── sweep_tiptilt.py           #   Tip/tilt performance sweep
│   ├── analyze_sweep.py           #   15-figure sweep analysis
│   ├── ptt_training_analysis.py   #   Training curve extraction from checkpoints
│   ├── train_ppo_autoencoder.py   #   Visual encoder pretraining
│   └── generate_autoencoder_dataset.py  # Dataset generation for autoencoder
├── figures/                       # Publication figure generation scripts
├── optomech/optimization/         # Simulated annealing optimization
├── optomech/imitation_learning/   # Behavioral cloning
├── optomech/supervised_ml/        # Supervised learning baselines
├── utils/                         # Sync, testing utilities
└── runs/                          # Training run outputs (checkpoints, TB logs)
```

## Environment Versions

| Version | Description | Speed |
|---------|-------------|-------|
| V3 | CPU-only, HCIPy propagation | ~200 SPS (laptop), ~20 SPS (HPC) |
| V4 | GPU-accelerated via extracted MFT matrices | ~650 SPS (H100) |
| V5 | Batched GPU VectorEnv (N envs share tensors) | ~4000 SPS @ 64 envs (H100) |

V4 is the default for local training. V5 is used with `--hpc` for cluster runs.

The key insight behind V4/V5: HCIPy's Fraunhofer propagator uses a Matrix Fourier Transform (MFT), not a plain FFT. The MFT is a separable matrix multiply `E_focal = norm * w_in * (M1 @ E_pupil @ M2)` with wavelength-dependent matrices. V4 pre-extracts these matrices at init and runs them on GPU via PyTorch.

## Training

Each experiment script embeds its full hyperparameter config (env kwargs + PPO params) so that runs are self-contained and reproducible. The shared core in `train_ppo_optomech.py` handles the training loop, evaluation, checkpointing, and TensorBoard logging.

### Local training (piston-only, quick validation)

```bash
poetry run python train/ppo/train_ppo_nanoelf_piston.py
```

This uses V4 with 8 envs. Piston-only (2 DOF) solves in ~1M steps.

### Local training (piston + tip/tilt)

```bash
# No curriculum — full TT error from step 0
poetry run python train/ppo/train_ppo_nanoelf_ptt_nocurr.py

# With curriculum — 0 TT for 100M steps, ramp to 2.0 arcsec over 200M
poetry run python train/ppo/train_ppo_nanoelf_ptt_curriculum.py
```

### CLI flags

All experiment scripts share these flags via `train_ppo_optomech.py`:

| Flag | Description |
|------|-------------|
| `--hpc` | Use HPC config (V5 batched env, 64 envs) |
| `--env-version v3\|v4\|v5` | Override environment version |
| `--no-eval` | Skip evaluation for max throughput |
| `--run-dir DIR` | Custom output directory |
| `--resume-from PATH` | Resume from full checkpoint (weights + optimizer + step) |
| `--init-from PATH` | Init weights only (restart training from step 0) |
| `--pretrained-encoder PATH` | Load autoencoder weights into CNN encoder |
| `--freeze-encoder` | Freeze pretrained encoder during PPO |
| `--learning-rate LR` | Override learning rate |
| `--num-envs N` | Override parallel env count |
| `--num-steps N` | Override rollout length per env |
| `--num-minibatches N` | Override minibatch count |
| `--model-save-interval N` | Checkpoint every N updates |
| `--action-penalty-weight W` | Override L1 action penalty |

### HPC / SLURM training

Write a SLURM batch script that activates the poetry environment and runs with `--hpc --no-eval`:

```bash
#!/bin/bash
#SBATCH --job-name=ptt-curriculum
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

cd /path/to/visuomotor-deep-optics
export PATH=$HOME/local/bin:$HOME/.local/bin:$PATH

poetry run python train/ppo/train_ppo_nanoelf_ptt_curriculum.py \
    --hpc --no-eval \
    --run-dir runs/ptt_curriculum_$(date +%s)
```

Submit:
```bash
sbatch train_ptt.slurm
```

Sync checkpoints from HPC to local:
```bash
poetry run python utils/sync_remote_runs.py
```

### Monitor training

```bash
poetry run tensorboard --logdir runs/
```

## Evaluation

### Single-checkpoint rollout

```bash
poetry run python train/ppo/rollout.py \
    --checkpoint runs/<run>/checkpoints/best.pt \
    --env-version v4 \
    --num-episodes 8
```

Produces per-episode GIFs and summary Strehl plots.

### Tip/tilt error sweep

Evaluate a trained agent across a range of initial tip/tilt disturbance magnitudes:

```bash
poetry run python train/ppo/sweep_tiptilt.py \
    --checkpoint runs/<run>/checkpoints/best.pt \
    --env-version v4 \
    --tt-min 0.0 --tt-max 3.0 --tt-steps 13 \
    --num-episodes 16
```

Generates violin plots, decile bar charts, cumulative Strehl curves, and per-TT-level GIFs. Raw per-step data saved to `sweep_results.json` for replotting:

```bash
poetry run python train/ppo/sweep_tiptilt.py --replot sweep_results.json
```

### Sweep analysis (deep dive)

```bash
poetry run python train/ppo/analyze_sweep.py sweep_results.json
```

Produces 15 figures analyzing the bimodal success/failure distribution, per-TT-level histograms, and failure rate progression.

## Experiment Variants

| Script | DOF | TT Error | Notes |
|--------|-----|----------|-------|
| `train_ppo_nanoelf_piston.py` | 2 (piston) | N/A | Baseline, solves ~1M steps |
| `train_ppo_nanoelf_ptt.py` | 6 (piston+TT) | 2.0 arcsec | Full difficulty |
| `train_ppo_nanoelf_ptt_nocurr.py` | 6 | 2.0 arcsec | Same, tuned hyperparams |
| `train_ppo_nanoelf_ptt_curriculum.py` | 6 | 0 -> 2.0 | Progressive TT ramp |
| `train_ppo_nanoelf_ptt_nocurr_hbanneal.py` | 6 | 0.0 | HB anneals, exploration-driven |

## Citation

```bibtex
@phdthesis{fletcher2026visuomotor,
  title={Visuomotor Deep Optics: Recurrent Reinforcement Learning for
         Segmented-Aperture Telescope Alignment},
  author={Fletcher, Justin},
  year={2026},
  school={University of Hawai'i as Manoa}
}
```
