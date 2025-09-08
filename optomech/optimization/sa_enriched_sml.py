# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import uuid
import json
import random
import time
import pickle
import shutil
from dataclasses import dataclass

from typing import Optional, Tuple

import math
from xml.parsers.expat import model

import gymnasium as gym
# from gymnasium.envs import box2d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchrl.data import ReplayBuffer
from tensordict import TensorDict

import torch.distributions as dist
from typing import Tuple
from torchrl.envs import GymWrapper, TransformedEnv
# from torchrl.envs.transforms import ToTensor
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchinfo import summary
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from anytree import Node, RenderTree

import matplotlib.pyplot as plt



from rollout import rollout_optomech_policy
from replay_buffers import *


class SmallCNNRegressor(nn.Module):
    """
    Lightweight CNN that maps images to a real-valued vector.
    Intended as a plug-in submodule; easy to save/load state_dict.
    """
    def __init__(self, in_channels: int = 3, out_dim: int = 4, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


class ImageRegressionDataset(Dataset):
    """
    CSV format:
      path,y0,y1,...,y{D-1}
    'path' may be absolute or relative to data_dir.
    """
    def __init__(self, csv_path: str, data_dir: str, transform=None, expected_dim: Optional[int] = None):
        self.root = data_dir
        self.transform = transform
        self.samples = []

        with open(csv_path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if i == 0 and ("path" in line or "," in line):
                    # Skip header if present
                    parts = line.split(",")
                    if parts[0].lower() == "path":
                        continue
                parts = line.split(",")
                rel = parts[0]
                path = rel if os.path.isabs(rel) else os.path.join(self.root, rel)
                targets = [float(v) for v in parts[1:]]
                self.samples.append((path, targets))

        if not self.samples:
            raise ValueError(f"No samples found in {csv_path}")
        self.target_dim = len(self.samples[0][1])
        if expected_dim is not None and expected_dim != self.target_dim:
            raise ValueError(f"Target dim in CSV={self.target_dim} != expected_dim={expected_dim}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = torch.tensor(y, dtype=torch.float32)
        return img, target


def save_checkpoint(state: dict, out_dir: str, epoch: int, is_best: bool):
    os.makedirs(out_dir, exist_ok=True)
    last_path = os.path.join(out_dir, "checkpoint_last.pt")
    epoch_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    torch.save(state, last_path)
    torch.save(state, epoch_path)
    if is_best:
        torch.save(state, os.path.join(out_dir, "checkpoint_best.pt"))


def try_auto_resume(out_dir: str) -> Optional[str]:
    last_path = os.path.join(out_dir, "checkpoint_last.pt")
    return last_path if os.path.isfile(last_path) else None


@torch.no_grad()
def validate(model, loader, device, loss_fn):
    model.eval()
    total, running = 0, 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(images)
        loss = loss_fn(preds, targets)
        bs = images.size(0)
        running += loss.item() * bs
        total += bs
    return running / max(1, total)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = np.random.randint(0, 10000)
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if using cuda, set the gpu"""
    gpu_list: int = 0
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    num_envs: int = 1
    """The number of environments to create."""
    async_env: bool = False
    """Whether to use an AsynchronousVectorEnv"""
    subproc_env: bool = False
    """Whether to use a SubprocVectorEnv"""
    model_save_interval: int = 10_000
    """The interval between saving model weights"""
    writer_interval: int = 1000
    """The interval between recording to tensorboard"""
    num_eval_rollouts: int = 1
    """The number of rollouts to perform for each evaluation."""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 100_000_000
    """total timesteps of the experiments"""
    # learning_rate: float = 3e-4
    actor_learning_rate: float = 1e-5
    """the learning rate of the optimizer"""
    critic_learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.004
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 16
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 256
    """timestep to start learning"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    decay_rate: float = 0.00001
    """Decay rate for noise decay"""
    action_scale: float = 1.0
    """The scale of the actors actions"""
    reward_scale: float = 1.0
    """The scale of the reward"""
    l2_reg: float = 0.0
    """The scale of the L2 regularization"""
    l1_reg: float = 0.0
    """The scale of the L1 regularization"""
    max_grad_norm: float = 1.0
    """The maximum gradient norm"""
    use_q_bias: bool = False
    """If toggled, compute q bias in the critic model."""
    normalize_returns: bool = False
    """If toggled, normalize the returns in the critic model."""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    replay_buffer_load_path: str = None
    """the path to load the replay buffer from"""

    # Actor model parameters
    actor_type: str = "vanilla"
    """Type of actor model to use."""
    actor_channel_scale: int = 16
    """The scale of the actor model channels."""
    actor_fc_scale: int = 64
    """The scale of the actor model fully connected layers."""
    low_dim_actor: bool = False
    """Whether the actor model is visual."""
    use_multiscale_head: bool = False
    """If toggled and using impala, it will have a multi-scale head"""

    # QNetwork model parameters
    critic_type: str = "vanilla"
    """Type of QNetwork model to use."""
    qnetwork_channel_scale: int = 16
    """The scale of the QNetwork model channels."""
    qnetwork_fc_scale: int = 64
    """The scale of the QNetwork model fully connected layers."""
    lstm_hidden_dim: int = 128
    """The scale of the QNetwork model fully connected layers."""
    low_dim_qnetwork: bool = False
    """Whether the qnetwork model is visual."""


    # Custom Algorthim Arguments
    """Which prelearning sample strategy to use (e.g., 'scales', 'normal')"""
    prelearning_sample: str = ""
    """How many steps to optimize the q function before actor training starts"""
    actor_training_delay: int = 10_000
    """How many steps to wait before populating the RB"""
    experience_sampling_delay: int = 10_000
    """Whether or not to use target smoothing"""
    target_smoothing: bool = False
    """How long the sequence length is for the LSTM"""
    tbptt_seq_len: int = 16

    # visual pendulum parameters
    # learning_rate: float = 3e-4
    # """the learning rate of the optimizer"""
    # buffer_size: int = int(1e6)
    # """the replay memory buffer size"""
    # gamma: float = 0.99
    # """the discount factor gamma"""
    # tau: float = 0.004
    # """target smoothing coefficient (default: 0.005)"""
    # batch_size: int = 64
    # """the batch size of sample from the reply memory"""
    # exploration_noise: float = 0.1
    # """the scale of exploration noise"""
    # learning_starts: int = 256
    # """timestep to start learning"""
    # policy_frequency: int = 4
    # """the frequency of training policy (delayed)"""
    # noise_clip: float = 0.5
    # """noise clip parameter of the Target Policy Smoothing Regularization"""

    # Environment specific arguments
    """Class for holding all arguments for the script."""
    gpu_list: str = "0"
    """The list of GPUs to use."""
    render: bool = False
    """Whether to render the environment."""
    report_time: bool = False
    """Whether to report time statistics."""
    action_type: str = "none"
    """The type of action to use."""
    object_type: str = "binary"
    """The type of object to use."""
    aperture_type: str = "elf"
    """The type of aperture to use."""
    max_episode_steps: int = 100


    """Toggle to enable discrete control."""
    discrete_control: bool = False
    """The number of discrete control steps."""
    discrete_control_steps: int = 128
    """Toggle to enable incremental control."""
    incremental_control: bool = False
    """Toggle to enable agent control of tensioners."""
    command_tensioners: bool = False
    """Toggle to enable agent control of secondaries."""
    command_secondaries: bool = False
    """Toggle to enable agent control of tip/tilt for large mirrors."""
    command_tip_tilt: bool = False
    """Toggle to enable agent control of dm."""
    command_dm: bool = False



    """ The type of observation to model 'image_only' or 'image_action'."""
    observation_mode: bool = "image_only"

    """The type of aperture to use."""
    ao_loop_active: bool = False
    """The maximum number of steps per episode."""
    num_episodes: int = 1
    """The number of episodes to run."""
    num_atmosphere_layers: int = 0
    """The number of atmosphere layers."""
    reward_threshold: float = 25.0
    """The reward threshold to reach."""
    num_steps: int = 16
    """The number of steps to take."""
    silence: bool = False
    """Whether to silence the output."""
    optomech_version: str = "test"
    """The version of optomech to use."""
    reward_function: str = "strehl"
    """The reward function to use."""
    render_frequency: int = 1
    """The frequency of rendering."""
    ao_interval_ms: float = 1.0
    """The interval between AO updates."""
    control_interval_ms: float = 2.0
    """The interval between control updates."""
    init_differential_motion: bool = False
    """Whether to initialize differential motion."""
    simulate_differential_motion: bool = False
    """Whether to simulate differential motion."""
    frame_interval_ms: float = 4.0
    """The interval between frames."""
    decision_interval_ms: float = 8.0
    """The interval between decisions."""
    focal_plane_image_size_pixels: int = 256
    """The size of the focal plane image in pixels."""
    render_dpi: float = 500.0
    """The DPI for rendering."""
    record_env_state_info: bool = False
    """Whether to record environment state information."""
    write_env_state_info: bool = False
    """Whether to write environment state information."""
    state_info_save_dir: str = "./tmp/"
    """The directory to save state information."""
    randomize_dm: bool = False
    """Whether to randomize the DM."""
    extended_object_image_file: str = ".\\resources\\sample_image.png"
    """The file for the extended object image."""
    extended_object_distance: str = None
    """The distance to the extended object."""
    extended_object_extent: str = None
    """The extent of the extended object."""
    observation_window_size: int = 2**1
    """The size of the observation window."""
    num_tensioners: int = 16
    """The number of tensioners."""
    model_wind_diff_motion: bool = False
    """Whether to model wind differential motion."""
    model_gravity_diff_motion: bool = False
    """Whether to model gravity differential motion."""
    model_temp_diff_motion: bool = False
    """Whether to model temperature differential motion."""

    # --- Supervised image->vector training knobs (kept optional to preserve compatibility) ---
    data_dir: str = "./data"
    """Root directory for images referenced in CSV."""
    train_csv: Optional[str] = None
    """CSV with columns: path,y0,y1,...  If None, no supervised training is run."""
    val_csv: Optional[str] = None
    """Optional CSV for validation; if not provided, val_split is used on train_csv."""
    out_dim: int = 4
    """Size of the regression target vector."""
    image_size: int = 128
    batch_size: int = 64
    num_epochs: int = 50
    """Number of epochs for supervised training (if train_csv provided)."""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    log_dir: str = "runs/tb"
    ckpt_dir: str = "runs/ckpts"
    save_every: int = 1
    auto_resume: bool = False
    resume_path: Optional[str] = None
    in_channels: int = 3
    grad_clip: float = 0.0
    val_split: float = 0.1
    normalize_mean: Optional[Tuple[float, float, float]] = None
    normalize_std: Optional[Tuple[float, float, float]] = None


if __name__ == "__main__":
    import sys
    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TensorBoard writer (kept compatible with original)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Persist args for reproducibility
    args_store_path = f"./runs/{run_name}/args.json"
    with open(args_store_path, "w") as f:
        json.dump(vars(args), f)

    # Device setup + multi-GPU data parallel (up to 8 GPUs)
    if torch.cuda.is_available() and args.cuda:
        print("Running with CUDA")
        torch.backends.cudnn.benchmark = True
        n_gpus = min(8, torch.cuda.device_count())
        device = torch.device("cuda:0")
        if n_gpus > 1:
            print(f"[INFO] Using DataParallel across {n_gpus} GPUs")
        torch.cuda.set_device(int(str(args.gpu_list).split(",")[0]))
    elif torch.backends.mps.is_available():
        print("Running with MPS")
        device = torch.device("mps")
    else:
        print("Running with CPU")
        device = torch.device("cpu")

    # If no CSV specified, just exit after logging (keeps compatibility with other uses of this script)
    if args.train_csv is None:
        print("[INFO] No --train_csv provided; nothing to train. Exiting.")
        writer.flush()
        writer.close()
        sys.exit(0)

    # -----------------------------
    # Datasets & loaders
    # -----------------------------
    tfm = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
    if args.normalize_mean is not None and args.normalize_std is not None:
        tfm.append(transforms.Normalize(mean=list(args.normalize_mean), std=list(args.normalize_std)))
    transform = transforms.Compose(tfm)

    if args.val_csv:
        train_ds = ImageRegressionDataset(args.train_csv, args.data_dir, transform, expected_dim=args.out_dim)
        val_ds = ImageRegressionDataset(args.val_csv, args.data_dir, transform, expected_dim=args.out_dim)
    else:
        full_ds = ImageRegressionDataset(args.train_csv, args.data_dir, transform, expected_dim=args.out_dim)
        n_total = len(full_ds)
        n_val = max(1, int(args.val_split * n_total))
        n_train = n_total - n_val
        gen = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # -----------------------------
    # Model / Optim / Sched
    # -----------------------------
    base_model = SmallCNNRegressor(in_channels=args.in_channels, out_dim=args.out_dim, width=32).to(device)
    if torch.cuda.is_available() and args.cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(base_model, device_ids=list(range(min(8, torch.cuda.device_count()))))
    else:
        model = base_model

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    # -----------------------------
    # Resume (optional)
    # -----------------------------
    start_epoch = 1
    global_step = 0
    best_val = float("inf")

    ckpt_dir = args.ckpt_dir if args.ckpt_dir else f"./runs/{run_name}/ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    resume_path = args.resume_path
    if args.auto_resume and resume_path is None:
        resume_path = try_auto_resume(ckpt_dir)

    if resume_path:
        print(f"[INFO] Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        # Handle DataParallel and non-DataParallel interchangeably
        model_state = ckpt["model"]
        try:
            model.load_state_dict(model_state)
        except RuntimeError:
            # If saved without DP but loading with DP (or vice versa), adapt keys
            from collections import OrderedDict
            new_state = OrderedDict()
            if next(iter(model_state.keys())).startswith("module."):
                # Loading a DP checkpoint into non-DP model
                for k, v in model_state.items():
                    new_state[k.replace("module.", "", 1)] = v
                model.load_state_dict(new_state, strict=True)
            else:
                # Loading a non-DP checkpoint into DP model
                new_state = OrderedDict((f"module.{k}", v) for k, v in model_state.items())
                model.load_state_dict(new_state, strict=True)

        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_val = ckpt.get("best_val", float("inf"))
        # Move optimizer tensors to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # Initial validation/log
    val_loss = validate(model, val_loader, device, loss_fn)
    writer.add_scalar("loss/val", val_loss, global_step)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
    best_val = min(best_val, val_loss)

    # -----------------------------
    # Train loop
    # -----------------------------
    print(f"[INFO] Starting supervised training for {args.num_epochs} epochs on {device}")
    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        epoch_loss_sum, epoch_items = 0.0, 0
        t0 = time.time()

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(images)
            loss = loss_fn(preds, targets)
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            bs = images.size(0)
            epoch_loss_sum += loss.item() * bs
            epoch_items += bs
            global_step += 1

            if args.writer_interval and (global_step % int(args.writer_interval) == 0):
                writer.add_scalar("loss/train_step", loss.item(), global_step)

        train_loss = epoch_loss_sum / max(1, epoch_items)
        val_loss = validate(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)

        writer.add_scalar("loss/train", train_loss, global_step)
        writer.add_scalar("loss/val", val_loss, global_step)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

        dt = time.time() - t0
        print(f"[E{epoch:03d}] train={train_loss:.5f}  val={val_loss:.5f}  lr={optimizer.param_groups[0]['lr']:.3e}  ({dt:.1f}s)")

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        if (epoch % args.save_every == 0) or is_best:
            # Save base_model weights (not wrapped with DataParallel) for easy reuse
            to_save = base_model.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            state = {
                "epoch": epoch,
                "global_step": global_step,
                "model": to_save,
                "optim": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "best_val": best_val,
                "config": vars(args),
            }
            save_checkpoint(state, ckpt_dir, epoch, is_best=is_best)

    writer.flush()
    writer.close()
    print("[INFO] Training complete.")
