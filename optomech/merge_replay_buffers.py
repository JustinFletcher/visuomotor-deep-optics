

import os
import uuid
import shutil
import multiprocessing
from pathlib import Path
import tyro

from dataclasses import dataclass

from replay_buffers import ReplayBufferWithHiddenStates

def merge_replay_buffers(
        parent_dir: str,
        filter_disadvantageous: bool = False,
        explore_to_exploit_ratio: int = 2) -> ReplayBufferWithHiddenStates:
    """
    Load all replay buffers in subdirectories under `parent_dir` into a single buffer
    using the ReplayBufferWithHiddenStates.restore() method.
    """
    # Use a large enough capacity to hold all samples; oversize is safe.
    combined_rb = ReplayBufferWithHiddenStates(capacity=10_000_000)
    combined_rb.restore(
        parent_dir,
        filter_disadvantageous=filter_disadvantageous,
        explore_to_exploit_ratio=explore_to_exploit_ratio
    )

    total_loaded = len(combined_rb)
    print(f"Restored {total_loaded} samples from {parent_dir}")
    assert total_loaded > 0, "Replay buffer restore yielded no data."

    return combined_rb

@dataclass
class Args: 
    """
    Command-line arguments for the simulated annealing rollout script.
    """
    # 1. Rollout and Dataset Settings
    parent_dir: str = "rollouts" 
    output_dir: str = "merged_buffers"  # Directory to save the output
    chunk_size: int = 100  # Size of each chunk in the replay buffer
    filter_disadvantageous: bool = False  # Whether to filter out disadvantageous samples
    explore_to_exploit_ratio: int = 2  # Ratio of exploration to exploitation samples

if __name__ == "__main__":

    args = tyro.cli(Args)
    combined_rb = merge_replay_buffers(
        args.parent_dir,
        filter_disadvantageous=args.filter_disadvantageous,
        explore_to_exploit_ratio=args.explore_to_exploit_ratio)
    combined_path = os.path.join(args.parent_dir, "combined_replay_buffer.pt")
    combined_rb.save(combined_path,
                     chunk_size=args.chunk_size,)
    print(f"Saved combined replay buffer to {combined_path}")