# multi_sa.py

import os
import uuid
import shutil
import multiprocessing
from pathlib import Path
import tyro

from dataclasses import dataclass


from sa import Args, sa
from replay_buffers import ReplayBufferWithHiddenStates

from merge_replay_buffers import merge_replay_buffers

def run_sa_in_subprocess(proc_id: int, args: Args, parent_dir: str):
    """
    Run SA in a subprocess, writing output to a unique subdirectory.
    """
    args_copy = Args(**vars(args))
    sub_dir = os.path.join(parent_dir, str(uuid.uuid4()))
    args_copy.eval_save_path = sub_dir
    sa(args_copy)


def multi_sa_main(args: Args, num_processes: int = 2):
    """
    Main function to run multiple SA rollouts in parallel.
    """

    os.makedirs(args.eval_save_path, exist_ok=True)
    parent_dir = args.eval_save_path

    print(f"[multi_sa] Writing rollout outputs to: {parent_dir}")


    procs = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=run_sa_in_subprocess, args=(i, args, parent_dir))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # combined_rb = merge_replay_buffers(parent_dir)
    # combined_path = os.path.join(parent_dir, "combined_replay_buffer.pt")
    # combined_rb.save(combined_path)
    # print(f"Saved combined replay buffer to {combined_path}")


if __name__ == "__main__":
    num_processes = 4

    # multi_args = tyro.cli(MultiArgs)

    sa_args = tyro.cli(Args)
    multi_sa_main(sa_args, num_processes)