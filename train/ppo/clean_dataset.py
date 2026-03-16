#!/usr/bin/env python3
"""Find and delete corrupted .npz files in an autoencoder dataset."""

import argparse
import glob
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Remove corrupted batch_*.npz files")
    parser.add_argument("--dataset-dir", type=str, default="datasets/autoencoder",
                        help="Directory containing batch_*.npz files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report corrupted files, don't delete")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dataset_dir, "batch_*.npz")))
    print(f"Scanning {len(files)} files in {args.dataset_dir}...")

    bad = []
    for i, f in enumerate(files):
        try:
            data = np.load(f)
            # force read all arrays to catch truncation
            _ = data["images"]
            data.close()
        except Exception as e:
            bad.append((f, str(e)))
        if (i + 1) % 100 == 0:
            print(f"\r  [{i + 1}/{len(files)}] checked, {len(bad)} corrupted", end="", flush=True)

    print(f"\r  [{len(files)}/{len(files)}] checked, {len(bad)} corrupted")

    if not bad:
        print("No corrupted files found.")
        return

    for path, err in bad:
        if args.dry_run:
            print(f"  CORRUPTED: {os.path.basename(path)} — {err}")
        else:
            os.remove(path)
            print(f"  DELETED: {os.path.basename(path)} — {err}")

    if args.dry_run:
        print(f"\nDry run: {len(bad)} files would be deleted. Re-run without --dry-run to delete.")
    else:
        print(f"\nDeleted {len(bad)} corrupted files.")


if __name__ == "__main__":
    main()
