#!/usr/bin/env python3
"""
Ultra-Simple SML Job Manager

Runs dataset generation until the target sample count is reached.
Counts samples by inspecting H5 files in the dataset directory.
No job splitting, no samples/job logic—just a loop until done.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path

try:
    import h5py
except ImportError:
    print("❌ h5py not available - cannot count samples in H5 files")
    sys.exit(1)


def count_samples_in_h5(file_path: Path) -> int:
    try:
        with h5py.File(file_path, 'r') as f:
            if 'observations' in f:
                return f['observations'].shape[0]
            elif 'perfect_actions' in f:
                return f['perfect_actions'].shape[0]
            else:
                return 0
    except Exception:
        return 0


def count_total_samples(dataset_dir: Path) -> int:
    total_samples = 0
    h5_files = list(dataset_dir.glob("*.h5")) + list(dataset_dir.glob("batch_*.h5"))
    for h5_file in h5_files:
        total_samples += count_samples_in_h5(h5_file)
    return total_samples


def run_dataset_generation(config: dict, dataset_dir: Path, job_id: int) -> bool:
    try:
        print(f"🚀 Starting dataset generation job {job_id}")
        cmd = [
            "poetry", "run", "python", 
            "optomech/supervised_ml/build_optomech_dataset.py",
            "--num_samples", "10000",  # Fixed chunk size
            "--dataset_save_path", str(dataset_dir),
            "--dataset_name", config["dataset_name"],
            "--write_frequency", str(config.get("write_frequency", 1000)),
            "--env_id", config.get("env_id", "optomech-v1"),
            "--object_type", config.get("object_type", "single"),
            "--aperture_type", config.get("aperture_type", "elf"),
            "--reward_function", config.get("reward_function", "align"),
            "--observation_mode", config.get("observation_mode", "image_only"),
            "--focal_plane_image_size_pixels", str(config.get("focal_plane_image_size_pixels", 256)),
            "--silence"
        ]
        for flag in config.get("environment_flags", []):
            cmd.append(flag)
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
        elapsed = time.time() - start_time
        if result.returncode == 0:
            print(f"✅ Job {job_id} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"❌ Job {job_id} failed after {elapsed:.1f}s")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Job {job_id} crashed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Ultra-Simple SML Job Manager")
    parser.add_argument("--config", type=str, default="sml_job_config.json",
                       help="Path to job configuration JSON file")
    parser.add_argument("--target_samples", type=int, required=True,
                       help="Target number of samples to generate")
    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="Directory where dataset should be stored")
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="Name of the dataset")
    parser.add_argument("--check_interval", type=int, default=30,
                       help="Seconds between progress checks")
    args = parser.parse_args()

    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load config file: {e}")
    config["dataset_name"] = args.dataset_name

    dataset_dir = Path(args.dataset_dir).resolve() / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("🎯 Ultra-Simple SML Job Manager")
    print("=" * 50)
    print(f"Target samples: {args.target_samples:,}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Check interval: {args.check_interval}s")
    print("=" * 50)

    job_counter = 0
    start_time = time.time()

    while True:
        current_samples = count_total_samples(dataset_dir)
        print(f"\n📊 Progress: {current_samples:,} / {args.target_samples:,} samples ({current_samples/args.target_samples*100:.1f}%)")
        if current_samples >= args.target_samples:
            total_time = time.time() - start_time
            print(f"\n🎉 TARGET REACHED!")
            print(f"✅ Generated {current_samples:,} samples (target: {args.target_samples:,})")
            print(f"⏱️  Total time: {total_time:.1f} seconds")
            if current_samples > 0:
                print(f"⚡ Rate: {current_samples/total_time:.2f} samples/sec")
            break
        remaining = args.target_samples - current_samples
        print(f"🚀 Need {remaining:,} more samples, starting generation job...")
        job_counter += 1
        success = run_dataset_generation(config, dataset_dir, job_counter)
        if not success:
            print(f"⚠️  Job failed, but continuing... (will retry in {args.check_interval}s)")
        print(f"⏳ Waiting {args.check_interval}s before next check...")
        time.sleep(args.check_interval)
    return 0

if __name__ == "__main__":
    sys.exit(main())
