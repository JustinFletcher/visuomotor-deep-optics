#!/usr/bin/env python3
"""
Simple job manager that launches SA jobs and monitors progress.
"""

import json
import time
import subprocess
from pathlib import Path
import os
import signal
import argparse

def count_transitions_in_dataset(dataset_dir):
    """Count transitions in a dataset directory using metadata."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return 0
    
    # Use stats file - much faster than parsing episode files
    stats_file = dataset_path / "dataset_stats.json"
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                return stats.get('total_steps', 0)
        except:
            return 0
    
    return 0

def count_specific_dataset_transitions(dataset_name, base_dataset_dir="./datasets"):
    """Count transitions in a specific dataset directory only."""
    dataset_path = Path(base_dataset_dir) / dataset_name
    if not dataset_path.exists():
        return 0
    
    # Use stats file - much faster than parsing episode files
    stats_file = dataset_path / "dataset_stats.json"
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                return stats.get('total_steps', 0)
        except:
            return 0
    
    return 0

def find_next_dataset_id(base_dataset_dir="./datasets", dataset_name="dataset"):
    """Find the next available dataset ID or return 1 for custom names."""
    base_path = Path(base_dataset_dir)
    
    # For custom dataset names, just check if it exists
    if dataset_name != "dataset":
        dataset_path = base_path / dataset_name
        return dataset_name if not dataset_path.exists() else f"{dataset_name}_2"
    
    # For numbered datasets, find the next number
    if not base_path.exists():
        return 1
    
    max_id = 0
    for i in range(1, 300):  # Check dataset_1 through dataset_299
        dataset_dir = base_path / f"dataset_{i}"
        if dataset_dir.exists():
            max_id = i
    
    return max_id + 1

def get_running_sa_jobs():
    """Get list of running SA job PIDs."""
    try:
        result = subprocess.run(['pgrep', '-f', 'sa.py'], capture_output=True, text=True)
        if result.returncode == 0:
            return [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]
        return []
    except:
        return []

def launch_sa_job(dataset_name, base_dataset_dir="./datasets", num_episodes=1, max_episode_steps=10000, env_id="optomech-v1"):
    """Launch a single SA job."""
    # Ensure base dataset directory exists
    base_path = Path(base_dataset_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create the specific dataset path
    dataset_path = base_path / dataset_name
    
    cmd = [
        'poetry', 'run', 'python', './optomech/optimization/sa.py',
        '--eval_save_path=./rollouts/',
        f'--env-id={env_id}',
        '--object_type=single',
        '--ao_interval_ms=5.0',
        '--control_interval_ms=5.0',
        '--frame_interval_ms=5.0',
        '--decision_interval_ms=5.0',
        '--num_atmosphere_layers=0',
        '--aperture_type=elf',
        '--focal_plane_image_size_pixels=256',
        '--observation_mode=image_only',
        '--command_secondaries',
        '--init_differential_motion',
        '--model_wind_diff_motion',
        '--num_envs=1',
        '--reward_function=align',
        '--dataset',
        f'--dataset_save_path={dataset_path}/',
        f'--max_episode_steps={max_episode_steps}',
        f'--num_episodes={num_episodes}',
        f'--dataset_name={dataset_name}',
        '--write_state_interval=1'
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Launched SA job for {dataset_path} (PID: {process.pid})")
        return process.pid
    except Exception as e:
        print(f"❌ Failed to launch job for {dataset_path}: {e}")
        return None

def stop_all_sa_jobs():
    """Stop all SA jobs."""
    pids = get_running_sa_jobs()
    if not pids:
        print("No SA jobs running")
        return
    
    print(f"Stopping {len(pids)} SA jobs...")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"  Sent SIGTERM to PID {pid}")
        except:
            pass
    
    # Wait a bit then force kill
    time.sleep(3)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Simple Job Manager for SA data collection")
    parser.add_argument('--target_transitions', type=int, default=1_000_000, 
                      help='Target number of total transitions to collect')
    parser.add_argument('--max_concurrent_jobs', type=int, default=8,
                      help='Maximum number of concurrent SA jobs')
    parser.add_argument('--check_interval', type=int, default=15,
                      help='Check interval in seconds')
    parser.add_argument('--base_dataset_dir', type=str, default='./datasets',
                      help='Base directory for datasets')
    parser.add_argument('--dataset_name', type=str, default='checkout_dataset',
                      help='Base name for dataset directories')
    parser.add_argument('--num_episodes', type=int, default=1,
                      help='Number of episodes per job')
    parser.add_argument('--max_episode_steps', type=int, default=10,
                      help='Maximum steps per episode')
    parser.add_argument('--env_id', type=str, default='optomech-v1',
                      help='Environment ID to use')
    
    args = parser.parse_args()
    
    print("🚀 Simple Job Manager")
    print(f"Target: {args.target_transitions:,} transitions")
    print(f"Max concurrent jobs: {args.max_concurrent_jobs}")
    print(f"Dataset directory: {args.base_dataset_dir}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Episodes per job: {args.num_episodes}")
    print(f"Steps per episode: {args.max_episode_steps}")
    print(f"Environment: {args.env_id}")
    print("-" * 50)
    
    next_dataset_name = find_next_dataset_id(args.base_dataset_dir, args.dataset_name)
    job_counter = 1  # For generating unique job names
    
    try:
        while True:
            # Count current progress in the specific dataset only
            total_transitions = count_specific_dataset_transitions(args.dataset_name, args.base_dataset_dir)
            
            # Check if target reached
            if total_transitions >= args.target_transitions:
                print(f"\n🎉 TARGET REACHED! ({total_transitions:,} transitions)")
                stop_all_sa_jobs()
                print("✅ All jobs stopped. Collection complete!")
                break
            
            # Show progress
            progress = (total_transitions / args.target_transitions) * 100
            dataset_path = Path(args.base_dataset_dir) / args.dataset_name
            print(f"\n📊 Progress: {total_transitions:,} / {args.target_transitions:,} ({progress:.1f}%)")
            print(f"  Dataset: {dataset_path} ({total_transitions:,} transitions)")
            
            # Count running jobs
            running_pids = get_running_sa_jobs()
            print(f"🔄 Active SA jobs: {len(running_pids)}")
            
            # Launch more jobs if needed
            jobs_to_launch = args.max_concurrent_jobs - len(running_pids)
            if jobs_to_launch > 0:
                print(f"🚀 Launching {jobs_to_launch} new jobs...")
                for i in range(jobs_to_launch):
                    # Use the dataset name directly for all episodes
                    launch_sa_job(
                        args.dataset_name, 
                        args.base_dataset_dir, 
                        args.num_episodes, 
                        args.max_episode_steps,
                        args.env_id
                    )
                    job_counter += 1
                    time.sleep(1)  # Small delay between launches
            
            print(f"⏰ Next check in {args.check_interval} seconds...")
            print("-" * 50)
            time.sleep(args.check_interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
        stop_all_sa_jobs()
        total_transitions = count_specific_dataset_transitions(args.dataset_name, args.base_dataset_dir)
        print(f"Final count: {total_transitions:,} transitions")

if __name__ == "__main__":
    main()
