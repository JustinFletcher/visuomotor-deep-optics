#!/usr/bin/env python3
"""
Simple Background Job Manager for SA Dataset Generation

Runs SA dataset generation jobs directly using threading.
Use the separate job watcher to monitor progress.
"""

import os
import sys
import time
import json
import argparse
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Add parent directory to path for local imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import the build function and Args class directly
from build_sa_dataset import build_sa_dataset, Args


@dataclass
class JobConfig:
    """Configuration for SA dataset generation jobs loaded from JSON"""
    total_samples: int
    samples_per_job: int
    max_workers: int
    timeout_minutes: int
    base_dataset_dir: str
    dataset_name: str
    write_frequency: int
    job_config_file: str
    init_temperature: float
    std_dev_patience: int
    sparsity_patience: int
    temperature_patience: int
    init_std_dev: float


def load_job_config(config_path: Path) -> JobConfig:
    """Load job configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    return JobConfig(**config_data)


def run_single_job(job_id: int, config: JobConfig, results: List):
    """Run a single SA dataset generation job in a thread"""
    try:
        # Calculate how many samples this job should generate
        samples_start = job_id * config.samples_per_job
        samples_remaining = config.total_samples - samples_start
        job_samples = min(config.samples_per_job, samples_remaining)
        
        print(f"🚀 Starting SA job {job_id}: {job_samples} samples (total progress: {samples_start + job_samples}/{config.total_samples})")
        start_time = time.time()
        
        # Create dataset subfolder path
        dataset_subfolder = Path(config.base_dataset_dir) / config.dataset_name
        dataset_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Create job-specific dataset name
        job_dataset_name = f"{config.dataset_name}_job_{job_id}"
        
        # Create Args object for this job
        args = Args()
        args.num_samples = job_samples
        args.dataset_save_path = str(dataset_subfolder)
        args.dataset_name = job_dataset_name
        args.write_frequency = config.write_frequency
        args.job_config_file = config.job_config_file
        args.init_temperature = config.init_temperature
        args.std_dev_patience = config.std_dev_patience
        args.sparsity_patience = config.sparsity_patience
        args.temperature_patience = config.temperature_patience
        args.init_std_dev = config.init_std_dev
        args.silence = True  # Keep it quiet for background running
        
        # Run the SA dataset generation
        build_sa_dataset(args)
        
        elapsed_time = time.time() - start_time
        print(f"✅ SA Job {job_id} completed successfully in {elapsed_time:.1f}s")
        
        results.append({
            'job_id': job_id,
            'success': True,
            'samples_generated': job_samples,
            'elapsed_time': elapsed_time,
            'dataset_name': job_dataset_name
        })
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = str(e)
        print(f"❌ SA Job {job_id} failed: {error_msg}")
        
        results.append({
            'job_id': job_id,
            'success': False,
            'samples_generated': 0,
            'elapsed_time': elapsed_time,
            'error': error_msg
        })


def main():
    """Main function to run the SA job manager"""
    parser = argparse.ArgumentParser(description="Simple Job Manager for SA Dataset Generation")
    parser.add_argument("--config", type=str, default="sa_job_config.json",
                       help="Path to job configuration JSON file")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="Override dataset name from config")
    parser.add_argument("--total_samples", type=int, default=None,
                       help="Override total samples from config")
    parser.add_argument("--dataset_dir", type=str, default=None,
                       help="Absolute path to top-level directory where dataset should be written")
    
    args = parser.parse_args()
    
    # Load configuration from JSON file
    config_path = args.config
    if not Path(config_path).exists():
        print(f"❌ Error: Configuration file not found at {config_path}")
        return 1
    
    try:
        config = load_job_config(config_path)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1
    
    # Apply command line overrides
    if args.dataset_name:
        config.dataset_name = args.dataset_name
    if args.total_samples:
        config.total_samples = args.total_samples
    if args.dataset_dir:
        config.base_dataset_dir = str(Path(args.dataset_dir).resolve())
    
    # Convert dataset_save_path to absolute path
    dataset_save_path = Path(config.base_dataset_dir).resolve()
    config.base_dataset_dir = str(dataset_save_path)
    
    # Calculate number of jobs needed
    num_jobs = (config.total_samples + config.samples_per_job - 1) // config.samples_per_job
    
    print("🎯 Simple SA Dataset Generation Job Manager")
    print("=" * 50)
    print(f"Total samples requested: {config.total_samples}")
    print(f"Samples per job: {config.samples_per_job}")
    print(f"Number of jobs: {num_jobs}")
    print(f"Max parallel workers: {config.max_workers}")
    print(f"Dataset name: {config.dataset_name}")
    print(f"Save path: {config.base_dataset_dir}")
    print(f"Dataset subfolder: {config.base_dataset_dir}/{config.dataset_name}")
    print(f"Job config file: {config.job_config_file}")
    print(f"SA Parameters: T={config.init_temperature}, std={config.init_std_dev}")
    print("💡 Use 'poetry run python sa_job_watcher.py' to monitor progress")
    print("=" * 50)
    
    # Create output directories
    base_dir = Path(config.base_dataset_dir)
    dataset_dir = base_dir / config.dataset_name
    
    base_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Save job configuration in the dataset subfolder
    config_file = dataset_dir / f"{config.dataset_name}_job_config.json"
    with open(config_file, 'w') as f:
        json.dump(vars(config), f, indent=2)
    print(f"📝 Job configuration saved to: {config_file}")
    
    # Results collection
    results = []
    threads = []
    start_time = time.time()
    
    # Start jobs in separate threads
    print(f"\n🚀 Starting {num_jobs} SA jobs...")
    for job_id in range(num_jobs):
        thread = threading.Thread(
            target=run_single_job, 
            args=(job_id, config, results)
        )
        thread.start()
        threads.append(thread)
        
        # Respect max_workers limit
        if len(threads) >= config.max_workers:
            for t in threads:
                t.join()
            threads = []
    
    # Wait for remaining threads
    for thread in threads:
        thread.join()
    
    # Calculate final statistics
    total_elapsed = time.time() - start_time
    successful_jobs = [r for r in results if r['success']]
    failed_jobs = [r for r in results if not r['success']]
    total_samples_generated = sum(r['samples_generated'] for r in successful_jobs)
    
    # Save results in the dataset subfolder
    results_file = dataset_dir / f"{config.dataset_name}_job_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': vars(config),
            'summary': {
                'total_jobs': num_jobs,
                'successful_jobs': len(successful_jobs),
                'failed_jobs': len(failed_jobs),
                'total_samples_requested': config.total_samples,
                'total_samples_generated': total_samples_generated,
                'total_elapsed_time': total_elapsed,
                'success_rate': len(successful_jobs) / num_jobs * 100
            },
            'job_results': results
        }, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 50)
    print("🏁 SA DATASET FINAL SUMMARY")
    print("=" * 50)
    print(f"✅ Successful jobs: {len(successful_jobs)}/{num_jobs}")
    print(f"❌ Failed jobs: {len(failed_jobs)}")
    print(f"📊 Total samples generated: {total_samples_generated:,}")
    print(f"🎯 Target samples: {config.total_samples:,}")
    print(f"📈 Completion rate: {total_samples_generated/config.total_samples*100:.1f}%")
    print(f"⏱️ Total time: {total_elapsed:.1f} seconds")
    if total_samples_generated > 0:
        print(f"⚡ Samples per second: {total_samples_generated/total_elapsed:.2f}")
    print(f"📋 Results saved to: {results_file}")
    
    if failed_jobs:
        print("\n❌ Failed job details:")
        for job in failed_jobs:
            print(f"  Job {job['job_id']}: {job.get('error', 'Unknown error')}")
    
    return 0 if len(failed_jobs) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
