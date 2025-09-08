#!/usr/bin/env python3
"""
Job Manager for Direct Supervised ML Dataset Generation

This script manages parallel execution of the build_optomech_dataset.py script
to generate large-scale IID datasets for direct supervised machine learning.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class JobConfig:
    """Configuration for dataset generation jobs"""
    samples_per_job: int = 1000
    total_samples: int = 10000
    max_workers: int = 4
    base_dataset_dir: str = "./datasets"
    dataset_name: str = "sml_dataset"
    timeout_minutes: int = 60
    
    # Environment settings
    env_id: str = "optomech-v1"
    object_type: str = "single"
    aperture_type: str = "elf"
    reward_function: str = "strehl"
    observation_mode: str = "image_only"
    focal_plane_image_size_pixels: int = 256


def run_dataset_job(job_id: int, config: JobConfig, script_path: str) -> dict:
    """
    Run a single dataset generation job.
    
    Args:
        job_id: Unique identifier for this job
        config: Job configuration
        script_path: Path to the build_optomech_dataset.py script
        
    Returns:
        dict: Job results including success status and metadata
    """
    job_dataset_name = f"{config.dataset_name}_job_{job_id}"
    
    # Build command
    cmd = [
        "python", script_path,
        "--num_samples", str(config.samples_per_job),
        "--dataset_save_path", config.base_dataset_dir,
        "--dataset_name", job_dataset_name,
        "--env_id", config.env_id,
        "--object_type", config.object_type,
        "--aperture_type", config.aperture_type,
        "--reward_function", config.reward_function,
        "--observation_mode", config.observation_mode,
        "--focal_plane_image_size_pixels", str(config.focal_plane_image_size_pixels),
        "--silence"
    ]
    
    print(f"🚀 Starting job {job_id}: {config.samples_per_job} samples")
    
    start_time = time.time()
    
    try:
        # Run the dataset generation script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout_minutes * 60,
            cwd=os.path.dirname(script_path)
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Job {job_id} completed successfully in {elapsed_time:.1f}s")
            return {
                'job_id': job_id,
                'success': True,
                'samples_generated': config.samples_per_job,
                'elapsed_time': elapsed_time,
                'dataset_name': job_dataset_name,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"❌ Job {job_id} failed with return code {result.returncode}")
            return {
                'job_id': job_id,
                'success': False,
                'samples_generated': 0,
                'elapsed_time': elapsed_time,
                'error': f"Return code: {result.returncode}",
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"⏰ Job {job_id} timed out after {config.timeout_minutes} minutes")
        return {
            'job_id': job_id,
            'success': False,
            'samples_generated': 0,
            'elapsed_time': elapsed_time,
            'error': f"Timeout after {config.timeout_minutes} minutes"
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"💥 Job {job_id} crashed: {str(e)}")
        return {
            'job_id': job_id,
            'success': False,
            'samples_generated': 0,
            'elapsed_time': elapsed_time,
            'error': str(e)
        }


def main():
    """Main job manager function"""
    parser = argparse.ArgumentParser(description="Job Manager for Direct SML Dataset Generation")
    
    # Core job settings
    parser.add_argument("--total_samples", type=int, default=10000,
                       help="Total number of samples to generate")
    parser.add_argument("--samples_per_job", type=int, default=1000,
                       help="Number of samples per job")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of parallel jobs")
    parser.add_argument("--timeout_minutes", type=int, default=60,
                       help="Timeout per job in minutes")
    
    # Dataset settings
    parser.add_argument("--dataset_name", type=str, default="sml_dataset",
                       help="Base name for the dataset")
    parser.add_argument("--dataset_save_path", type=str, default="./datasets",
                       help="Path to save datasets")
    
    # Environment settings
    parser.add_argument("--env_id", type=str, default="optomech-v1",
                       help="Environment ID")
    parser.add_argument("--object_type", type=str, default="single",
                       help="Object type")
    parser.add_argument("--aperture_type", type=str, default="elf",
                       help="Aperture type")
    parser.add_argument("--reward_function", type=str, default="strehl",
                       help="Reward function")
    parser.add_argument("--observation_mode", type=str, default="image_only",
                       help="Observation mode")
    parser.add_argument("--focal_plane_image_size_pixels", type=int, default=256,
                       help="Image size in pixels")
    
    args = parser.parse_args()
    
    # Create job configuration
    config = JobConfig(
        samples_per_job=args.samples_per_job,
        total_samples=args.total_samples,
        max_workers=args.max_workers,
        base_dataset_dir=args.dataset_save_path,
        dataset_name=args.dataset_name,
        timeout_minutes=args.timeout_minutes,
        env_id=args.env_id,
        object_type=args.object_type,
        aperture_type=args.aperture_type,
        reward_function=args.reward_function,
        observation_mode=args.observation_mode,
        focal_plane_image_size_pixels=args.focal_plane_image_size_pixels
    )
    
    # Find the build_optomech_dataset.py script
    script_path = Path(__file__).parent / "build_optomech_dataset.py"
    if not script_path.exists():
        print(f"❌ Error: Could not find build_optomech_dataset.py at {script_path}")
        return 1
    
    # Calculate number of jobs needed
    num_jobs = (config.total_samples + config.samples_per_job - 1) // config.samples_per_job
    
    print("🎯 Direct SML Dataset Generation Job Manager")
    print("=" * 50)
    print(f"Total samples requested: {config.total_samples}")
    print(f"Samples per job: {config.samples_per_job}")
    print(f"Number of jobs: {num_jobs}")
    print(f"Max parallel workers: {config.max_workers}")
    print(f"Timeout per job: {config.timeout_minutes} minutes")
    print(f"Dataset name: {config.dataset_name}")
    print(f"Save path: {config.base_dataset_dir}")
    print(f"Environment: {config.env_id}")
    print("=" * 50)
    
    # Create output directory
    Path(config.base_dataset_dir).mkdir(parents=True, exist_ok=True)
    
    # Save job configuration
    config_file = Path(config.base_dataset_dir) / f"{config.dataset_name}_job_config.json"
    with open(config_file, 'w') as f:
        json.dump(vars(config), f, indent=2)
    print(f"📝 Job configuration saved to: {config_file}")
    
    start_time = time.time()
    results = []
    
    # Execute jobs in parallel
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(run_dataset_job, job_id, config, str(script_path)): job_id
            for job_id in range(num_jobs)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            job_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                # Progress update
                completed_jobs = len(results)
                total_samples_generated = sum(r['samples_generated'] for r in results)
                progress = completed_jobs / num_jobs * 100
                
                print(f"📊 Progress: {completed_jobs}/{num_jobs} jobs ({progress:.1f}%) | "
                      f"Samples: {total_samples_generated}/{config.total_samples}")
                
            except Exception as e:
                print(f"💥 Job {job_id} generated an exception: {e}")
                results.append({
                    'job_id': job_id,
                    'success': False,
                    'samples_generated': 0,
                    'elapsed_time': 0,
                    'error': str(e)
                })
    
    # Calculate final statistics
    total_elapsed = time.time() - start_time
    successful_jobs = [r for r in results if r['success']]
    failed_jobs = [r for r in results if not r['success']]
    total_samples_generated = sum(r['samples_generated'] for r in successful_jobs)
    
    # Save results
    results_file = Path(config.base_dataset_dir) / f"{config.dataset_name}_job_results.json"
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
    print("🏁 FINAL SUMMARY")
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
