#!/usr/bin/env python3
"""
SML Job Manager - Manages parallel dataset generation jobs with progress tracking.

This script coordinates multiple dataset generation jobs to create large-scale SML datasets
efficiently using parallel processing and comprehensive progress tracking.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
from threading import Thread, Lock
from datetime import timedelta

# Add the parent directory to sys.path to import build_optomech_dataset
sys.path.append(str(Path(__file__).parent))
from build_optomech_dataset import main as build_dataset, Args

# Add parent directory to path for local imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import the build function directly
from build_optomech_dataset import build_dataset, Argshon3
"""
Simple Job Manager for Direct Supervised ML Dataset Generation

Just runs dataset generation jobs directly in parallel threads.
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
from datetime import datetime


@dataclass
class JobStatus:
    """Real-time status tracking for a job"""
    job_id: int
    start_time: float
    samples_completed: int = 0
    total_samples: int = 0
    is_running: bool = True
    is_complete: bool = False
    success: bool = False
    error_message: str = ""


class ProgressTracker:
    """Track progress of all running jobs with ETA calculations"""
    
    def __init__(self, total_jobs: int, total_samples: int):
        self.total_jobs = total_jobs
        self.total_samples = total_samples
        self.job_statuses: Dict[int, JobStatus] = {}
        self.overall_start_time = time.time()
        self.lock = threading.Lock()
    
    def add_job(self, job_id: int, samples_per_job: int):
        """Register a new job for tracking"""
        with self.lock:
            self.job_statuses[job_id] = JobStatus(
                job_id=job_id,
                start_time=time.time(),
                total_samples=samples_per_job
            )
    
    def update_job_progress(self, job_id: int, samples_completed: int):
        """Update progress for a specific job"""
        with self.lock:
            if job_id in self.job_statuses:
                self.job_statuses[job_id].samples_completed = samples_completed
    
    def complete_job(self, job_id: int, success: bool, error_message: str = ""):
        """Mark a job as completed"""
        with self.lock:
            if job_id in self.job_statuses:
                self.job_statuses[job_id].is_running = False
                self.job_statuses[job_id].is_complete = True
                self.job_statuses[job_id].success = success
                self.job_statuses[job_id].error_message = error_message
    
    def get_progress_summary(self) -> Dict:
        """Get current progress summary with ETA calculations"""
        with self.lock:
            current_time = time.time()
            
            # Calculate overall statistics
            completed_jobs = sum(1 for status in self.job_statuses.values() if status.is_complete)
            running_jobs = sum(1 for status in self.job_statuses.values() if status.is_running and not status.is_complete)
            successful_jobs = sum(1 for status in self.job_statuses.values() if status.success)
            failed_jobs = sum(1 for status in self.job_statuses.values() if status.is_complete and not status.success)
            
            # Calculate total samples generated
            total_samples_completed = sum(
                status.samples_completed if status.is_complete else status.samples_completed
                for status in self.job_statuses.values()
            )
            
            # Calculate ETAs
            overall_elapsed = current_time - self.overall_start_time
            overall_progress = total_samples_completed / self.total_samples if self.total_samples > 0 else 0
            
            overall_eta = None
            if overall_progress > 0.01:  # At least 1% progress
                estimated_total_time = overall_elapsed / overall_progress
                remaining_time = estimated_total_time - overall_elapsed
                overall_eta = remaining_time
            
            # Calculate per-job ETAs
            job_etas = {}
            for job_id, status in self.job_statuses.items():
                if status.is_running and not status.is_complete and status.samples_completed > 0:
                    job_elapsed = current_time - status.start_time
                    job_progress = status.samples_completed / status.total_samples
                    if job_progress > 0.01:
                        estimated_job_time = job_elapsed / job_progress
                        job_eta = estimated_job_time - job_elapsed
                        job_etas[job_id] = max(0, job_eta)
            
            return {
                'overall_elapsed': overall_elapsed,
                'overall_eta': overall_eta,
                'total_jobs': self.total_jobs,
                'completed_jobs': completed_jobs,
                'running_jobs': running_jobs,
                'successful_jobs': successful_jobs,
                'failed_jobs': failed_jobs,
                'total_samples_completed': total_samples_completed,
                'total_samples_target': self.total_samples,
                'overall_progress_pct': overall_progress * 100,
                'job_etas': job_etas,
                'job_statuses': dict(self.job_statuses)
            }
    
    def print_detailed_status(self):
        """Print detailed status of all jobs"""
        summary = self.get_progress_summary()
        
        print("\n" + "="*80)
        print("📊 DETAILED PROGRESS STATUS")
        print("="*80)
        
        # Overall progress
        elapsed_str = str(timedelta(seconds=int(summary['overall_elapsed'])))
        eta_str = "calculating..." if summary['overall_eta'] is None else str(timedelta(seconds=int(summary['overall_eta'])))
        
        print(f"🎯 Overall: {summary['total_samples_completed']:,}/{summary['total_samples_target']:,} samples "
              f"({summary['overall_progress_pct']:.1f}%)")
        print(f"⏱️  Elapsed: {elapsed_str} | ETA: {eta_str}")
        print(f"🏃 Jobs: {summary['running_jobs']} running, {summary['completed_jobs']} completed "
              f"({summary['successful_jobs']} success, {summary['failed_jobs']} failed)")
        
        print("\n📋 Individual Job Status:")
        print("-" * 80)
        
        for job_id, status in summary['job_statuses'].items():
            if status.is_complete:
                status_icon = "✅" if status.success else "❌"
                job_elapsed = str(timedelta(seconds=int(time.time() - status.start_time)))
                print(f"{status_icon} Job {job_id:2d}: {status.samples_completed:,}/{status.total_samples:,} samples "
                      f"({job_elapsed})")
                if not status.success and status.error_message:
                    print(f"    Error: {status.error_message}")
            elif status.is_running:
                progress_pct = (status.samples_completed / status.total_samples * 100) if status.total_samples > 0 else 0
                job_eta_str = "calculating..."
                if job_id in summary['job_etas']:
                    job_eta_str = str(timedelta(seconds=int(summary['job_etas'][job_id])))
                
                print(f"🏃 Job {job_id:2d}: {status.samples_completed:,}/{status.total_samples:,} samples "
                      f"({progress_pct:.1f}%) | ETA: {job_eta_str}")
        
        print("="*80)


@dataclass
class JobConfig:
    """Configuration for dataset generation jobs loaded from JSON"""
    total_samples: int
    samples_per_job: int
    max_workers: int
    timeout_minutes: int
    base_dataset_dir: str
    dataset_name: str
    env_id: str
    object_type: str
    aperture_type: str
    reward_function: str
    observation_mode: str
    focal_plane_image_size_pixels: int
    environment_flags: List[str]


def load_job_config(config_path: Path) -> JobConfig:
    """Load job configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    return JobConfig(**config_data)


def run_dataset_job(job_id: int, config: JobConfig, progress_tracker: ProgressTracker, results: List) -> None:
    """
    Run a single dataset generation job in a thread.
    
    Args:
        job_id: Unique identifier for this job
        config: Job configuration
        progress_tracker: Shared progress tracking object
        results: Shared results list
    """
    try:
        job_dataset_name = f"{config.dataset_name}_job_{job_id}"
        
        # Register job with progress tracker
        progress_tracker.add_job(job_id, config.samples_per_job)
        
        print(f"🚀 Starting job {job_id}: {config.samples_per_job} samples")
        start_time = time.time()
        
        # Create dataset subfolder path
        dataset_subfolder = Path(config.base_dataset_dir) / config.dataset_name
        dataset_subfolder.mkdir(parents=True, exist_ok=True)
        
        # Create Args object for this job
        args = Args()
        args.num_samples = config.samples_per_job
        args.dataset_save_path = str(dataset_subfolder)
        args.dataset_name = job_dataset_name
        args.env_id = config.env_id
        args.object_type = config.object_type
        args.aperture_type = config.aperture_type
        args.reward_function = config.reward_function
        args.observation_mode = config.observation_mode
        args.focal_plane_image_size_pixels = config.focal_plane_image_size_pixels
        args.silence = True  # Keep it quiet for background running
        
        # Apply environment flags
        for flag in config.environment_flags:
            if "=" in flag:
                key, value = flag.replace("--", "").split("=", 1)
                if hasattr(args, key):
                    # Convert value to appropriate type
                    attr_type = type(getattr(args, key))
                    if attr_type == bool:
                        setattr(args, key, value.lower() in ('true', '1', 'yes'))
                    elif attr_type == int:
                        setattr(args, key, int(value))
                    elif attr_type == float:
                        setattr(args, key, float(value))
                    else:
                        setattr(args, key, value)
            else:
                # Boolean flag
                key = flag.replace("--", "")
                if hasattr(args, key):
                    setattr(args, key, True)
        
        # Run the dataset generation directly
        build_dataset(args)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Job {job_id} completed successfully in {elapsed_time:.1f}s")
        
        progress_tracker.complete_job(job_id, True)
        results.append({
            'job_id': job_id,
            'success': True,
            'samples_generated': config.samples_per_job,
            'elapsed_time': elapsed_time,
            'dataset_name': job_dataset_name,
            'stdout': '',
            'stderr': ''
        })
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = str(e)
        print(f"❌ Job {job_id} failed: {error_msg}")
        
        progress_tracker.complete_job(job_id, False, error_msg)
        results.append({
            'job_id': job_id,
            'success': False,
            'samples_generated': 0,
            'elapsed_time': elapsed_time,
            'error': error_msg
        })


def main():
    """Main function to run the job manager"""
    parser = argparse.ArgumentParser(description="Job Manager for Direct SML Dataset Generation")
    parser.add_argument("--config", type=str, default="sml_job_config.json",
                       help="Path to job configuration JSON file")
    parser.add_argument("--dataset_name", type=str, default=None,
                       help="Override dataset name from config")
    parser.add_argument("--total_samples", type=int, default=None,
                       help="Override total samples from config")
    parser.add_argument("--background", action="store_true", default=False,
                       help="Run with minimal output for background execution")
    parser.add_argument("--watch", action="store_true", default=False,
                       help="Launch job watcher after starting jobs")
    
    args = parser.parse_args()
    
    # Load configuration from JSON file
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
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
    
    # Convert dataset_save_path to absolute path to avoid confusion
    # with relative paths when subprocess changes working directory
    dataset_save_path = Path(config.base_dataset_dir).resolve()
    config.base_dataset_dir = str(dataset_save_path)
    
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
    print(f"Dataset subfolder: {config.base_dataset_dir}/{config.dataset_name}")
    print(f"Environment: {config.env_id}")
    
    if args.background:
        print("🌙 Running in background mode - use job watcher for progress")
    else:
        print("👁️  Running with verbose progress tracking")
        
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
    
    start_time = time.time()
    results = []
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(num_jobs, config.total_samples)
    
    # Start a separate thread for periodic status updates (only if not in background mode)
    status_thread = None
    if not args.background:
        def status_updater():
            while True:
                time.sleep(30)  # Update every 30 seconds
                if len(results) >= num_jobs:
                    break
                progress_tracker.print_detailed_status()
        
        status_thread = threading.Thread(target=status_updater, daemon=True)
        status_thread.start()
    
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
                
                if not args.background:
                    print(f"📊 Progress: {completed_jobs}/{num_jobs} jobs ({progress:.1f}%) | "
                          f"Samples: {total_samples_generated}/{config.total_samples}")
                    
                    # Print detailed status every few completions
                    if completed_jobs % max(1, num_jobs // 10) == 0 or completed_jobs == num_jobs:
                        progress_tracker.print_detailed_status()
                else:
                    # Minimal output for background mode
                    if completed_jobs % max(1, num_jobs // 5) == 0 or completed_jobs == num_jobs:
                        print(f"📊 {completed_jobs}/{num_jobs} jobs ({progress:.1f}%) | "
                              f"{total_samples_generated:,}/{config.total_samples:,} samples")
                
            except Exception as e:
                print(f"💥 Job {job_id} generated an exception: {e}")
                progress_tracker.complete_job(job_id, False, str(e))
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
    
    # Launch job watcher if requested
    if args.watch:
        print(f"\n🔍 Launching job watcher...")
        watcher_script = Path(__file__).parent / "sml_job_watcher.py"
        if watcher_script.exists():
            subprocess.Popen([
                "poetry", "run", "python", str(watcher_script),
                "--dataset_dir", config.base_dataset_dir,
                "--detailed"
            ])
            print(f"Job watcher started - monitoring {config.base_dataset_dir}")
        else:
            print(f"⚠️ Job watcher not found at {watcher_script}")
    
    return 0 if len(failed_jobs) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
