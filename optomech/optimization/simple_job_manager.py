#!/usr/bin/env python3
"""
Simple job manager that launches jobs without complex resource monitoring.
"""

import json
import time
import subprocess
from pathlib import Path

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

def count_all_transitions():
    """Count transitions across all dataset directories."""
    base_path = Path('.')
    total = 0
    
    for i in range(1, 50):
        dataset_dir = base_path / f"dataset_{i}"
        if dataset_dir.exists():
            count = count_transitions_in_dataset(dataset_dir)
            if count > 0:
                print(f"  dataset_{i}: {count:,} transitions")
                total += count
    
    return total

class SimpleJobManager:
    """Simple job manager that launches jobs without resource monitoring."""
    
    def __init__(self):
        # Load config
        with open('job_config.json', 'r') as f:
            self.config = json.load(f)
        
        self.active_jobs = {}
        self.next_dataset_id = self.find_next_dataset_id()
        
    def find_next_dataset_id(self):
        """Find the next available dataset ID."""
        for i in range(1, 100):
            dataset_dir = Path(f"dataset_{i}")
            if not dataset_dir.exists():
                return i
        return 1
    
def launch_sa_job(dataset_id):
    """Launch a single SA job."""
    cmd = [
        'poetry', 'run', 'python', './optomech/sa.py',
        '--eval_save_path=./rollouts/',
        '--env-id', 'optomech-v1',
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
        f'--dataset_save_path=./dataset_{dataset_id}/',
        '--max_episode_steps=10_000',
        '--write_state_interval=1'
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Launched SA job for dataset_{dataset_id} (PID: {process.pid})")
        return process.pid
    except Exception as e:
        print(f"❌ Failed to launch job for dataset_{dataset_id}: {e}")
        return None
    
    def cleanup_finished_jobs(self):
        """Remove finished jobs from active list."""
        finished_jobs = []
        for job_id, (process, dataset_id) in self.active_jobs.items():
            if process.poll() is not None:
                finished_jobs.append(job_id)
        
        for job_id in finished_jobs:
            del self.active_jobs[job_id]
    
    def get_active_job_count(self):
        """Get number of active jobs."""
        self.cleanup_finished_jobs()
        return len(self.active_jobs)
    
    def stop_all_jobs(self):
        """Stop all active jobs."""
        if not self.active_jobs:
            return
            
        print(f"Stopping {len(self.active_jobs)} active jobs...")
        
        for job_id, (process, dataset_id) in self.active_jobs.items():
            try:
                process.terminate()
                print(f"  Sent SIGTERM to {job_id} (PID: {process.pid})")
            except:
                pass
        
        time.sleep(3)
        
        for job_id, (process, dataset_id) in self.active_jobs.items():
            try:
                if process.poll() is None:
                    process.kill()
                    print(f"  Force killed {job_id}")
            except:
                pass
        
        self.active_jobs.clear()

def main():
    """Main loop."""
    print("🚀 Simple Job Manager")
    print("Target: 1,000,000 transitions")
    print("Max concurrent jobs: 8")
    print("-" * 50)
    
    job_manager = SimpleJobManager()
    target = 1_000_000
    max_jobs = 8
    check_interval = 15
    
    try:
        while True:
            total_transitions = count_all_transitions()
            
            if total_transitions >= target:
                print(f"\n🎉 TARGET REACHED! ({total_transitions:,} >= {target:,})")
                job_manager.stop_all_jobs()
                print("\n✅ Collection completed!")
                break
            
            active_jobs = job_manager.get_active_job_count()
            progress = (total_transitions / target) * 100
            
            print(f"\n📊 Progress: {total_transitions:,} / {target:,} ({progress:.1f}%)")
            print(f"💼 Active jobs: {active_jobs}")
            
            # Launch jobs up to max limit
            jobs_to_launch = max_jobs - active_jobs
            if jobs_to_launch > 0:
                launched = 0
                for _ in range(min(jobs_to_launch, 3)):  # Launch max 3 at once
                    if job_manager.launch_job():
                        launched += 1
                    else:
                        break
                
                if launched > 0:
                    print(f"🚀 Launched {launched} new job(s)")
            else:
                print("✅ At maximum concurrent jobs")
            
            print(f"⏰ Next check in {check_interval} seconds...")
            print("-" * 50)
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
        job_manager.stop_all_jobs()
        print(f"Final count: {count_all_transitions():,} transitions")

if __name__ == "__main__":
    main()
