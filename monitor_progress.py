#!/usr/bin/env python3
"""
Intelligent job manager for dataset collection with resource monitoring.
"""

import json
import time
import re
from pathlib import Path
import subprocess
import signal
import os
import threading
from typing import Dict, List, Optional


class ResourceMonitor:
    """Monitor system CPU and memory utilization using system commands."""
    
    def __init__(self, target_cpu_percent: float = 90.0, target_memory_percent: float = 85.0):
        self.target_cpu_percent = target_cpu_percent
        self.target_memory_percent = target_memory_percent
        self._lock = threading.Lock()
        self._last_readings = {'cpu': 0.0, 'memory': 0.0}
        
    def get_utilization(self) -> Dict[str, float]:
        """Get current CPU and memory utilization using system commands."""
        with self._lock:
            try:
                # Get CPU usage using top (macOS/Linux compatible)
                top_result = subprocess.run(['top', '-l', '1', '-n', '0'], 
                                          capture_output=True, text=True, timeout=5)
                cpu_percent = 0.0
                
                if top_result.returncode == 0:
                    # Parse CPU usage from top output
                    for line in top_result.stdout.split('\n'):
                        if 'CPU usage' in line:
                            # Example: "CPU usage: 12.34% user, 5.67% sys, 81.99% idle"
                            idle_match = re.search(r'(\d+\.?\d*)% idle', line)
                            if idle_match:
                                idle_percent = float(idle_match.group(1))
                                cpu_percent = 100.0 - idle_percent
                                break
                
                # Get memory usage using vm_stat (macOS) or free (Linux)
                memory_percent = 0.0
                available_memory_gb = 0.0
                
                # Try macOS vm_stat first
                vm_result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5)
                if vm_result.returncode == 0:
                    # Parse vm_stat output for macOS
                    pages_free = pages_inactive = pages_speculative = page_size = 0
                    total_pages = 0
                    
                    for line in vm_result.stdout.split('\n'):
                        if 'page size of' in line:
                            page_size_match = re.search(r'(\d+)', line)
                            if page_size_match:
                                page_size = int(page_size_match.group(1))
                        elif 'Pages free:' in line:
                            pages_free = int(re.search(r'(\d+)', line).group(1))
                        elif 'Pages inactive:' in line:
                            pages_inactive = int(re.search(r'(\d+)', line).group(1))
                        elif 'Pages speculative:' in line:
                            pages_speculative = int(re.search(r'(\d+)', line).group(1))
                    
                    if page_size > 0:
                        # Get total memory from system_profiler or sysctl
                        try:
                            sysctl_result = subprocess.run(['sysctl', 'hw.memsize'], 
                                                         capture_output=True, text=True, timeout=3)
                            if sysctl_result.returncode == 0:
                                mem_match = re.search(r'(\d+)', sysctl_result.stdout)
                                if mem_match:
                                    total_memory_bytes = int(mem_match.group(1))
                                    total_pages = total_memory_bytes // page_size
                        except:
                            total_pages = 0
                    
                    if total_pages > 0:
                        available_pages = pages_free + pages_inactive + pages_speculative
                        available_memory_gb = (available_pages * page_size) / (1024**3)
                        used_pages = total_pages - available_pages
                        memory_percent = (used_pages / total_pages) * 100.0
                
                else:
                    # Try Linux free command as fallback
                    free_result = subprocess.run(['free', '-m'], capture_output=True, text=True, timeout=3)
                    if free_result.returncode == 0:
                        for line in free_result.stdout.split('\n'):
                            if line.startswith('Mem:'):
                                parts = line.split()
                                if len(parts) >= 7:
                                    total_mb = float(parts[1])
                                    available_mb = float(parts[6])  # Available column
                                    memory_percent = ((total_mb - available_mb) / total_mb) * 100.0
                                    available_memory_gb = available_mb / 1024.0
                                    break
                
                self._last_readings = {
                    'cpu': cpu_percent,
                    'memory': memory_percent,
                    'available_memory_gb': max(available_memory_gb, 1.0)  # Ensure at least 1GB reported
                }
                
            except Exception as e:
                print(f"Warning: Could not get system utilization: {e}")
                # Use conservative defaults
                self._last_readings = {
                    'cpu': 80.0,  # Assume high usage if we can't measure
                    'memory': 75.0,
                    'available_memory_gb': 2.0
                }
            
            return self._last_readings.copy()
    
    def can_launch_job(self, estimated_cpu_per_job: float = 15.0, 
                       estimated_memory_per_job_gb: float = 2.0) -> bool:
        """
        Conservative check if we can launch another job without exceeding targets.
        
        Args:
            estimated_cpu_per_job: Estimated CPU usage per SA job (%)
            estimated_memory_per_job_gb: Estimated memory usage per SA job (GB)
        """
        current = self.get_utilization()
        
        # Conservative safety margins
        cpu_safety_margin = 10.0  # Keep 10% CPU buffer
        memory_safety_margin = 5.0  # Keep 5% memory buffer
        
        projected_cpu = current['cpu'] + estimated_cpu_per_job
        projected_memory_gb = (current['available_memory_gb'] - estimated_memory_per_job_gb)
        
        # Check if we would exceed targets with safety margins
        cpu_ok = projected_cpu <= (self.target_cpu_percent - cpu_safety_margin)
        memory_ok = projected_memory_gb > estimated_memory_per_job_gb  # At least one job's worth remaining
        
        return cpu_ok and memory_ok


class JobConfig:
    """Load and manage job configuration using JSON instead of YAML."""
    
    def __init__(self, config_file: str = "job_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        if not self.config_file.exists():
            # Create default config if it doesn't exist
            default_config = {
                "target_transitions": 10000,
                "check_interval_seconds": 30,
                "max_concurrent_jobs": 12,
                "resource_limits": {
                    "target_cpu_percent": 90.0,
                    "target_memory_percent": 85.0,
                    "estimated_cpu_per_job": 15.0,
                    "estimated_memory_per_job_gb": 2.0
                },
                "job_command": [
                    "poetry", "run", "python", "./optomech/sa.py",
                    "--eval_save_path=./rollouts/",
                    "--env-id", "optomech-v1",
                    "--object_type=single",
                    "--ao_interval_ms=5.0",
                    "--control_interval_ms=5.0", 
                    "--frame_interval_ms=5.0",
                    "--decision_interval_ms=5.0",
                    "--num_atmosphere_layers=0",
                    "--aperture_type=elf",
                    "--focal_plane_image_size_pixels=256",
                    "--observation_mode=image_only",
                    "--command_secondaries",
                    "--init_differential_motion", 
                    "--model_wind_diff_motion",
                    "--num_envs=1",
                    "--reward_function=align",
                    "--dataset",
                    "--max_episode_steps=10_000",
                    "--record_env_state_info",
                    "--write_env_state_info",
                    "--write_state_interval=1"
                ]
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            print(f"Created default config file: {self.config_file}")
            return default_config
        
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def get_command_for_dataset(self, dataset_id: int) -> List[str]:
        """Get command with dataset path substituted."""
        cmd = self.config['job_command'].copy()
        cmd.append(f'--dataset_save_path=./dataset_{dataset_id}/')
        return cmd


class JobManager:
    """Manage parallel SA jobs with resource monitoring."""
    
    def __init__(self, config: JobConfig, resource_monitor: ResourceMonitor):
        self.config = config
        self.resource_monitor = resource_monitor
        self.active_jobs = {}  # job_id -> (process, dataset_id)
        self.next_dataset_id = 1
        self._lock = threading.Lock()
    
    def get_active_job_count(self) -> int:
        """Get number of currently active jobs."""
        with self._lock:
            # Clean up finished jobs
            finished_jobs = []
            for job_id, (process, dataset_id) in self.active_jobs.items():
                if process.poll() is not None:  # Process finished
                    finished_jobs.append(job_id)
            
            for job_id in finished_jobs:
                del self.active_jobs[job_id]
            
            return len(self.active_jobs)
    
    def launch_job_if_possible(self) -> bool:
        """Launch a new job if resources allow. Returns True if job was launched."""
        with self._lock:
            # Check resource availability
            if not self.resource_monitor.can_launch_job(
                self.config.config['resource_limits']['estimated_cpu_per_job'],
                self.config.config['resource_limits']['estimated_memory_per_job_gb']
            ):
                return False
            
            # Check max concurrent limit
            if self.get_active_job_count() >= self.config.config['max_concurrent_jobs']:
                return False
            
            # Launch new job
            dataset_id = self.next_dataset_id
            self.next_dataset_id += 1
            
            cmd = self.config.get_command_for_dataset(dataset_id)
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True  # Prevent inheriting terminal signals
                )
                
                job_id = f"job_{dataset_id}"
                self.active_jobs[job_id] = (process, dataset_id)
                
                print(f"✓ Launched {job_id} (dataset_{dataset_id}) - PID: {process.pid}")
                return True
                
            except Exception as e:
                print(f"✗ Failed to launch job for dataset_{dataset_id}: {e}")
                return False
    
    def stop_all_jobs(self):
        """Stop all active jobs."""
        with self._lock:
            if not self.active_jobs:
                return
                
            print(f"Stopping {len(self.active_jobs)} active jobs...")
            
            # Send SIGTERM first
            for job_id, (process, dataset_id) in self.active_jobs.items():
                try:
                    process.terminate()
                    print(f"  Sent SIGTERM to {job_id} (PID: {process.pid})")
                except:
                    pass
            
            # Wait for graceful shutdown
            time.sleep(3)
            
            # Force kill if needed
            for job_id, (process, dataset_id) in self.active_jobs.items():
                try:
                    if process.poll() is None:  # Still running
                        process.kill()
                        print(f"  Force killed {job_id} (PID: {process.pid})")
                except:
                    pass
            
            self.active_jobs.clear()
    
    def get_status_summary(self) -> Dict:
        """Get summary of job status."""
        with self._lock:
            active_datasets = [dataset_id for _, (_, dataset_id) in self.active_jobs.items()]
            return {
                'active_jobs': len(self.active_jobs),
                'active_datasets': sorted(active_datasets),
                'next_dataset_id': self.next_dataset_id
            }
def count_transitions_in_dataset(dataset_dir):
    """Count total transitions across all dataset directories."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return 0
    
    total_transitions = 0
    
    # Check stats file first (faster)
    stats_file = dataset_path / "dataset_stats.json"
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                return stats.get('total_steps', 0)
        except:
            pass
    
    # Fallback: count from episode files
    episode_files = list(dataset_path.glob("episode_*.json"))
    for episode_file in episode_files:
        try:
            with open(episode_file, 'r') as f:
                episode_data = json.load(f)
                transitions = len(episode_data.get('observations', []))
                total_transitions += transitions
        except:
            continue
    
    return total_transitions


def count_all_transitions():
    """Count transitions across all dataset directories."""
    base_path = Path('.')
    total = 0
    
    # Count from individual dataset directories
    for i in range(1, 50):  # Check dataset_1 through dataset_49
        dataset_dir = base_path / f"dataset_{i}"
        if dataset_dir.exists():
            count = count_transitions_in_dataset(dataset_dir)
            if count > 0:
                print(f"  dataset_{i}: {count:,} transitions")
                total += count
    
    return total


class IntelligentJobManager:
    """Main class that orchestrates job management and monitoring."""
    
    def __init__(self, config_file: str = "job_config.json"):
        self.config = JobConfig(config_file)
        self.resource_monitor = ResourceMonitor(
            self.config.config['resource_limits']['target_cpu_percent'],
            self.config.config['resource_limits']['target_memory_percent']
        )
        self.job_manager = JobManager(self.config, self.resource_monitor)
        self.target_transitions = self.config.config['target_transitions']
        self.check_interval = self.config.config['check_interval_seconds']
    
    def run(self):
        """Main monitoring and job management loop."""
        print(f"🚀 Intelligent Job Manager Started")
        print(f"Target: {self.target_transitions:,} transitions")
        print(f"Config: {self.config.config_file}")
        print(f"CPU target: {self.resource_monitor.target_cpu_percent:.1f}%")
        print(f"Memory target: {self.resource_monitor.target_memory_percent:.1f}%")
        print("-" * 60)
        
        try:
            while True:
                # Check current progress
                total_transitions = count_all_transitions()
                
                # Check if target reached
                if total_transitions >= self.target_transitions:
                    print(f"\n🎉 TARGET REACHED! ({total_transitions:,} >= {self.target_transitions:,})")
                    print("Stopping all jobs...")
                    self.job_manager.stop_all_jobs()
                    
                    print("\n✅ Collection completed!")
                    print("You can now merge datasets with:")
                    print("poetry run python ./optomech/merge_datasets.py --target_dir ./merged_dataset/")
                    break
                
                # Get system status
                utilization = self.resource_monitor.get_utilization()
                job_status = self.job_manager.get_status_summary()
                
                # Try to launch new jobs if resources allow
                jobs_launched = 0
                while (self.job_manager.launch_job_if_possible() and 
                       jobs_launched < 3):  # Limit burst launches
                    jobs_launched += 1
                
                if jobs_launched > 0:
                    print(f"🚀 Launched {jobs_launched} new job(s)")
                
                # Display status
                progress = min(100, (total_transitions / self.target_transitions) * 100)
                print(f"\n📊 Progress: {total_transitions:,} / {self.target_transitions:,} ({progress:.1f}%)")
                print(f"💼 Active jobs: {job_status['active_jobs']} (datasets: {job_status['active_datasets']})")
                print(f"🖥️  CPU: {utilization['cpu']:.1f}% | Memory: {utilization['memory']:.1f}% | Available: {utilization['available_memory_gb']:.1f}GB")
                
                # Check if we should launch more jobs
                can_launch = self.resource_monitor.can_launch_job(
                    self.config.config['resource_limits']['estimated_cpu_per_job'],
                    self.config.config['resource_limits']['estimated_memory_per_job_gb']
                )
                
                if not can_launch and job_status['active_jobs'] == 0:
                    print("⚠️  No active jobs and cannot launch new ones - system may be overloaded")
                elif not can_launch:
                    print(f"⏸️  Resource limits reached - waiting with {job_status['active_jobs']} active jobs")
                elif job_status['active_jobs'] < self.config.config['max_concurrent_jobs']:
                    print(f"🔄 Resources available - will try launching more jobs next cycle")
                
                print(f"⏰ Next check in {self.check_interval} seconds...")
                print("-" * 60)
                
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            print("Stopping all active jobs...")
            self.job_manager.stop_all_jobs()
            print(f"Final count: {count_all_transitions():,} transitions")


def main():
    """Entry point."""
    import sys
    
    config_file = sys.argv[1] if len(sys.argv) > 1 else "job_config.json"
    
    try:
        manager = IntelligentJobManager(config_file)
        manager.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
