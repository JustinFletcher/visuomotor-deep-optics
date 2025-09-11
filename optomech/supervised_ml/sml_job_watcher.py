#!/usr/bin/env python3
"""
Standalone Job Watcher for SML Dataset Generation

Monitors dataset generation progress by directly analyzing dataset files.
Works independently of job managers and config files.
"""

import os
import sys
import time
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class DatasetJobWatcher:
    """Watches SML dataset generation jobs by analyzing dataset files directly"""
    
    def __init__(self, dataset_base_dir: str = "./datasets", verbose: bool = False):
        self.dataset_base_dir = Path(dataset_base_dir).resolve()
        self.refresh_interval = 10  # seconds
        self.detailed_interval = 60  # seconds for detailed updates
        self.last_detailed_update = 0
        self.verbose = verbose
    
    def find_active_datasets(self) -> List[Dict]:
        """Find all dataset directories with SML data files"""
        active_datasets = []
        
        if not self.dataset_base_dir.exists():
            return active_datasets
        
        # Look for any directories containing dataset files
        for dataset_dir in self.dataset_base_dir.iterdir():
            if dataset_dir.is_dir():
                status = self.assess_dataset_status(dataset_dir)
                if status['has_dataset_files']:
                    dataset_info = {
                        'name': dataset_dir.name,
                        'path': str(dataset_dir),
                        'status': status
                    }
                    active_datasets.append(dataset_info)
        
        return active_datasets
    
    def assess_dataset_status(self, dataset_dir: Path) -> Dict:
        """Assess current status by analyzing dataset files directly"""
        status = {
            'total_samples_generated': 0,
            'total_batches': 0,
            'total_files': 0,
            'temp_files': 0,
            'has_dataset_files': False,
            'dataset_type': 'unknown',
            'start_time': None,
            'last_activity': None,
            'estimated_target': None,
            'batch_size': None,
            'file_details': [],
            'avg_bytes_per_sample': None
        }

        print(f"🔍 Scanning directory: {dataset_dir}")
        
        # Find all dataset files
        dataset_files = []
        temp_files = []

        # Look for JSON files (standard format) but skip config and metadata files
        try:
            json_files = list(dataset_dir.glob("*.json"))
            print(f"📁 Found {len(json_files)} JSON files")
            
            for file_path in json_files:
                filename = file_path.name
                
                # Skip temporary files (more comprehensive patterns)
                if (filename.startswith('.tmp_') or 
                    filename.endswith('.tmp') or
                    filename.startswith('tmp_') or
                    '.tmp.' in filename):
                    temp_files.append(file_path)
                    print(f"⏳ Temp file: {filename}")
                    continue
                    
                # Skip known config and metadata files that aren't actual dataset files
                if (filename.endswith('_job_config.json') or 
                    filename.endswith('_job_results.json') or
                    filename == 'dataset_stats.json' or
                    filename == 'config.json'):
                    print(f"⚙️  Config file: {filename}")
                    continue
                    
                # Only include files that look like episode/batch data
                if (filename.startswith('episode_') or 
                    filename.startswith('batch_') or
                    filename.startswith('data_')):
                    dataset_files.append(file_path)
                    print(f"📊 Dataset file: {filename}")
                    
        except Exception as e:
            print(f"⚠️  Error scanning JSON files: {e}")
                
        # Look for pickle files  
        try:
            pkl_files = list(dataset_dir.glob("*.pkl"))
            print(f"📦 Found {len(pkl_files)} PKL files")
            
            for file_path in pkl_files:
                filename = file_path.name
                if (filename.startswith('.tmp_') or 
                    filename.endswith('.tmp') or
                    filename.startswith('tmp_') or
                    '.tmp.' in filename):
                    temp_files.append(file_path)
                    print(f"⏳ Temp PKL file: {filename}")
                else:
                    dataset_files.append(file_path)
                    print(f"📦 Dataset PKL file: {filename}")
        except Exception as e:
            print(f"⚠️  Error scanning PKL files: {e}")
        
        status['total_files'] = len(dataset_files)
        status['temp_files'] = len(temp_files)
        status['has_dataset_files'] = len(dataset_files) > 0 or len(temp_files) > 0
        
        print(f"📈 Summary: {len(dataset_files)} dataset files, {len(temp_files)} temp files")
        
        if not status['has_dataset_files']:
            print("❌ No dataset files found")
            return status

        # Analyze each file to count samples - SKIP temp files to avoid race conditions
        total_samples = 0
        total_batches = 0
        dataset_types = set()
        batch_sizes = []
        avg_bytes_per_sample = None
        
        print(f"🔍 Analyzing {len(dataset_files)} dataset files for sample counts...")
        
        # Parse the first file completely to get baseline metrics
        first_file_analyzed = False
        for i, file_path in enumerate(dataset_files):
            try:
                if not first_file_analyzed:
                    # Parse first file completely to get sample size baseline
                    print(f"   📊 Analyzing first file completely: {file_path.name}")
                    samples, metadata = self.analyze_dataset_file(file_path, full_analysis=True)
                    
                    if samples > 0:
                        file_size = file_path.stat().st_size
                        avg_bytes_per_sample = file_size / samples
                        status['avg_bytes_per_sample'] = avg_bytes_per_sample
                        print(f"   📏 Established baseline: {avg_bytes_per_sample:.0f} bytes/sample")
                        first_file_analyzed = True
                else:
                    # For subsequent files, estimate based on file size
                    file_size = file_path.stat().st_size
                    if avg_bytes_per_sample and avg_bytes_per_sample > 0:
                        samples = max(1, int(file_size / avg_bytes_per_sample))
                        metadata = None  # Don't parse metadata for speed
                        print(f"   📄 {file_path.name}: ~{samples} samples (estimated from {file_size:,} bytes)")
                    else:
                        # Fallback to basic estimation
                        samples = max(1, file_size // 50000)
                        metadata = None
                        print(f"   📄 {file_path.name}: ~{samples} samples (fallback estimate)")
                
                total_samples += samples
                total_batches += 1
                
                if metadata:
                    if 'dataset_type' in metadata:
                        dataset_types.add(metadata['dataset_type'])
                    if 'batch_size' in metadata:
                        batch_sizes.append(metadata['batch_size'])
                    if 'total_samples_planned' in metadata and not status['estimated_target']:
                        status['estimated_target'] = metadata['total_samples_planned']
                
                status['file_details'].append({
                    'file': file_path.name,
                    'samples': samples,
                    'metadata': metadata,
                    'estimated': first_file_analyzed and i > 0  # Mark if this was estimated
                })
                
            except Exception as e:
                print(f"⚠️  Error analyzing file {file_path.name}: {e}")
                # Continue processing other files instead of crashing
                
        status['total_samples_generated'] = total_samples
        status['total_batches'] = total_batches
        
        if dataset_types:
            status['dataset_type'] = ', '.join(dataset_types)
        
        if batch_sizes:
            status['batch_size'] = int(sum(batch_sizes) / len(batch_sizes))  # Average
        
        # Check timing from file modification times (with safety checks)
        all_files = dataset_files + temp_files
        if all_files:
            timestamps = []
            for f in all_files:
                try:
                    timestamps.append(f.stat().st_mtime)
                except (FileNotFoundError, OSError):
                    # File disappeared during processing - skip it
                    continue
            
            if timestamps:  # Only proceed if we have valid timestamps
                status['start_time'] = min(timestamps)
                status['last_activity'] = max(timestamps)
        
        return status
    
    def analyze_dataset_file(self, file_path: Path, full_analysis: bool = False) -> tuple:
        """Analyze a single dataset file to extract sample count and metadata
        
        Args:
            file_path: Path to the dataset file
            full_analysis: If True, parse the file completely. If False, estimate from size.
        """
        samples = 0
        metadata = None
        
        try:
            # Extra safety check: ensure file still exists and is readable
            if not file_path.exists():
                print(f"⚠️  File disappeared during analysis: {file_path.name}")
                return 0, None
            
            # If not doing full analysis, estimate from file size
            if not full_analysis:
                file_size = file_path.stat().st_size
                # Use a reasonable estimate: ~50KB per 100 samples for JSON episode files
                samples = max(1, int(file_size / 500))  # Assume ~500 bytes per sample
                return samples, None
                
            # Full analysis: parse the file content
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                print(f"⚠️  Unknown file type: {file_path.suffix}")
                return 0, None
            
            # Extract metadata if present
            if isinstance(data, dict) and 'metadata' in data:
                metadata = data['metadata']
                episode_data = data.get('episode_data', data)
            else:
                episode_data = data
            
            # Count samples based on data structure
            if isinstance(episode_data, dict):
                # Look for observation arrays
                if 'observations' in episode_data:
                    samples = len(episode_data['observations'])
                elif 'perfect_actions' in episode_data:
                    samples = len(episode_data['perfect_actions'])
                elif 'next_observations' in episode_data:
                    samples = len(episode_data['next_observations'])
                else:
                    # Try to find any list-like structure
                    for key, value in episode_data.items():
                        if isinstance(value, list) and len(value) > 0:
                            samples = len(value)
                            break
            elif isinstance(episode_data, list):
                samples = len(episode_data)
            
        except FileNotFoundError:
            print(f"⚠️  File not found (may have been moved/deleted): {file_path.name}")
            return 0, None
        except PermissionError:
            print(f"⚠️  Permission denied accessing: {file_path.name}")
            return 0, None
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON decode error in {file_path.name}: {e}")
            # Fallback: estimate from file size
            try:
                file_size = file_path.stat().st_size
                samples = max(1, file_size // 500)  # Rough estimate
            except:
                samples = 0
        except Exception as e:
            print(f"⚠️  Could not parse {file_path.name}: {e}")
            # Fallback: estimate from file size
            try:
                file_size = file_path.stat().st_size
                samples = max(1, file_size // 500)  # Rough estimate
            except:
                samples = 0

        return samples, metadata
    
    def print_dataset_summary(self, datasets: List[Dict]):
        """Print summary of all datasets"""
        if not datasets:
            print("📭 No dataset files found")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 DATASET MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        for dataset in datasets:
            name = dataset['name']
            status = dataset['status']
            
            # Format timing info
            start_str = "Unknown"
            last_activity_str = "Unknown"
            if status['start_time']:
                start_dt = datetime.fromtimestamp(status['start_time'])
                start_str = start_dt.strftime('%H:%M:%S')
                
            if status['last_activity']:
                last_dt = datetime.fromtimestamp(status['last_activity'])
                last_activity_str = last_dt.strftime('%H:%M:%S')
                
                # Calculate time since last activity
                time_since = datetime.now() - last_dt
                if time_since.total_seconds() > 300:  # 5 minutes
                    last_activity_str += f" ({time_since.total_seconds()/60:.0f}m ago)"
            
            print(f"\n📁 Dataset: {name}")
            print(f"   Path: {dataset['path']}")
            print(f"   📊 Total samples: {status['total_samples_generated']:,}")
            print(f"   📦 Batches/Files: {status['total_batches']}")
            if status['temp_files'] > 0:
                print(f"   🔄 In progress: {status['temp_files']} temp files")
            
            # Show estimation info if available
            if status.get('avg_bytes_per_sample'):
                print(f"   📏 Estimation: {status['avg_bytes_per_sample']:.0f} bytes/sample (fast mode)")
            
            if status['estimated_target']:
                progress_pct = (status['total_samples_generated'] / status['estimated_target']) * 100
                print(f"   🎯 Progress: {progress_pct:.1f}% ({status['total_samples_generated']}/{status['estimated_target']})")
            
            if status['batch_size']:
                print(f"   📏 Avg batch size: {status['batch_size']} samples")
                
            print(f"   🕐 Started: {start_str}")
            print(f"   🕑 Last activity: {last_activity_str}")
            print(f"   🏷️  Type: {status['dataset_type']}")
            
            # Calculate generation rate if we have timing info
            if status['start_time'] and status['total_samples_generated'] > 0:
                elapsed = time.time() - status['start_time']
                rate = status['total_samples_generated'] / elapsed if elapsed > 0 else 0
                print(f"   ⚡ Rate: {rate:.2f} samples/sec")
                
                if status['estimated_target'] and rate > 0:
                    remaining = status['estimated_target'] - status['total_samples_generated']
                    eta_seconds = remaining / rate
                    if eta_seconds > 0:
                        eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f}h"
                        print(f"   ⏰ ETA: {eta_str}")
    
    def print_detailed_info(self, datasets: List[Dict]):
        """Print detailed file-by-file breakdown"""
        print(f"\n{'='*80}")
        print("📋 DETAILED FILE BREAKDOWN")
        print(f"{'='*80}")
        
        for dataset in datasets:
            print(f"\n📁 {dataset['name']}")
            status = dataset['status']
            
            if status['file_details']:
                print("   Files:")
                for detail in status['file_details']:
                    metadata_str = ""
                    if detail['metadata']:
                        batch_start = detail['metadata'].get('batch_start_idx', '?')
                        metadata_str = f" (batch {batch_start})"
                    
                    # Show if this was estimated vs fully analyzed
                    estimate_marker = " ~" if detail.get('estimated', False) else ""
                    print(f"     📄 {detail['file']}: {detail['samples']}{estimate_marker} samples{metadata_str}")
            else:
                print("   No detailed file information available")
    
    def monitor_continuous(self):
        """Continuously monitor datasets"""
        print("🔍 Starting continuous dataset monitoring...")
        print(f"📂 Watching directory: {self.dataset_base_dir}")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                datasets = self.find_active_datasets()
                
                # Clear screen and show summary
                os.system('clear' if os.name == 'posix' else 'cls')
                self.print_dataset_summary(datasets)
                
                # Show detailed info periodically only if verbose
                current_time = time.time()
                if (self.verbose and 
                    current_time - self.last_detailed_update > self.detailed_interval):
                    self.print_detailed_info(datasets)
                    self.last_detailed_update = current_time
                
                print(f"\n🔄 Refreshing every {self.refresh_interval}s... (Press Ctrl+C to stop)")
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
    
    def show_once(self):
        """Show current status once and exit"""
        datasets = self.find_active_datasets()
        self.print_dataset_summary(datasets)
        if self.verbose:
            self.print_detailed_info(datasets)


def main():
    """Main entry point for the job watcher"""
    parser = argparse.ArgumentParser(description="Monitor SML dataset generation progress")
    parser.add_argument("--dataset_dir", type=str, default="./datasets",
                       help="Base directory containing datasets")
    parser.add_argument("--once", action="store_true",
                       help="Show status once and exit (don't monitor continuously)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed file breakdown information")
    parser.add_argument("--refresh", type=int, default=10,
                       help="Refresh interval in seconds for continuous monitoring")
    
    args = parser.parse_args()
    
    watcher = DatasetJobWatcher(args.dataset_dir, verbose=args.verbose)
    watcher.refresh_interval = args.refresh
    
    if args.once:
        watcher.show_once()
    else:
        watcher.monitor_continuous()


if __name__ == "__main__":
    main()
