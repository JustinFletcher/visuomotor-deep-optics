#!/usr/bin/env python3
"""
Simple H5 Dataset Watcher - Counts samples directly from H5 files

Opens each H5 file and counts the number of examples inside.
No metadata parsing, no job manager dependencies, just pure sample counting.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Try to import h5py
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("❌ h5py not available - cannot read HDF5 files")
    sys.exit(1)


def get_directory_size(directory: Path) -> int:
    """Get total size of directory in bytes"""
    total_size = 0
    try:
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except (OSError, PermissionError):
        pass
    return total_size


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


def count_samples_in_h5(file_path: Path) -> int:
    """Count samples in a single H5 file by reading the observations array"""
    try:
        with h5py.File(file_path, 'r') as f:
            # Look for observations array first (primary data)
            if 'observations' in f:
                return f['observations'].shape[0]
            # Fallback to perfect_actions if observations not found
            elif 'perfect_actions' in f:
                return f['perfect_actions'].shape[0]
            # If neither found, return 0
            else:
                print(f"   ⚠️  No 'observations' or 'perfect_actions' found in {file_path.name}")
                return 0
    except Exception as e:
        print(f"   ❌ Error reading {file_path.name}: {e}")
        return 0


def scan_directory(directory: Path) -> Dict:
    """Scan directory and count all samples in H5 files"""
    if not directory.exists():
        return {
            'total_samples': 0,
            'total_files': 0,
            'directory_size': 0,
            'oldest_file': None,
            'newest_file': None,
            'file_details': []
        }
    
    # Find all H5 files (including batch_*.h5 pattern)
    h5_files = []
    h5_files.extend(directory.glob("*.h5"))
    h5_files.extend(directory.glob("batch_*.h5"))
    
    # Remove duplicates and sort
    h5_files = sorted(list(set(h5_files)))
    
    total_samples = 0
    file_details = []
    timestamps = []
    
    print(f"🔍 Found {len(h5_files)} H5 files in {directory}")
    
    if len(h5_files) == 0:
        print("   No H5 files found")
        directory_size = get_directory_size(directory)
        return {
            'total_samples': 0,
            'total_files': 0,
            'directory_size': directory_size,
            'oldest_file': None,
            'newest_file': None,
            'file_details': []
        }
    
    for h5_file in h5_files:
        try:
            samples = count_samples_in_h5(h5_file)
            file_size = h5_file.stat().st_size
            file_time = h5_file.stat().st_mtime
            
            total_samples += samples
            timestamps.append(file_time)
            
            file_details.append({
                'name': h5_file.name,
                'samples': samples,
                'size': file_size,
                'modified': file_time
            })
            
            print(f"   📄 {h5_file.name}: {samples:,} samples ({format_bytes(file_size)})")
            
        except Exception as e:
            print(f"   ⚠️  Error processing {h5_file.name}: {e}")
    
    directory_size = get_directory_size(directory)
    
    return {
        'total_samples': total_samples,
        'total_files': len(h5_files),
        'directory_size': directory_size,
        'oldest_file': min(timestamps) if timestamps else None,
        'newest_file': max(timestamps) if timestamps else None,
        'file_details': file_details
    }


def compute_generation_rate(data: Dict) -> float:
    """Compute empirical generation rate from file timestamps"""
    if not data['oldest_file'] or not data['newest_file'] or data['total_samples'] == 0:
        return 0.0
    
    time_span = data['newest_file'] - data['oldest_file']
    if time_span <= 0:
        return 0.0
    
    return data['total_samples'] / time_span


def print_report(directory: Path, data: Dict):
    """Print a simple, clean report"""
    print(f"\n{'='*60}")
    print(f"📊 H5 DATASET REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    print(f"📁 Directory: {directory}")
    print(f"💾 Total size: {format_bytes(data['directory_size'])}")
    print(f"📦 H5 files: {data['total_files']}")
    print(f"🔢 Total samples: {data['total_samples']:,}")
    
    if data['total_samples'] > 0 and data['total_files'] > 0:
        avg_samples_per_file = data['total_samples'] / data['total_files']
        avg_bytes_per_sample = data['directory_size'] / data['total_samples']
        print(f"📊 Avg samples/file: {avg_samples_per_file:.1f}")
        print(f"📏 Avg bytes/sample: {avg_bytes_per_sample:.0f}")
    
    # Timing information
    if data['oldest_file'] and data['newest_file']:
        oldest = datetime.fromtimestamp(data['oldest_file'])
        newest = datetime.fromtimestamp(data['newest_file'])
        print(f"🕐 Oldest file: {oldest.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕑 Newest file: {newest.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Generation rate
        rate = compute_generation_rate(data)
        if rate > 0:
            print(f"⚡ Generation rate: {rate:.2f} samples/sec")
            
            # Time since last update
            time_since_last = time.time() - data['newest_file']
            if time_since_last > 0:
                minutes_since = time_since_last / 60
                if minutes_since < 60:
                    print(f"⏱️  Last update: {minutes_since:.1f} minutes ago")
                else:
                    hours_since = minutes_since / 60
                    print(f"⏱️  Last update: {hours_since:.1f} hours ago")


def monitor_directory(directory: Path, refresh_interval: int = 10):
    """Continuously monitor a directory"""
    print(f"🔍 Monitoring {directory} (refresh every {refresh_interval}s)")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Scan and report
            data = scan_directory(directory)
            print_report(directory, data)
            
            print(f"\n🔄 Refreshing in {refresh_interval}s... (Ctrl+C to stop)")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Simple H5 dataset sample counter")
    parser.add_argument("directory", nargs="?", default=".", 
                       help="Directory to scan (default: current directory)")
    parser.add_argument("--watch", "-w", action="store_true",
                       help="Continuously monitor the directory")
    parser.add_argument("--refresh", type=int, default=10,
                       help="Refresh interval in seconds for watch mode")
    
    args = parser.parse_args()
    
    directory = Path(args.directory).resolve()
    
    if not directory.exists():
        print(f"❌ Directory does not exist: {directory}")
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"❌ Path is not a directory: {directory}")
        sys.exit(1)
    
    if args.watch:
        monitor_directory(directory, args.refresh)
    else:
        data = scan_directory(directory)
        print_report(directory, data)


if __name__ == "__main__":
    main()
