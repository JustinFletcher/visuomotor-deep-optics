#!/usr/bin/env python3
"""
Simple H5 SA Dataset Watcher - Counts samples directly from H5 files

Opens each H5 file and counts the number of SA examples inside.
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
    """Count samples in a single H5 file by reading the SA actions array"""
    try:
        with h5py.File(file_path, 'r') as f:
            # Look for SA-specific arrays first
            if 'sa_actions' in f:
                return f['sa_actions'].shape[0]
            # Fallback to observations if sa_actions not found
            elif 'observations' in f:
                return f['observations'].shape[0]
            # If neither found, return 0
            else:
                print(f"   ⚠️  No 'sa_actions' or 'observations' found in {file_path.name}")
                return 0
    except Exception as e:
        print(f"   ❌ Error reading {file_path.name}: {e}")
        return 0


def scan_dataset_directory(dataset_dir: Path, detailed: bool = False) -> Dict:
    """Scan a dataset directory and count all samples"""
    result = {
        'dataset_name': dataset_dir.name,
        'total_samples': 0,
        'total_files': 0,
        'total_size': 0,
        'files': []
    }
    
    if not dataset_dir.exists():
        return result
    
    # Find all H5 files
    h5_files = list(dataset_dir.glob('*.h5'))
    result['total_files'] = len(h5_files)
    result['total_size'] = get_directory_size(dataset_dir)
    
    for h5_file in sorted(h5_files):
        samples = count_samples_in_h5(h5_file)
        result['total_samples'] += samples
        
        if detailed:
            file_info = {
                'name': h5_file.name,
                'samples': samples,
                'size': h5_file.stat().st_size,
                'modified': datetime.fromtimestamp(h5_file.stat().st_mtime)
            }
            result['files'].append(file_info)
    
    return result


def print_dataset_summary(dataset_info: Dict, detailed: bool = False):
    """Print a summary of dataset information"""
    name = dataset_info['dataset_name']
    samples = dataset_info['total_samples']
    files = dataset_info['total_files']
    size = format_bytes(dataset_info['total_size'])
    
    print(f"📁 {name}: {samples:,} samples in {files} files ({size})")
    
    if detailed and dataset_info['files']:
        print("   📄 Files:")
        for file_info in dataset_info['files']:
            samples = file_info['samples']
            size = format_bytes(file_info['size'])
            modified = file_info['modified'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"      {file_info['name']}: {samples:,} samples ({size}) - {modified}")


def watch_datasets(dataset_dir: Path, detailed: bool = False, interval: int = 5):
    """Continuously watch dataset directories"""
    print("🔍 SA Dataset Watcher - Monitoring HDF5 files")
    print("=" * 60)
    print(f"📂 Watching: {dataset_dir}")
    print(f"🕐 Update interval: {interval} seconds")
    print(f"📊 Detailed mode: {'ON' if detailed else 'OFF'}")
    print("💡 Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        while True:
            # Clear screen (works on Unix-like systems)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("🔍 SA Dataset Status")
            print("=" * 60)
            print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Scan for dataset directories
            if dataset_dir.exists():
                subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
                
                if not subdirs:
                    # No subdirectories, scan the main directory
                    dataset_info = scan_dataset_directory(dataset_dir, detailed)
                    if dataset_info['total_samples'] > 0:
                        print_dataset_summary(dataset_info, detailed)
                    else:
                        print("📂 No SA datasets found yet...")
                else:
                    # Scan each subdirectory
                    total_all_samples = 0
                    total_all_files = 0
                    total_all_size = 0
                    
                    for subdir in sorted(subdirs):
                        dataset_info = scan_dataset_directory(subdir, detailed)
                        if dataset_info['total_samples'] > 0:
                            print_dataset_summary(dataset_info, detailed)
                            total_all_samples += dataset_info['total_samples']
                            total_all_files += dataset_info['total_files']
                            total_all_size += dataset_info['total_size']
                    
                    if total_all_samples > 0:
                        print("─" * 60)
                        print(f"📊 TOTAL: {total_all_samples:,} samples in {total_all_files} files ({format_bytes(total_all_size)})")
                    else:
                        print("📂 No SA datasets found yet...")
            else:
                print(f"❌ Directory not found: {dataset_dir}")
                print("💡 Waiting for directory to be created...")
            
            print()
            print(f"🔄 Next update in {interval} seconds... (Ctrl+C to stop)")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n👋 SA Dataset Watcher stopped.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Watch SA dataset generation progress")
    parser.add_argument("--dataset_dir", type=str, default="./datasets",
                       help="Directory containing SA datasets to monitor")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed file information")
    parser.add_argument("--interval", type=int, default=5,
                       help="Update interval in seconds")
    parser.add_argument("--once", action="store_true",
                       help="Run once and exit (don't watch continuously)")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir).resolve()
    
    if args.once:
        # Single scan mode
        print("🔍 SA Dataset Scanner")
        print("=" * 50)
        print(f"📂 Scanning: {dataset_dir}")
        print()
        
        if dataset_dir.exists():
            subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
            
            if not subdirs:
                # No subdirectories, scan the main directory
                dataset_info = scan_dataset_directory(dataset_dir, args.detailed)
                if dataset_info['total_samples'] > 0:
                    print_dataset_summary(dataset_info, args.detailed)
                else:
                    print("📂 No SA datasets found.")
            else:
                # Scan each subdirectory
                total_all_samples = 0
                total_all_files = 0
                total_all_size = 0
                
                for subdir in sorted(subdirs):
                    dataset_info = scan_dataset_directory(subdir, args.detailed)
                    if dataset_info['total_samples'] > 0:
                        print_dataset_summary(dataset_info, args.detailed)
                        total_all_samples += dataset_info['total_samples']
                        total_all_files += dataset_info['total_files']
                        total_all_size += dataset_info['total_size']
                
                if total_all_samples > 0:
                    print("─" * 50)
                    print(f"📊 TOTAL: {total_all_samples:,} samples in {total_all_files} files ({format_bytes(total_all_size)})")
                else:
                    print("📂 No SA datasets found.")
        else:
            print(f"❌ Directory not found: {dataset_dir}")
    else:
        # Continuous watch mode
        watch_datasets(dataset_dir, args.detailed, args.interval)


if __name__ == "__main__":
    main()
