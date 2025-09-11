#!/usr/bin/env python3
"""
Simple launcher for SML dataset generation with background job monitoring.
"""

import subprocess
import time
import sys
from pathlib import Path

def main():
    print("🚀 Simple SML Dataset Generation System")
    print("=" * 50)
    print("Features:")
    print("  ✅ Atomic file writes (no corruption)")
    print("  ✅ Simple background execution")
    print("  ✅ Separate job monitoring")
    print("=" * 50)
    
    # Check if we're in the right directory
    sml_dir = Path("optomech/supervised_ml")
    if not sml_dir.exists():
        print("❌ Error: Run this from the visuomotor-deep-optics root directory")
        return 1
    
    print("\nChoose an option:")
    print("1. Run 100K sample job in background + monitor")
    print("2. Run 1K sample test in background + monitor")
    print("3. Just start the job watcher")
    print("4. Run small 20 sample test")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🌙 Starting 100K sample job in background...")
        # Start job manager in background
        subprocess.Popen([
            "poetry", "run", "python", "optomech/supervised_ml/sml_job_manager_simple.py",
            "--config", "optomech/supervised_ml/sml_job_config.json"
        ])
        time.sleep(3)  # Give it a moment to start
        
        print("📊 Starting job watcher...")
        # Start job watcher in foreground
        subprocess.run([
            "poetry", "run", "python", "optomech/supervised_ml/sml_job_watcher.py",
            "--dataset_dir", "optomech/supervised_ml/datasets",
            "--detailed"
        ])
        
    elif choice == "2":
        print("\n🧪 Starting 1K sample test in background...")
        subprocess.Popen([
            "poetry", "run", "python", "optomech/supervised_ml/sml_job_manager_simple.py",
            "--config", "optomech/supervised_ml/sml_job_config.json",
            "--total_samples", "1000"
        ])
        time.sleep(3)
        
        print("📊 Starting job watcher...")
        subprocess.run([
            "poetry", "run", "python", "optomech/supervised_ml/sml_job_watcher.py",
            "--dataset_dir", "optomech/supervised_ml/datasets",
            "--detailed"
        ])
        
    elif choice == "3":
        print("\n🔍 Starting job watcher...")
        subprocess.run([
            "poetry", "run", "python", "optomech/supervised_ml/sml_job_watcher.py",
            "--dataset_dir", "optomech/supervised_ml/datasets",
            "--detailed"
        ])
        
    elif choice == "4":
        print("\n🧪 Running small test (20 samples)...")
        subprocess.run([
            "poetry", "run", "python", "optomech/supervised_ml/sml_job_manager_simple.py",
            "--config", "optomech/supervised_ml/sml_job_config.json",
            "--total_samples", "20"
        ])
        
    else:
        print("❌ Invalid choice")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
