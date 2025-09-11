#!/usr/bin/env python3
"""
Launcher script demonstrating the enhanced SML dataset generation system
with atomic file writes and asynchronous job monitoring.
"""

import subprocess
import time
import sys
from pathlib import Path

def main():
    print("🚀 Enhanced SML Dataset Generation System")
    print("=" * 50)
    print("Features:")
    print("  ✅ Atomic file writes (no corruption during concurrent access)")
    print("  ✅ Asynchronous job monitoring")
    print("  ✅ Background execution support")
    print("  ✅ Real-time progress tracking")
    print("=" * 50)
    
    # Check if we're in the right directory
    sml_dir = Path("optomech/supervised_ml")
    if not sml_dir.exists():
        print("❌ Error: Run this from the visuomotor-deep-optics root directory")
        return 1
    
    print("\nChoose an option:")
    print("1. Run 100K sample job in background with job watcher")
    print("2. Run 100K sample job with verbose progress")
    print("3. Just start the job watcher (for existing jobs)")
    print("4. Run small test (1K samples)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🌙 Starting 100K sample job in background...")
        # Start job manager in background
        subprocess.Popen([
            "poetry", "run", "python", "optomech/supervised_ml/sml_job_manager.py",
            "--config", "sml_job_config.json",
            "--background"
        ])
        time.sleep(2)  # Give it a moment to start
        
        print("📊 Starting job watcher...")
        # Start job watcher in foreground
        subprocess.run([
            "poetry", "run", "python", "optomech/supervised_ml/sml_job_watcher.py",
            "--dataset_dir", "./datasets",
            "--detailed"
        ])
        
    elif choice == "2":
        print("\n👁️ Starting 100K sample job with verbose progress...")
        subprocess.run([
            "poetry", "run", "python", "optomech/supervised_ml/sml_job_manager.py",
            "--config", "sml_job_config.json"
        ])
        
    elif choice == "3":
        print("\n🔍 Starting job watcher...")
        subprocess.run([
            "poetry", "run", "python", "optomech/supervised_ml/sml_job_watcher.py",
            "--dataset_dir", "./datasets",
            "--detailed"
        ])
        
    elif choice == "4":
        print("\n🧪 Running small test (1K samples)...")
        subprocess.run([
            "poetry", "run", "python", "optomech/supervised_ml/sml_job_manager.py",
            "--config", "sml_job_config.json",
            "--total_samples", "1000"
        ])
        
    else:
        print("❌ Invalid choice")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
