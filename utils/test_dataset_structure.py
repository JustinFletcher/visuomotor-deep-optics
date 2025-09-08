#!/usr/bin/env python3
"""
Test script to verify the new dataset directory structure works properly.
"""

import json
from pathlib import Path
from simple_job_manager_clean import launch_sa_job, find_next_dataset_id, count_all_transitions
import time

def test_directory_structure():
    """Test that the new directory structure works properly."""
    
    # Load job configuration (path relative to project root)
    import os
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "job_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_dataset_dir = config.get('base_dataset_dir', os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets"))
    print(f"🧪 Testing dataset structure with base directory: {base_dataset_dir}")
    
    # Test 1: Check that base directory gets created
    base_path = Path(base_dataset_dir)
    if base_path.exists():
        print(f"  ✅ Base directory already exists: {base_path}")
    else:
        print(f"  📁 Base directory will be created: {base_path}")
    
    # Test 2: Find next dataset ID
    next_id = find_next_dataset_id(base_dataset_dir)
    expected_path = base_path / f"dataset_{next_id}"
    print(f"  🔢 Next dataset ID: {next_id}")
    print(f"  📂 Expected dataset path: {expected_path}")
    
    # Test 3: Count current transitions
    total, datasets = count_all_transitions(base_dataset_dir)
    print(f"  📊 Current total transitions: {total:,}")
    print(f"  📈 Existing datasets: {datasets}")
    
    # Test 4: Simulate what a job launch command would look like
    print(f"\n🚀 Simulated job launch command for dataset_{next_id}:")
    print(f"  Dataset path would be: {expected_path}")
    
    # Create the base directory structure to verify it works
    base_path.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ Created base directory: {base_path}")
    
    # Test creating a subdirectory
    test_dataset = base_path / f"dataset_{next_id}"
    test_dataset.mkdir(exist_ok=True)
    print(f"  ✅ Created test dataset directory: {test_dataset}")
    
    # Test that the counting function now sees it
    total_after, datasets_after = count_all_transitions(base_dataset_dir)
    print(f"  📊 After creating directory - Total: {total_after:,}, Datasets: {datasets_after}")
    
    print(f"\n✨ Directory structure test completed successfully!")
    print(f"   Base directory: {base_path}")
    print(f"   Ready to launch jobs that will create: dataset_1, dataset_2, etc.")
    
    return base_dataset_dir

if __name__ == "__main__":
    test_directory_structure()
