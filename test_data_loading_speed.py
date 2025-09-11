#!/usr/bin/env python3
"""
Quick test to compare data loading speeds
"""
import time
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent / "optomech"))

from optomech.supervised_ml.train_sml_model import load_dataset_pairs_parallel

def test_loading_speed():
    """Test different loading configurations"""
    dataset_path = "datasets/sml_100k_dataset"
    
    print("🧪 Testing Data Loading Speeds")
    print("=" * 50)
    
    # Test with different worker counts
    worker_counts = [1, 4, 8, 16]
    
    for workers in worker_counts:
        print(f"\n🔄 Testing with {workers} workers...")
        
        start_time = time.time()
        pairs = load_dataset_pairs_parallel(dataset_path, max_workers=workers, use_cache=False)
        end_time = time.time()
        
        loading_time = end_time - start_time
        print(f"  Loaded {len(pairs)} pairs in {loading_time:.1f}s")
        print(f"  Speed: {len(pairs)/loading_time:.1f} pairs/second")
        
        if workers == 1:
            baseline_time = loading_time
        else:
            speedup = baseline_time / loading_time
            print(f"  Speedup: {speedup:.1f}x faster than 1 worker")
    
    print(f"\n🎯 Recommendation: Use 8 workers for optimal balance")

if __name__ == "__main__":
    test_loading_speed()
