#!/usr/bin/env python3
"""
Test loading performance with different approaches
"""

import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import numpy as np
from typing import List, Tuple

def load_single_episode(episode_file: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load a single episode file and extract observation-action pairs"""
    try:
        with open(episode_file, 'r') as f:
            data = json.load(f)
        
        if 'episode_data' not in data:
            return []
        
        episode_data = data['episode_data']
        pairs = []
        
        for step in episode_data:
            if 'observation' in step and 'perfect_action' in step:
                obs = np.array(step['observation'])
                action = np.array(step['perfect_action'])
                if obs.size > 0 and action.size > 0:
                    pairs.append((obs, action))
        
        return pairs
    except Exception as e:
        print(f"Error loading {episode_file}: {e}")
        return []

def load_episode_chunk(episode_files: List[Path]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load a chunk of episode files (for multiprocessing)"""
    pairs = []
    for episode_file in episode_files:
        episode_pairs = load_single_episode(episode_file)
        pairs.extend(episode_pairs)
    return pairs

def test_sequential(files: List[Path]) -> float:
    """Test sequential loading"""
    print(f"Testing sequential loading with {len(files)} files...")
    start_time = time.time()
    
    pairs = []
    for file in files:
        episode_pairs = load_single_episode(file)
        pairs.extend(episode_pairs)
    
    load_time = time.time() - start_time
    print(f"Sequential: {len(pairs)} pairs in {load_time:.2f}s ({len(pairs)/load_time:.1f} pairs/sec)")
    return load_time

def test_threads(files: List[Path], max_workers: int) -> float:
    """Test ThreadPoolExecutor"""
    print(f"Testing ThreadPoolExecutor with {max_workers} workers, {len(files)} files...")
    start_time = time.time()
    
    pairs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_single_episode, f) for f in files]
        for future in futures:
            episode_pairs = future.result()
            pairs.extend(episode_pairs)
    
    load_time = time.time() - start_time
    print(f"Threads({max_workers}): {len(pairs)} pairs in {load_time:.2f}s ({len(pairs)/load_time:.1f} pairs/sec)")
    return load_time

def test_processes(files: List[Path], max_workers: int) -> float:
    """Test ProcessPoolExecutor"""
    print(f"Testing ProcessPoolExecutor with {max_workers} workers, {len(files)} files...")
    start_time = time.time()
    
    # Chunk files for better load balancing
    chunk_size = max(1, len(files) // (max_workers * 2))
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    
    pairs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_episode_chunk, chunk) for chunk in chunks]
        for future in futures:
            chunk_pairs = future.result()
            pairs.extend(chunk_pairs)
    
    load_time = time.time() - start_time
    print(f"Processes({max_workers}): {len(pairs)} pairs in {load_time:.2f}s ({len(pairs)/load_time:.1f} pairs/sec)")
    return load_time

def main():
    dataset_path = Path("datasets/sml_100k_dataset")
    
    # Find episode files
    all_files = list(dataset_path.glob("*.json"))
    episode_files = [f for f in all_files 
                    if f.name.startswith('episode_') and not f.name.startswith('.tmp_')]
    
    print(f"Found {len(episode_files)} episode files")
    
    # Test with different subset sizes
    test_sizes = [10, 50]
    
    for test_size in test_sizes:
        if test_size > len(episode_files):
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing with {test_size} files")
        print('='*60)
        
        test_files = episode_files[:test_size]
        
        # Test different approaches
        sequential_time = test_sequential(test_files)
        
        # Test different worker counts
        for workers in [1, 2, 4, 8]:
            if workers <= cpu_count():
                thread_time = test_threads(test_files, workers)
                process_time = test_processes(test_files, workers)
                
                print(f"  Speedup vs sequential: Thread({workers})={sequential_time/thread_time:.2f}x, Process({workers})={sequential_time/process_time:.2f}x")

if __name__ == "__main__":
    main()
