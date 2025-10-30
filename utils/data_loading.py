"""
Data loading utilities for different file formats and dataset discovery.

Provides unified interfaces for loading data from HDF5, NPZ, JSON files with
memory-efficient caching and lazy loading capabilities.
"""

import os
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np

# Optional imports
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    print("Warning: h5py not available. HDF5 support disabled.")
    HDF5_AVAILABLE = False


class DatasetDiscovery:
    """Discover and analyze dataset files in a directory."""
    
    @staticmethod
    def discover_files(dataset_path: Union[str, Path]) -> Tuple[List[Path], str, Dict]:
        """
        Discover dataset files and return metadata about the dataset.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Tuple of (file_paths, dataset_type, metadata)
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        # Priority order: HDF5 > NPZ > JSON
        
        # 1. Check for HDF5 files
        if HDF5_AVAILABLE:
            hdf5_files = list(dataset_path.glob("*.h5")) + list(dataset_path.glob("*.hdf5"))
            if hdf5_files:
                return DatasetDiscovery._analyze_hdf5_files(hdf5_files)
        
        # 2. Check for NPZ files
        npz_files = list(dataset_path.glob("*.npz"))
        if npz_files:
            return DatasetDiscovery._analyze_npz_files(npz_files)
        
        # 3. Check for JSON files
        json_files = (list(dataset_path.glob("episode_*.json")) + 
                     list(dataset_path.glob("batch_*.json")))
        if json_files:
            return DatasetDiscovery._analyze_json_files(json_files)
        
        raise ValueError(f"No supported dataset files found in {dataset_path}")
    
    @staticmethod
    def _analyze_hdf5_files(hdf5_files: List[Path]) -> Tuple[List[Path], str, Dict]:
        """Analyze HDF5 files and extract metadata."""
        sample_file = hdf5_files[0]
        
        try:
            with h5py.File(sample_file, 'r') as f:
                # Determine data structure
                keys = list(f.keys())
                
                if 'observations' in keys:
                    # Standard format with observations key
                    obs_shape = f['observations'].shape
                    obs_dtype = f['observations'].dtype
                    
                    # Count total observations
                    total_obs = 0
                    for hdf5_file in hdf5_files:
                        try:
                            with h5py.File(hdf5_file, 'r') as f2:
                                total_obs += f2['observations'].shape[0]
                        except Exception:
                            continue
                    
                    metadata = {
                        'total_observations': total_obs,
                        'observation_shape': obs_shape[1:],  # Remove batch dimension
                        'dtype': obs_dtype,
                        'format': 'observations_only',
                        'keys': keys
                    }
                    
                elif 'observations' in keys and 'actions' in keys:
                    # Observation-action pairs
                    obs_shape = f['observations'].shape
                    action_shape = f['actions'].shape
                    obs_dtype = f['observations'].dtype
                    action_dtype = f['actions'].dtype
                    
                    # Count total pairs
                    total_pairs = 0
                    for hdf5_file in hdf5_files:
                        try:
                            with h5py.File(hdf5_file, 'r') as f2:
                                total_pairs += min(f2['observations'].shape[0], 
                                                 f2['actions'].shape[0])
                        except Exception:
                            continue
                    
                    metadata = {
                        'total_observations': total_pairs,
                        'observation_shape': obs_shape[1:],
                        'action_shape': action_shape[1:],
                        'obs_dtype': obs_dtype,
                        'action_dtype': action_dtype,
                        'format': 'observation_action_pairs',
                        'keys': keys
                    }
                    
                else:
                    # Generic HDF5 format
                    metadata = {
                        'format': 'generic_hdf5',
                        'keys': keys,
                        'file_info': {key: f[key].shape for key in keys if hasattr(f[key], 'shape')}
                    }
                
                return sorted(hdf5_files), 'hdf5', metadata
                
        except Exception as e:
            raise ValueError(f"Error analyzing HDF5 files: {e}")
    
    @staticmethod
    def _analyze_npz_files(npz_files: List[Path]) -> Tuple[List[Path], str, Dict]:
        """Analyze NPZ files and extract metadata."""
        sample_file = npz_files[0]
        
        try:
            data = np.load(sample_file)
            keys = list(data.keys())
            
            # Look for common observation keys
            obs_key = None
            action_key = None
            
            for key in ['observations', 'obs', 'states', 'images']:
                if key in keys:
                    obs_key = key
                    break
            
            for key in ['actions', 'action', 'perfect_actions']:
                if key in keys:
                    action_key = key
                    break
            
            if obs_key:
                obs_shape = data[obs_key].shape
                obs_dtype = data[obs_key].dtype
                
                # Count total observations
                total_obs = 0
                for npz_file in npz_files:
                    try:
                        file_data = np.load(npz_file)
                        if obs_key in file_data:
                            total_obs += file_data[obs_key].shape[0]
                    except Exception:
                        continue
                
                metadata = {
                    'total_observations': total_obs,
                    'observation_shape': obs_shape[1:] if len(obs_shape) > 1 else obs_shape,
                    'dtype': obs_dtype,
                    'obs_key': obs_key,
                    'keys': keys
                }
                
                if action_key:
                    action_shape = data[action_key].shape
                    action_dtype = data[action_key].dtype
                    metadata.update({
                        'action_shape': action_shape[1:] if len(action_shape) > 1 else action_shape,
                        'action_dtype': action_dtype,
                        'action_key': action_key,
                        'format': 'observation_action_pairs'
                    })
                else:
                    metadata['format'] = 'observations_only'
            else:
                metadata = {
                    'format': 'generic_npz',
                    'keys': keys
                }
            
            return sorted(npz_files), 'npz', metadata
            
        except Exception as e:
            raise ValueError(f"Error analyzing NPZ files: {e}")
    
    @staticmethod
    def _analyze_json_files(json_files: List[Path]) -> Tuple[List[Path], str, Dict]:
        """Analyze JSON files and extract metadata."""
        sample_file = json_files[0]
        
        try:
            with open(sample_file, 'r') as f:
                episode_data = json.load(f)
            
            # Count total observations
            total_obs = 0
            obs_shape = None
            action_shape = None
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        
                        if 'observations' in data:
                            file_obs = len(data['observations'])
                            total_obs += file_obs
                            
                            if obs_shape is None and file_obs > 0:
                                obs_array = np.array(data['observations'][0])
                                obs_shape = obs_array.shape
                        
                        if 'actions' in data and action_shape is None:
                            if len(data['actions']) > 0:
                                action_array = np.array(data['actions'][0])
                                action_shape = action_array.shape
                                
                except Exception:
                    continue
            
            metadata = {
                'total_observations': total_obs,
                'format': 'episode_json',
                'keys': list(episode_data.keys())
            }
            
            if obs_shape is not None:
                metadata['observation_shape'] = obs_shape
            if action_shape is not None:
                metadata['action_shape'] = action_shape
                metadata['format'] = 'observation_action_pairs'
            
            return sorted(json_files), 'json', metadata
            
        except Exception as e:
            raise ValueError(f"Error analyzing JSON files: {e}")


class FileLoader:
    """Efficient file loading with format-specific optimizations."""
    
    @staticmethod
    def load_hdf5_observations(file_path: Path, keys: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Load data from HDF5 file."""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py not available for HDF5 loading")
        
        with h5py.File(file_path, 'r') as f:
            if keys is None:
                keys = list(f.keys())
            
            data = {}
            for key in keys:
                if key in f:
                    data[key] = f[key][:]
            
            return data
    
    @staticmethod
    def load_npz_observations(file_path: Path, keys: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Load data from NPZ file."""
        data_raw = np.load(file_path)
        
        if keys is None:
            keys = list(data_raw.keys())
        
        data = {}
        for key in keys:
            if key in data_raw:
                data[key] = data_raw[key]
        
        return data
    
    @staticmethod
    def load_json_observations(file_path: Path) -> Dict[str, Any]:
        """Load data from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)


class CacheManager:
    """Manage dataset caching for improved loading performance."""
    
    @staticmethod
    def get_cache_path(dataset_path: Path, 
                      cache_type: str = 'processed_pairs',
                      max_examples: Optional[int] = None) -> Path:
        """Get cache file path based on dataset and parameters."""
        if max_examples is not None:
            cache_name = f"{cache_type}_cache_{max_examples}.pkl"
        else:
            cache_name = f"{cache_type}_cache.pkl"
        
        return dataset_path / cache_name
    
    @staticmethod
    def save_cache(data: Any, cache_path: Path) -> None:
        """Save data to cache file."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_cache(cache_path: Path) -> Any:
        """Load data from cache file."""
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Cache loading failed: {e}")
            return None
    
    @staticmethod
    def clear_cache(dataset_path: Path, cache_pattern: str = "*_cache*.pkl") -> None:
        """Clear cache files matching pattern."""
        cache_files = list(dataset_path.glob(cache_pattern))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                print(f"Removed cache file: {cache_file.name}")
            except Exception as e:
                print(f"Warning: Could not remove {cache_file.name}: {e}")


# Export main classes and functions
__all__ = [
    'DatasetDiscovery',
    'FileLoader', 
    'CacheManager',
    'HDF5_AVAILABLE',
]