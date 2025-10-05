#!/usr/bin/env python3
"""
SA Dataset Episode Renderer

This script loads SA datasets and renders episodes as animated GIFs showing:
- Observation progression (optical images)
- SA actions, perfect actions, and incremental actions as plots
- Reward progression over episode steps
- Episode metadata and statistics

Usage:
    python sa_dataset_renderer.py --dataset-path ./datasets/my_sa_dataset --num-episodes 2
    python sa_dataset_renderer.py --dataset-path ./datasets/my_sa_dataset --episode-ids episode_uuid_1 episode_uuid_2
"""

import os
import sys
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import uuid

# Try to import additional dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL not available - some image processing features may be limited")

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class SADatasetRenderer:
    """Renders SA dataset episodes as animated visualizations"""
    
    def __init__(self, dataset_path: str, output_dir: str = "./episode_renders"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all episode data
        self.episodes = self._load_all_episodes()
        print(f"📊 Loaded {len(self.episodes)} episodes from dataset")
        
    def _load_all_episodes(self) -> Dict[str, List[Dict]]:
        """Load and group all samples by episode ID"""
        episodes = defaultdict(list)
        
        # Find all batch files
        batch_files = list(self.dataset_path.glob("*.h5")) + list(self.dataset_path.glob("*.npz"))
        
        if not batch_files:
            raise ValueError(f"No batch files found in {self.dataset_path}")
        
        print(f"📁 Found {len(batch_files)} batch files")
        
        for batch_file in batch_files:
            print(f"   Loading {batch_file.name}...")
            
            if batch_file.suffix == '.h5':
                samples = self._load_h5_batch(batch_file)
            else:
                samples = self._load_npz_batch(batch_file)
            
            # Group by episode
            for sample in samples:
                episodes[sample['episode_id']].append(sample)
        
        # Sort each episode by step number
        for episode_id in episodes:
            episodes[episode_id].sort(key=lambda x: x['episode_step'])
        
        return dict(episodes)
    
    def _load_h5_batch(self, batch_file: Path) -> List[Dict]:
        """Load samples from HDF5 batch file"""
        samples = []
        
        with h5py.File(batch_file, 'r') as f:
            batch_size = f['observations'].shape[0]
            
            for i in range(batch_size):
                episode_id = f['episode_ids'][i]
                if isinstance(episode_id, bytes):
                    episode_id = episode_id.decode('utf-8')
                
                sample = {
                    'observation': f['observations'][i],
                    'sa_action': f['sa_actions'][i],
                    'perfect_action': f['perfect_actions'][i],
                    'sa_incremental_action': f['sa_incremental_actions'][i],
                    'perfect_incremental_action': f['perfect_incremental_actions'][i],
                    'reward': float(f['rewards'][i]),
                    'temperature': float(f['temperatures'][i]),
                    'acceptance_delta': float(f['acceptance_deltas'][i]),
                    'episode_id': episode_id,
                    'episode_step': int(f['episode_steps'][i]),
                    'batch_file': batch_file.name
                }
                samples.append(sample)
        
        return samples
    
    def _load_npz_batch(self, batch_file: Path) -> List[Dict]:
        """Load samples from NPZ batch file"""
        samples = []
        
        with np.load(batch_file) as data:
            batch_size = data['observations'].shape[0]
            
            for i in range(batch_size):
                sample = {
                    'observation': data['observations'][i],
                    'sa_action': data['sa_actions'][i],
                    'perfect_action': data['perfect_actions'][i],
                    'sa_incremental_action': data['sa_incremental_actions'][i],
                    'perfect_incremental_action': data['perfect_incremental_actions'][i],
                    'reward': float(data['rewards'][i]),
                    'temperature': float(data['temperatures'][i]),
                    'acceptance_delta': float(data['acceptance_deltas'][i]),
                    'episode_id': str(data['episode_ids'][i]),
                    'episode_step': int(data['episode_steps'][i]),
                    'batch_file': batch_file.name
                }
                samples.append(sample)
        
        return samples
    
    def get_episode_stats(self) -> Dict:
        """Get statistics about loaded episodes"""
        stats = {
            'num_episodes': len(self.episodes),
            'total_samples': sum(len(samples) for samples in self.episodes.values()),
            'episode_lengths': [len(samples) for samples in self.episodes.values()],
            'episode_ids': list(self.episodes.keys())
        }
        
        stats['min_episode_length'] = min(stats['episode_lengths']) if stats['episode_lengths'] else 0
        stats['max_episode_length'] = max(stats['episode_lengths']) if stats['episode_lengths'] else 0
        stats['mean_episode_length'] = np.mean(stats['episode_lengths']) if stats['episode_lengths'] else 0
        
        return stats
    
    def render_episode_gif(self, episode_id: str, fps: int = 2, figsize: Tuple[int, int] = (16, 12)) -> str:
        """Render a single episode as an animated GIF"""
        if episode_id not in self.episodes:
            raise ValueError(f"Episode {episode_id} not found in dataset")
        
        episode_samples = self.episodes[episode_id]
        num_steps = len(episode_samples)
        
        print(f"🎬 Rendering episode {episode_id[:8]}... ({num_steps} steps)")
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        
        # Define subplot layout: 2x3 grid
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1, 1], 
                             hspace=0.3, wspace=0.3)
        
        # Observation image (top left, spans 2 columns)
        ax_obs = fig.add_subplot(gs[0, :2])
        
        # Reward progression (top right, spans 2 columns)
        ax_reward = fig.add_subplot(gs[0, 2:])
        
        # SA actions (middle left)
        ax_sa = fig.add_subplot(gs[1, :2])
        
        # Perfect actions (middle right)
        ax_perfect = fig.add_subplot(gs[1, 2:])
        
        # Incremental actions (bottom)
        ax_inc_sa = fig.add_subplot(gs[2, :2])
        ax_inc_perfect = fig.add_subplot(gs[2, 2:])
        
        # Prepare data for plotting
        steps = list(range(num_steps))
        rewards = [sample['reward'] for sample in episode_samples]
        temperatures = [sample['temperature'] for sample in episode_samples]
        action_dim = len(episode_samples[0]['sa_action'])
        
        # Animation function
        def animate(frame):
            # Clear all axes
            for ax in [ax_obs, ax_reward, ax_sa, ax_perfect, ax_inc_sa, ax_inc_perfect]:
                ax.clear()
            
            current_sample = episode_samples[frame]
            
            # 1. Observation image
            obs = current_sample['observation']
            if len(obs.shape) == 3 and obs.shape[0] == 1:
                obs = obs[0]  # Remove channel dimension if present
            
            im = ax_obs.imshow(obs, cmap='gray', vmin=0, vmax=65535)
            ax_obs.set_title(f"Observation - Step {current_sample['episode_step']}")
            ax_obs.axis('off')
            
            # 2. Reward progression
            current_steps = steps[:frame+1]
            current_rewards = rewards[:frame+1]
            current_temps = temperatures[:frame+1]
            
            ax_reward.plot(current_steps, current_rewards, 'b-o', label='Reward', linewidth=2, markersize=6)
            ax_reward.set_xlabel('Episode Step')
            ax_reward.set_ylabel('Reward', color='b')
            ax_reward.tick_params(axis='y', labelcolor='b')
            ax_reward.grid(True, alpha=0.3)
            ax_reward.set_title(f"Reward: {current_sample['reward']:.4f}")
            
            # Add temperature on secondary axis
            ax_temp = ax_reward.twinx()
            ax_temp.plot(current_steps, current_temps, 'r--s', label='Temperature', alpha=0.7, markersize=4)
            ax_temp.set_ylabel('Temperature', color='r')
            ax_temp.tick_params(axis='y', labelcolor='r')
            
            # 3. SA Actions
            sa_action = current_sample['sa_action']
            action_indices = list(range(action_dim))
            bars_sa = ax_sa.bar(action_indices, sa_action, alpha=0.7, color='blue')
            ax_sa.set_title('SA Action')
            ax_sa.set_xlabel('Action Dimension')
            ax_sa.set_ylabel('Action Value')
            ax_sa.set_ylim(-1, 1)
            ax_sa.grid(True, alpha=0.3)
            
            # 4. Perfect Actions
            perfect_action = current_sample['perfect_action']
            bars_perfect = ax_perfect.bar(action_indices, perfect_action, alpha=0.7, color='green')
            ax_perfect.set_title('Perfect Action')
            ax_perfect.set_xlabel('Action Dimension')
            ax_perfect.set_ylabel('Action Value')
            ax_perfect.set_ylim(-1, 1)
            ax_perfect.grid(True, alpha=0.3)
            
            # 5. SA Incremental Actions
            sa_inc = current_sample['sa_incremental_action']
            bars_sa_inc = ax_inc_sa.bar(action_indices, sa_inc, alpha=0.7, color='cyan')
            ax_inc_sa.set_title('SA Incremental Action')
            ax_inc_sa.set_xlabel('Action Dimension')
            ax_inc_sa.set_ylabel('Incremental Value')
            ax_inc_sa.set_ylim(-2, 2)
            ax_inc_sa.grid(True, alpha=0.3)
            
            # 6. Perfect Incremental Actions
            perfect_inc = current_sample['perfect_incremental_action']
            bars_perfect_inc = ax_inc_perfect.bar(action_indices, perfect_inc, alpha=0.7, color='orange')
            ax_inc_perfect.set_title('Perfect Incremental Action')
            ax_inc_perfect.set_xlabel('Action Dimension')
            ax_inc_perfect.set_ylabel('Incremental Value')
            ax_inc_perfect.set_ylim(-2, 2)
            ax_inc_perfect.grid(True, alpha=0.3)
            
            # Add episode info as text
            info_text = (f"Episode: {episode_id[:8]}...\n"
                        f"Step: {current_sample['episode_step']}\n"
                        f"Temperature: {current_sample['temperature']:.3f}\n"
                        f"Accept Δ: {current_sample['acceptance_delta']:.4f}")
            
            fig.text(0.02, 0.98, info_text, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=num_steps, interval=1000//fps, repeat=True)
        
        # Save as GIF
        gif_filename = f"episode_{episode_id[:8]}_{num_steps}steps.gif"
        gif_path = self.output_dir / gif_filename
        
        print(f"💾 Saving GIF to {gif_path}")
        anim.save(gif_path, writer='pillow', fps=fps)
        plt.close(fig)
        
        return str(gif_path)
    
    def render_multiple_episodes(self, episode_ids: Optional[List[str]] = None, 
                                num_episodes: Optional[int] = None, fps: int = 2) -> List[str]:
        """Render multiple episodes as GIFs"""
        if episode_ids is None:
            if num_episodes is None:
                num_episodes = min(3, len(self.episodes))  # Default to 3 episodes
            
            # Select episodes (prefer longer episodes for better visualization)
            episode_lengths = [(ep_id, len(samples)) for ep_id, samples in self.episodes.items()]
            episode_lengths.sort(key=lambda x: x[1], reverse=True)
            episode_ids = [ep_id for ep_id, _ in episode_lengths[:num_episodes]]
        
        gif_paths = []
        for episode_id in episode_ids:
            try:
                gif_path = self.render_episode_gif(episode_id, fps=fps)
                gif_paths.append(gif_path)
                print(f"✅ Successfully rendered episode {episode_id[:8]}...")
            except Exception as e:
                print(f"❌ Failed to render episode {episode_id[:8]}...: {e}")
        
        return gif_paths
    
    def print_dataset_summary(self):
        """Print a summary of the loaded dataset"""
        stats = self.get_episode_stats()
        
        print("\n" + "="*60)
        print("📊 SA DATASET SUMMARY")
        print("="*60)
        print(f"Dataset path: {self.dataset_path}")
        print(f"Total episodes: {stats['num_episodes']}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Episode lengths: min={stats['min_episode_length']}, max={stats['max_episode_length']}, mean={stats['mean_episode_length']:.1f}")
        
        print(f"\n📋 Episode IDs:")
        for i, ep_id in enumerate(stats['episode_ids'][:10]):  # Show first 10
            ep_len = len(self.episodes[ep_id])
            print(f"  {i+1:2d}. {ep_id[:8]}... ({ep_len} steps)")
        
        if len(stats['episode_ids']) > 10:
            print(f"  ... and {len(stats['episode_ids']) - 10} more episodes")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Render SA dataset episodes as animated GIFs")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to the SA dataset directory")
    parser.add_argument("--num-episodes", type=int, default=None,
                       help="Number of episodes to render (default: auto-select)")
    parser.add_argument("--episode-ids", nargs='+', default=None,
                       help="Specific episode IDs to render")
    parser.add_argument("--output-dir", type=str, default="./episode_renders",
                       help="Directory to save rendered GIFs")
    parser.add_argument("--fps", type=int, default=2,
                       help="Frames per second for GIF animation")
    parser.add_argument("--summary-only", action="store_true",
                       help="Only print dataset summary, don't render")
    
    args = parser.parse_args()
    
    print("🎬 SA Dataset Episode Renderer")
    print("="*50)
    
    try:
        # Initialize renderer
        renderer = SADatasetRenderer(args.dataset_path, args.output_dir)
        
        # Print summary
        renderer.print_dataset_summary()
        
        if args.summary_only:
            return 0
        
        # Render episodes
        print(f"\n🎬 Rendering episodes...")
        gif_paths = renderer.render_multiple_episodes(
            episode_ids=args.episode_ids,
            num_episodes=args.num_episodes,
            fps=args.fps
        )
        
        print(f"\n✅ Rendering complete! Generated {len(gif_paths)} GIFs:")
        for gif_path in gif_paths:
            print(f"   📄 {gif_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
