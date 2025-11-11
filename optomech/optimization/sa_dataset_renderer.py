#!/usr/bin/env python3
"""
SA Dataset Episode Renderer

This script loads SA datasets and renders episodes as animated GIFs showing:
- Observation progression (optical images)
- SA actions, perfect actions, and incre        # Animation function
        def animate(frame):
            try:
                print(f"DEBUG: Animating frame {frame}")
            # Clear all axes
            for ax in ax_dict.values():
                ax.clear()                print(f"DEBUG: About to access episode_samples[{frame}]")
                current_sample = episode_samples[frame]
                print(f"DEBUG: Successfully got current_sample")
            except Exception as e:
                print(f"DEBUG: Error in animation frame {frame}: {e}")
                import traceback
                traceback.print_exc()
                raiseas plots
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
                
                # Handle both old (acceptance_deltas) and new (cost_deltas) field names
                cost_delta_key = 'cost_deltas' if 'cost_deltas' in f else 'acceptance_deltas'
                
                sample = {
                    'observation': f['observations'][i],
                    'sa_action': f['sa_actions'][i],
                    'perfect_action': f['perfect_actions'][i],
                    'sa_incremental_action': f['sa_incremental_actions'][i],
                    'perfect_incremental_action': f['perfect_incremental_actions'][i],
                    'reward': float(f['rewards'][i]),
                    'temperature': float(f['temperatures'][i]),
                    'cost_delta': float(f[cost_delta_key][i]),
                    'accepted': bool(f['accepted'][i]) if 'accepted' in f else True,  # Old datasets only had accepted transitions
                    'next_observation': f['next_observations'][i] if 'next_observations' in f else None,
                    'episode_id': episode_id,
                    'episode_step': int(f['episode_steps'][i]),
                    'optimization_step': int(f['optimization_steps'][i]) if 'optimization_steps' in f else None,
                    'batch_file': batch_file.name
                }
                samples.append(sample)
        
        return samples
    
    def _load_npz_batch(self, batch_file: Path) -> List[Dict]:
        """Load samples from NPZ batch file"""
        samples = []
        
        with np.load(batch_file) as data:
            batch_size = data['observations'].shape[0]
            
            # Handle both old and new field names
            cost_delta_key = 'cost_deltas' if 'cost_deltas' in data else 'acceptance_deltas'
            
            for i in range(batch_size):
                sample = {
                    'observation': data['observations'][i],
                    'sa_action': data['sa_actions'][i],
                    'perfect_action': data['perfect_actions'][i],
                    'sa_incremental_action': data['sa_incremental_actions'][i],
                    'perfect_incremental_action': data['perfect_incremental_actions'][i],
                    'reward': float(data['rewards'][i]),
                    'temperature': float(data['temperatures'][i]),
                    'cost_delta': float(data[cost_delta_key][i]),
                    'accepted': bool(data['accepted'][i]) if 'accepted' in data else True,
                    'next_observation': data['next_observations'][i] if 'next_observations' in data else None,
                    'episode_id': str(data['episode_ids'][i]),
                    'episode_step': int(data['episode_steps'][i]),
                    'optimization_step': int(data['optimization_steps'][i]) if 'optimization_steps' in data else None,
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
    
    def render_episode_gif(self, episode_id: str, fps: int = 2, figsize: Tuple[int, int] = (24, 16), 
                          render_interval: int = 1, max_transitions: Optional[int] = None, save_frames: bool = False) -> str:
        """Render a single episode as an animated GIF"""
        if episode_id not in self.episodes:
            raise ValueError(f"Episode {episode_id} not found in dataset")
        
        episode_samples = self.episodes[episode_id]
        
        # Limit number of transitions if requested
        if max_transitions is not None and len(episode_samples) > max_transitions:
            episode_samples = episode_samples[:max_transitions]
            print(f"⚠️  Limiting to first {max_transitions} transitions (out of {len(self.episodes[episode_id])})")
        
        num_steps = len(episode_samples)
        
        # Create images subfolder if saving frames
        if save_frames:
            images_dir = self.output_dir / "images" / f"episode_{episode_id[:8]}"
            images_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Saving individual frames to: {images_dir}")
        
        print(f"🎬 Rendering episode {episode_id[:8]}... ({num_steps} steps)")
        
        # Debug: Check episode data structure
        if num_steps == 0:
            raise ValueError(f"Episode {episode_id} has no samples")
        
        print(f"DEBUG: First sample keys: {list(episode_samples[0].keys())}")
        print(f"DEBUG: First sample sa_action: {episode_samples[0]['sa_action']}")
        print(f"DEBUG: SA action type: {type(episode_samples[0]['sa_action'])}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        
        # Define subplot layout: Large images on top, smaller action plots below
        # 5 rows: 2 for large images, 1 for reward plot, 2 for action plots
        gs = fig.add_gridspec(5, 6, height_ratios=[3, 3, 1.5, 1, 1], width_ratios=[1, 1, 1, 1, 1, 1], 
                             hspace=0.4, wspace=0.3)
        
        # Large linear scale observation image (top left, spans 3 columns)
        ax_obs_linear = fig.add_subplot(gs[0:2, :3])
        
        # Large log scale observation image (top right, spans 3 columns)  
        ax_obs_log = fig.add_subplot(gs[0:2, 3:])
        
        # Reward and temperature progression (third row, spans all columns)
        ax_reward = fig.add_subplot(gs[2, :])
        # Create temperature axis once outside animation function to avoid stacking
        ax_temp = ax_reward.twinx()
        
        # SA actions and Perfect actions (fourth row, smaller)
        ax_sa = fig.add_subplot(gs[3, :3])
        ax_perfect = fig.add_subplot(gs[3, 3:])
        
        # Incremental actions (bottom row, smaller)
        ax_inc_sa = fig.add_subplot(gs[4, :3])
        ax_inc_perfect = fig.add_subplot(gs[4, 3:])
        
        # Prepare data for plotting
        steps = list(range(num_steps))
        rewards = [sample['reward'] for sample in episode_samples]
        temperatures = [sample['temperature'] for sample in episode_samples]
        
        print(f"DEBUG: About to get action_dim from sa_action")
        print(f"DEBUG: episode_samples[0]['sa_action'] = {episode_samples[0]['sa_action']}")
        print(f"DEBUG: len(episode_samples[0]['sa_action']) = {len(episode_samples[0]['sa_action'])}")
        
        action_dim = len(episode_samples[0]['sa_action'])
        
        print(f"DEBUG: action_dim = {action_dim}")
        
        # Create colorbar axes once outside animation function to avoid stacking
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider_linear = make_axes_locatable(ax_obs_linear)
        cax_linear = divider_linear.append_axes("right", size="3%", pad=0.1)
        divider_log = make_axes_locatable(ax_obs_log)
        cax_log = divider_log.append_axes("right", size="3%", pad=0.1)
        
        # Store colorbar references for reuse
        cbar_linear_ref = [None]  # Use list to allow modification in nested function
        cbar_log_ref = [None]
        
        # Animation function
        def animate(frame):
            # Clear all axes except colorbar axes
            for ax in [ax_obs_linear, ax_obs_log, ax_reward, ax_temp, ax_sa, ax_perfect, ax_inc_sa, ax_inc_perfect]:
                ax.clear()
            # Clear colorbar axes too
            cax_linear.clear()
            cax_log.clear()
            
            current_sample = episode_samples[frame]
            
            # 1. Large linear scale observation image
            obs = current_sample['observation']
            
            # Handle multi-channel observations
            if len(obs.shape) == 3:
                if obs.shape[0] == 1:
                    obs = obs[0]  # Remove channel dimension if present
                elif obs.shape[0] == 2:
                    # Two channels - compute magnitude (assuming complex or dual measurements)
                    obs = np.sqrt(obs[0]**2 + obs[1]**2)
                else:
                    # Multiple channels - take first channel
                    obs = obs[0]
            
            im_linear = ax_obs_linear.imshow(obs, cmap='viridis', vmin=0, vmax=65535, interpolation='nearest')
            
            # Build title with optimization step if available
            title_parts = [f"Observation (Linear Scale) - Episode Step {current_sample['episode_step']}"]
            if current_sample['optimization_step'] is not None:
                title_parts.append(f"Optimization Step {current_sample['optimization_step']}")
            title = " | ".join(title_parts)
            
            ax_obs_linear.set_title(title, fontsize=14)
            
            # Add pixel count axes with tick labels every 8 pixels
            height, width = obs.shape
            ax_obs_linear.set_xlabel('X (pixels)', fontsize=12)
            ax_obs_linear.set_ylabel('Y (pixels)', fontsize=12)
            # Set tick positions every 8 pixels
            x_ticks = range(0, width, 8)
            y_ticks = range(0, height, 8)
            ax_obs_linear.set_xticks(x_ticks)
            ax_obs_linear.set_yticks(y_ticks)
            ax_obs_linear.set_xticklabels(x_ticks, fontsize=6)
            ax_obs_linear.set_yticklabels(y_ticks, fontsize=6)
            
            # Add gridlines every 4 pixels
            for i in range(0, width, 4):
                ax_obs_linear.axvline(x=i, color='white', alpha=0.3, linewidth=0.5)
            for i in range(0, height, 4):
                ax_obs_linear.axhline(y=i, color='white', alpha=0.3, linewidth=0.5)
            
            # Add center region boxes (256px, 224px, and 128px)
            center_x, center_y = width // 2, height // 2
            # 256px box (128px radius from center)
            box_256 = plt.Rectangle((center_x - 128, center_y - 128), 256, 256, 
                                   fill=False, edgecolor='cyan', linewidth=2, alpha=0.8)
            ax_obs_linear.add_patch(box_256)
            # 224px box (112px radius from center)
            box_224 = plt.Rectangle((center_x - 112, center_y - 112), 224, 224, 
                                   fill=False, edgecolor='yellow', linewidth=2, alpha=0.8)
            ax_obs_linear.add_patch(box_224)
            # 128px box (64px radius from center)
            box_128 = plt.Rectangle((center_x - 64, center_y - 64), 128, 128, 
                                   fill=False, edgecolor='magenta', linewidth=2, alpha=0.8)
            ax_obs_linear.add_patch(box_128)
            # Add center point dot
            ax_obs_linear.plot(center_x, center_y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
            
            # Add colorbar for linear image (reuse existing axes)
            cbar_linear_ref[0] = plt.colorbar(im_linear, cax=cax_linear)
            cbar_linear_ref[0].set_label('Intensity (counts)', fontsize=10)
            
            # 2. Large log scale observation image
            # Add small epsilon to avoid log(0)
            obs_log = np.log10(obs + 1)
            im_log = ax_obs_log.imshow(obs_log, cmap='plasma', interpolation='nearest')
            ax_obs_log.set_title(f"Observation (Log10 Scale) - Step {current_sample['episode_step']}", fontsize=14)
            
            # Add pixel count axes for log image too with tick labels every 8 pixels
            ax_obs_log.set_xlabel('X (pixels)', fontsize=12)
            ax_obs_log.set_ylabel('Y (pixels)', fontsize=12)
            # Set tick positions every 8 pixels (same as linear image)
            ax_obs_log.set_xticks(x_ticks)
            ax_obs_log.set_yticks(y_ticks)
            ax_obs_log.set_xticklabels(x_ticks, fontsize=6)
            ax_obs_log.set_yticklabels(y_ticks, fontsize=6)
            
            # Add gridlines every 4 pixels to log image too
            for i in range(0, width, 4):
                ax_obs_log.axvline(x=i, color='white', alpha=0.3, linewidth=0.5)
            for i in range(0, height, 4):
                ax_obs_log.axhline(y=i, color='white', alpha=0.3, linewidth=0.5)
            
            # Add center region boxes to log image
            box_256_log = plt.Rectangle((center_x - 128, center_y - 128), 256, 256, 
                                       fill=False, edgecolor='cyan', linewidth=2, alpha=0.8)
            ax_obs_log.add_patch(box_256_log)
            box_224_log = plt.Rectangle((center_x - 112, center_y - 112), 224, 224, 
                                       fill=False, edgecolor='yellow', linewidth=2, alpha=0.8)
            ax_obs_log.add_patch(box_224_log)
            box_128_log = plt.Rectangle((center_x - 64, center_y - 64), 128, 128, 
                                       fill=False, edgecolor='magenta', linewidth=2, alpha=0.8)
            ax_obs_log.add_patch(box_128_log)
            # Add center point dot to log image
            ax_obs_log.plot(center_x, center_y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
            
            # Add colorbar for log image (reuse existing axes)
            cbar_log_ref[0] = plt.colorbar(im_log, cax=cax_log)
            cbar_log_ref[0].set_label('Log10(Intensity + 1)', fontsize=10)
            
            # 3. Reward and temperature progression
            current_steps = steps[:frame+1]
            current_rewards = rewards[:frame+1]
            current_temps = temperatures[:frame+1]
            
            # Plot rewards
            ax_reward.plot(current_steps, current_rewards, 'b-o', label='Reward', linewidth=3, markersize=6)
            ax_reward.set_xlabel('Episode Step', fontsize=12)
            ax_reward.set_ylabel('Reward', color='b', fontsize=12)
            ax_reward.tick_params(axis='y', labelcolor='b')
            ax_reward.grid(True, alpha=0.3)
            
            # Add temperature on secondary axis (reuse existing axis)
            if len(current_temps) > 0:  # Make sure we have temperature data
                ax_temp.plot(current_steps, current_temps, 'r--s', label='Temperature', alpha=0.8, markersize=4, linewidth=2)
                ax_temp.set_ylabel('Temperature', color='r', fontsize=12)
                ax_temp.tick_params(axis='y', labelcolor='r')
                # Set temperature scale to be meaningful (handle NaN/Inf values)
                valid_temps = [t for t in current_temps if np.isfinite(t)]
                if len(valid_temps) > 0 and max(valid_temps) > min(valid_temps):
                    ax_temp.set_ylim(0, max(valid_temps) * 1.1)
            
            # Build progress title with optimization step if available
            progress_parts = [f"Reward={current_sample['reward']:.4f}"]
            progress_parts.append(f"Temperature={current_sample['temperature']:.3f}")
            if current_sample['optimization_step'] is not None:
                progress_parts.append(f"Opt.Step={current_sample['optimization_step']}")
            progress_title = "Progress: " + ", ".join(progress_parts)
            
            ax_reward.set_title(progress_title, fontsize=13)
            
            # 4. SA Actions (smaller) - Calculate error metrics relative to perfect action
            sa_action = current_sample['sa_action']
            perfect_action = current_sample['perfect_action']
            action_indices = list(range(action_dim))
            
            # Calculate MSE and MAE for SA action relative to perfect action
            action_error = sa_action - perfect_action
            sa_mse = np.mean(action_error**2)
            sa_mae = np.mean(np.abs(action_error))
            
            bars_sa = ax_sa.bar(action_indices, sa_action, alpha=0.7, color='blue', width=0.8)
            ax_sa.set_title(f'SA Action (MSE: {sa_mse:.3f}, MAE: {sa_mae:.3f})', fontsize=11)
            ax_sa.set_xlabel('Action Dimension', fontsize=10)
            ax_sa.set_ylabel('Value', fontsize=10)
            ax_sa.set_ylim(-1, 1)
            ax_sa.grid(True, alpha=0.3)
            ax_sa.tick_params(labelsize=9)
            
            # 5. Perfect Actions (smaller)
            bars_perfect = ax_perfect.bar(action_indices, perfect_action, alpha=0.7, color='green', width=0.8)
            ax_perfect.set_title('Perfect Action', fontsize=11)
            ax_perfect.set_xlabel('Action Dimension', fontsize=10)
            ax_perfect.set_ylabel('Value', fontsize=10)
            ax_perfect.set_ylim(-1, 1)
            ax_perfect.grid(True, alpha=0.3)
            ax_perfect.tick_params(labelsize=9)
            
            # 6. SA Incremental Actions (smaller) - Calculate error metrics relative to zero
            sa_inc = current_sample['sa_incremental_action']
            sa_inc_mse = np.mean(sa_inc**2)  # MSE relative to zero
            sa_inc_mae = np.mean(np.abs(sa_inc))  # MAE relative to zero
            
            bars_sa_inc = ax_inc_sa.bar(action_indices, sa_inc, alpha=0.7, color='cyan', width=0.8)
            ax_inc_sa.set_title(f'SA Incremental Action (MSE: {sa_inc_mse:.3f}, MAE: {sa_inc_mae:.3f})', fontsize=11)
            ax_inc_sa.set_xlabel('Action Dimension', fontsize=10)
            ax_inc_sa.set_ylabel('Δ Value', fontsize=10)
            ax_inc_sa.set_ylim(-2, 2)
            ax_inc_sa.grid(True, alpha=0.3)
            ax_inc_sa.tick_params(labelsize=9)
            
            # 7. Perfect Incremental Actions (smaller) - Calculate error metrics relative to zero
            perfect_inc = current_sample['perfect_incremental_action']
            perfect_inc_mse = np.mean(perfect_inc**2)  # MSE relative to zero
            perfect_inc_mae = np.mean(np.abs(perfect_inc))  # MAE relative to zero
            
            bars_perfect_inc = ax_inc_perfect.bar(action_indices, perfect_inc, alpha=0.7, color='orange', width=0.8)
            ax_inc_perfect.set_title(f'Perfect Incremental Action (MSE: {perfect_inc_mse:.3f}, MAE: {perfect_inc_mae:.3f})', fontsize=11)
            ax_inc_perfect.set_xlabel('Action Dimension', fontsize=10)
            ax_inc_perfect.set_ylabel('Δ Value', fontsize=10)
            ax_inc_perfect.set_ylim(-2, 2)
            ax_inc_perfect.grid(True, alpha=0.3)
            ax_inc_perfect.tick_params(labelsize=9)
            
            # Add episode info as text (positioned to not overlap with large images)
            info_text = (f"Episode: {episode_id[:8]}...\n"
                        f"Step: {current_sample['episode_step']}\n"
                        f"Image Size: {width}×{height} px\n"
                        f"Temperature: {current_sample['temperature']:.4f}\n"
                        f"Cost Δ: {current_sample['cost_delta']:.4f}\n"
                        f"Accepted: {current_sample.get('accepted', True)}")
            
            fig.text(0.01, 0.99, info_text, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
            
            # Save frame as individual image if requested
            if save_frames:
                frame_filename = f"frame_{frame:04d}_step_{current_sample['episode_step']:04d}.png"
                frame_path = images_dir / frame_filename
                fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        
        # Create list of frame indices to render based on interval
        frame_indices = list(range(0, num_steps, render_interval))
        num_rendered_frames = len(frame_indices)
        
        print(f"DEBUG: Rendering {num_rendered_frames} frames (every {render_interval} frames from {num_steps} total)")
        
        # Modified animate function that uses frame indices
        def animate_with_interval(frame_idx):
            return animate(frame_indices[frame_idx])
        
        # Create animation
        print(f"DEBUG: Creating FuncAnimation with {num_rendered_frames} frames")
        try:
            anim = animation.FuncAnimation(fig, animate_with_interval, frames=num_rendered_frames, interval=1000//fps, repeat=True)
            print(f"DEBUG: FuncAnimation created successfully")
        except Exception as e:
            print(f"DEBUG: Error creating FuncAnimation: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Save as GIF
        if render_interval == 1:
            gif_filename = f"episode_{episode_id[:8]}_{num_steps}steps.gif"
        else:
            gif_filename = f"episode_{episode_id[:8]}_{num_steps}steps_every{render_interval}.gif"
        gif_path = self.output_dir / gif_filename
        
        print(f"💾 Saving GIF to {gif_path}")
        try:
            anim.save(gif_path, writer='pillow', fps=fps)
            print(f"DEBUG: GIF saved successfully")
        except Exception as e:
            print(f"DEBUG: Error saving GIF: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        plt.close(fig)
        
        return str(gif_path)
    
    def render_multiple_episodes(self, episode_ids: Optional[List[str]] = None, 
                                num_episodes: Optional[int] = None, fps: int = 2, render_interval: int = 1,
                                max_transitions: Optional[int] = None, save_frames: bool = False) -> List[str]:
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
                gif_path = self.render_episode_gif(episode_id, fps=fps, render_interval=render_interval,
                                                   max_transitions=max_transitions, save_frames=save_frames)
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
    parser.add_argument("--render-interval", type=int, default=1,
                       help="Render every Nth frame (default: 1, render all frames)")
    parser.add_argument("--max-transitions", type=int, default=None,
                       help="Maximum number of transitions to render per episode (default: None, render all)")
    parser.add_argument("--render-images", action="store_true",
                       help="Save individual frame images to images/ subfolder during rendering")
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
            fps=args.fps,
            render_interval=args.render_interval,
            max_transitions=args.max_transitions,
            save_frames=args.render_images
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
