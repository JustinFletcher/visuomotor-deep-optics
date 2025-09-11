#!/usr/bin/env python3

import pickle
import glob
import numpy as np
import sys
from pathlib import Path

# Add the optomech directory to path for imports
optomech_dir = Path(__file__).parent / "optomech"
if str(optomech_dir) not in sys.path:
    sys.path.insert(0, str(optomech_dir))

# Import required modules to allow pickle loading
try:
    import hcipy
    from optomech import optomech
    print("Successfully imported required modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying to load pickle with reduced functionality...")

# Find the most recent rollout
rollout_dirs = list(Path("saved_models/rollout_results/epoch_001").glob("*"))
if rollout_dirs:
    episode_dir = rollout_dirs[0]
    print(f"Examining episode: {episode_dir}")
    
    # Find step files - use the same pattern as render_history.py
    step_files = sorted(episode_dir.glob("step_*.pkl"))
    print(f"Found {len(step_files)} step files")
    
    # Reproduce the exact logic from render_history.py
    print(f"\n=== Reproducing render_history.py surface_max logic ===")
    
    surface_min = None
    surface_max = None
    
    for step_info_pickle_filename in step_files:
        print(f"Processing: {step_info_pickle_filename.name}")
        
        try:
            with open(step_info_pickle_filename, 'rb') as f:
                a = pickle.load(f)
                
            # This is the exact check from render_history.py line 84
            print(f"  Checking: a['state_content'][0]['segmented_mirror_surfaces']")
            
            state_content = a['state_content']
            print(f"  state_content type: {type(state_content)}")
            print(f"  state_content length: {len(state_content)}")
            
            state_content_0 = state_content[0] 
            print(f"  state_content[0] type: {type(state_content_0)}")
            
            has_sms = 'segmented_mirror_surfaces' in state_content_0
            print(f"  Has segmented_mirror_surfaces: {has_sms}")
            
            if has_sms:
                sms = state_content_0['segmented_mirror_surfaces']
                print(f"  segmented_mirror_surfaces type: {type(sms)}")
                print(f"  segmented_mirror_surfaces length: {len(sms)}")
                print(f"  segmented_mirror_surfaces bool value: {bool(sms)}")
                
                # This is the exact condition from render_history.py
                if sms:
                    b = sms[0]
                    print(f"  Processing surface data...")
                    current_min = np.min(b)
                    current_max = np.max(b) 
                    print(f"  Current min: {current_min}, max: {current_max}")
                    
                    if surface_min is None or current_min < surface_min:
                        surface_min = current_min
                    if surface_max is None or current_max > surface_max:
                        surface_max = current_max
                        
                    print(f"  Updated surface_min: {surface_min}, surface_max: {surface_max}")
                else:
                    print(f"  segmented_mirror_surfaces evaluated to False!")
            else:
                print(f"  No segmented_mirror_surfaces key found!")
                
        except Exception as e:
            print(f"  ERROR processing {step_info_pickle_filename.name}: {e}")
            import traceback
            traceback.print_exc()
            
        # Only process first few files for debugging
        if step_info_pickle_filename.name > "step_005.pkl":
            break
            
    print(f"\n=== Final Results ===")
    print(f"surface_min: {surface_min}")
    print(f"surface_max: {surface_max}")
    
    if surface_max is not None:
        print(f"surface_max * 1e6 = {surface_max * 1e6}")
        print("SUCCESS: surface_max calculation should work")
    else:
        print("FAILURE: surface_max is None - this is the problem!")
            
else:
    print("No rollout directories found")
