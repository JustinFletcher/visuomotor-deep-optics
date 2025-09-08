#!/usr/bin/env python3
"""
Simple test to launch one SA job and see if it works.
"""

import json
import subprocess
from pathlib import Path

def launch_test_job():
    """Launch a single test SA job."""
    print("🧪 Testing SA job launch...")
    
        # Load job configuration (path relative to project root)
    import os
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "job_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Build command for dataset_test (relative to project root)
    cmd = config['job_command'].copy()
    dataset_test_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset_test")
    cmd.append(f'--dataset_save_path={dataset_test_path}/')
    
    print(f"Launching command:")
    print(' '.join(cmd))
    print()
    
    try:
        # Launch the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        print(f"✅ Job launched successfully!")
        print(f"   PID: {process.pid}")
        print(f"   Command: {' '.join(cmd[:4])}...")
        print(f"   Dataset: {dataset_test_path}/")
        
        # Wait a few seconds to see if it starts properly
        import time
        time.sleep(5)
        
        if process.poll() is None:
            print(f"✅ Process still running after 5 seconds - looks good!")
            print(f"   You can check progress with: ls -la dataset_test/")
            print(f"   To stop: kill {process.pid}")
        else:
            print(f"❌ Process exited early with code: {process.returncode}")
            stdout, stderr = process.communicate()
            if stdout:
                print(f"STDOUT: {stdout.decode()}")
            if stderr:
                print(f"STDERR: {stderr.decode()}")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to launch job: {e}")
        return None

if __name__ == "__main__":
    launch_test_job()
