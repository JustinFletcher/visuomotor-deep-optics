# Coral HPC Access Guide

## Authentication and Login

### Step 1: Kerberos Authentication
```bash
/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL
```

### Step 2: SSH to Coral
```bash
/usr/local/ossh/bin/ssh fletch@coral.mhpcc.hpc.mil
```

### Combined Login Script
```bash
/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL && /usr/local/ossh/bin/ssh fletch@coral.mhpcc.hpc.mil
```

### 🤖 AI Assistant Note: Background Execution
**Important for AI tools**: When using `run_in_terminal` for interactive commands like SSH:
- Use `isBackground=true` to prevent blocking the AI interpreter
- This allows continued interaction while maintaining the SSH session
- Essential for commands that wait for user input or maintain persistent connections
- Example: SSH, interactive shells, long-running processes

## SLURM Job Configuration

### Account Information
- **Account**: `MHPCC96650DAS`
- **Partition**: `standard`

### GPU Resources
- **GPU Type**: A100-PCIe
- **Max GPUs per node**: 8
- **Standard allocation**: `--gres=gpu:A100-PCIe:8`

### Standard SLURM Headers
```bash
#!/bin/bash
#SBATCH --job-name=your_job_name        # Job name
#SBATCH --partition=standard            # Partition or queue name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=16                     # Total number of tasks
#SBATCH --ntasks-per-node=16            # Number of tasks per node
#SBATCH --account=MHPCC96650DAS         # Account
#SBATCH --output=./job.out              # Standard output
#SBATCH --error=./job.err               # Standard error
#SBATCH --gres=gpu:A100-PCIe:8          # GPU allocation (8 A100s)
#SBATCH --time=7-00:00:00               # Max runtime (7 days)
```

### Common Resource Requests

#### Full Node with 8 GPUs (Multi-GPU Training)
```bash
#SBATCH --gres=gpu:A100-PCIe:8
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
```

#### Single GPU
```bash
#SBATCH --gres=gpu:A100-PCIe:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
```

#### High Memory
```bash
#SBATCH --mem=1764642                   # ~1.7TB memory
```

## Interactive Sessions

### Interactive Shell with Full GPU Node
```bash
srun --time=08:00:00 --account=MHPCC96650DAS --gres=gpu:A100-PCIe:8 --partition=standard --pty bash -i
```

### Interactive Shell with Single GPU
```bash
srun --time=08:00:00 --account=MHPCC96650DAS --gres=gpu:A100-PCIe:1 --partition=standard --pty bash -i
```

### Interactive Shell CPU Only
```bash
srun --time=08:00:00 --account=MHPCC96650DAS --ntasks-per-node=1 --partition=standard --pty bash -i
```

## File Transfer (rsync)

### From Coral to Local

#### Continuous Sync of Training Runs
```bash
# Authenticate first
/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL

# Continuous sync loop
while :; do 
    clear
    rsync -chavzP --stats fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/runs/ /Users/fletcher/research/visuomotor-deep-optics/runs
    sleep 5
done
```

#### Sync Rollouts
```bash
/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL
rsync -chavzP --stats fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/rollouts /Users/fletcher/research/visuomotor-deep-optics/rollouts/remote
```

#### Sync Saved Models
```bash
/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL
rsync -chavzP --stats fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/saved_models/ /Users/fletcher/research/visuomotor-deep-optics/saved_models/
```

### From Local to Coral

#### Upload Saved Models
```bash
/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL
rsync -chavzP --stats /Users/fletcher/research/visuomotor-deep-optics/saved_models/* fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/saved_models/
```

#### Upload Code Changes
```bash
/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL
rsync -chavzP --stats --exclude='__pycache__' --exclude='*.pyc' --exclude='runs/' --exclude='rollouts/' --exclude='datasets/' /Users/fletcher/research/visuomotor-deep-optics/ fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/
```

## Workspace Structure

### Coral Paths
- **Home**: `/wdata/home/fletch/`
- **Project**: `/wdata/home/fletch/visuomotor-deep-optics/`
- **Runs**: `/wdata/home/fletch/visuomotor-deep-optics/runs/`
- **Models**: `/wdata/home/fletch/visuomotor-deep-optics/saved_models/`

### Local Paths
- **Project**: `/Users/fletcher/research/visuomotor-deep-optics/`
- **Runs**: `/Users/fletcher/research/visuomotor-deep-optics/runs/`
- **Models**: `/Users/fletcher/research/visuomotor-deep-optics/saved_models/`

## Monitoring and Debugging

### Check GPU Usage
```bash
nvidia-smi
```

### Check Job Status
```bash
squeue -u fletch
```

### Check Job Details
```bash
scontrol show job <job_id>
```

### Cancel Job
```bash
scancel <job_id>
```

### View Job Output (while running)
```bash
tail -f job.out
tail -f job.err
```

## Multi-GPU Training Considerations

### DataParallel-Ready SLURM Script
```bash
#!/bin/bash
#SBATCH --job-name=sml_training
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --account=MHPCC96650DAS
#SBATCH --output=./sml_training.out
#SBATCH --error=./sml_training.err
#SBATCH --gres=gpu:A100-PCIe:8          # Request all 8 GPUs
#SBATCH --time=24:00:00                 # 24 hour limit

cd ~/visuomotor-deep-optics/

# Set CUDA visible devices to use all GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run training with DataParallel support
poetry run python optomech/supervised_ml/train_sml_model.py \
    --dataset_path optomech/supervised_ml/datasets/sml_100k_dataset \
    --num_epochs 100 \
    --batch_size 32 \
    --device auto
```

### GPU Memory per A100
- **Total Memory**: ~40GB per A100-PCIe
- **Effective Memory**: ~38GB usable per GPU
- **8 GPUs Total**: ~304GB GPU memory available

## Performance Notes

### CPU Resources
- **28 cores** typically available for work per node
- **High memory nodes**: Up to ~1.7TB RAM available

### Network Performance
- rsync can be slow over long distances
- Use compression (`-z`) for better transfer rates
- Consider `--bwlimit` for bandwidth control during active work

## Common Issues and Solutions

### Authentication Timeout
```bash
# Re-authenticate if connection drops
/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL
```

### Job Won't Start
```bash
# Check queue status
squeue -p standard
# Check available resources
sinfo -p standard
```

### Out of Memory
- Reduce batch size for GPU OOM
- Request more memory with `--mem=` for CPU OOM
- Use gradient accumulation instead of larger batches

### Multi-GPU Issues
- Ensure `CUDA_VISIBLE_DEVICES` is set correctly
- Verify DataParallel is working: check GPU utilization with `nvidia-smi`
- Scale batch size appropriately for multi-GPU

## Dataset Generation Commands

### Large-Scale Dataset Creation (100K samples)

Once you have both terminal panes set up, you can run:

**In Pane 1 (Dataset Generation):**
```bash
cd ~/visuomotor-deep-optics && poetry run python optomech/supervised_ml/sml_job_manager_simple.py --config sml_job_config.json --total_samples 100000
```

**In Pane 2 (Progress Monitoring):**
```bash
cd ~/visuomotor-deep-optics && poetry run python optomech/supervised_ml/sml_job_watcher.py --dataset_dir datasets/sml_100k_dataset
```

### Alternative Commands (without poetry)
```bash
# Pane 1
cd ~/visuomotor-deep-optics && python optomech/supervised_ml/sml_job_manager_simple.py --config sml_job_config.json --total_samples 100000

# Pane 2  
cd ~/visuomotor-deep-optics && python optomech/supervised_ml/sml_job_watcher.py --dataset_dir datasets/sml_100k_dataset
```

## Quick Reference Commands

```bash
# Login
/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL && /usr/local/ossh/bin/ssh fletch@coral.mhpcc.hpc.mil

# Interactive GPU session
srun --time=08:00:00 --account=MHPCC96650DAS --gres=gpu:A100-PCIe:8 --partition=standard --pty bash -i

# Submit job
sbatch your_script.slurm

# Check status
squeue -u fletch

# Sync results
/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL && rsync -chavzP --stats fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/runs/ ./runs/
```