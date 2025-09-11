## Coral Workflows

Login to Coral and check system resources.
```
/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL
/usr/local/ossh/bin/ssh fletch@coral.mhpcc.hpc.mil
```

Grab one full node for one hour.
```
srun --time=08:00:00 --account=MHPCC96650DAS --gres=gpu:A100-PCIe:8 --partition=standard --pty bash -i
```

```
srun --time=08:00:00 --account=MHPCC96650DAS  --ntasks-per-node=1 --partition=standard --pty bash -i
```

```
srun --time=08:00:00 --mem=1764642 --account=MHPCC96650DAS --gres=gpu:A100-PCIe:8 --partition=standard --pty bash -i
```


Run scp from your own machine to pull the tensorboard records down.

while :; do clear; 
/usr/local/watch nvidiarb5/bin/pkinit fletch@HPCMP.HPC.MIL;
rsync -chavzP --stats fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/runs /Users/fletcher/research/visuomotor-deep-optics/runs
; sleep 5; done


## Workspace
watch -n 30 rsync -avhze ssh fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/runs /Users/fletcher/research/visuomotor-deep-optics/deep-optics-gym


while :; do clear; rsync -avhze ssh fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/runs /Users/fletcher/research/visuomotor-deep-optics/deep-optics-gym
; sleep 10; done

while :; do clear; rsync ssh fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/runs /Users/fletcher/research/visuomotor-deep-optics/runs; sleep 5; done


poetry run python ./optomech/rollout.py \
--model_path=/wdata/home/fletch/visuomotor-deep-optics/runs/optomech-v1__optomech-reward-exp-tau__222222__1742683939/eval_optomech-reward-exp-tau_2300000/optomech-reward-exp-tau_2300000_policy.pt \
--env_vars_path=/wdata/home/fletch/visuomotor-deep-optics/runs/optomech-v1__optomech-reward-exp-tau__222222__1742683939/args.json \
--write_env_state_info=True \
--record_env_state_info=True \
--eval_save_path=./rollouts/ \
--num_episodes=1 \
--seed=746464

# From coral to local
## Runs


/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL;
while :; do clear; 
rsync -chavzP --stats fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/runs/ /Users/fletcher/research/visuomotor-deep-optics/runs
; sleep 5; done

## Rollouts
/usr/local/krb5/bin/pkinit  fletch@HPCMP.HPC.MIL;
rsync -chavzP --stats fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/rollouts /Users/fletcher/research/visuomotor-deep-optics/rollouts/remote


# From local to coral
# Saved models
/usr/local/krb5/bin/pkinit  fletch@HPCMP.HPC.MIL;
rsync -chavzP --stats  /Users/fletcher/research/visuomotor-deep-optics/saved_models/* fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/saved_models/



/usr/local/krb5/bin/pkinit fletch@HPCMP.HPC.MIL;
rsync -chavzP --stats fletch@coral.mhpcc.hpc.mil:/wdata/home/fletch/visuomotor-deep-optics/runs /Users/fletcher/research/visuomotor-deep-optics/runs

28 cores are free for work.
Step times

sync, 1: 0.010
async, 1: 0.022
async, 2: 0.034
async, 2, n=16: 0.065


## Questions for Mark
- [ ] How can I expose a resource from the Coral login nodes? (TensorBoard)
- [ ] How can I best watch resource utilization on a particular node from login?
- [ ] What's your preferred approach to rsync? I've noticed it's a little slow.
- [ ] Can I get a dedicated reservation? 

 
nvidia-smi

ssh into active 


```
#SBATCH --constraint=intel
```