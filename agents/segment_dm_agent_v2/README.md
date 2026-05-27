# segment_dm_agent_v2

15-phase composite agent transfer-learned from the bootstrap agent
to add a 32-mode Zernike DM head. Trained under a random Kolmogorov
atmosphere (r0 sampled uniformly from 10-25 cm per episode).

Native policy action per phase: [seg_PTT_45 | zernike_32] (77 dim).
Env-facing action: [seg_PTT_45 | dm_1225] (1270 dim), via the
SegmentZernikePassthroughAdapter (matches training-time
SegmentZernikeDMVectorEnv math: Zernike -> DM projection M, scale
by 0.01, clamp to [-1, 1]).

To roll out:

    poetry run python train/ppo/rollout_elf_bootstrap_ptt.py \
        --policy-spec agents/segment_dm_agent_v2/composed.yaml \
        --num-episodes 4 \
        --env-kwarg 'atmosphere={r0_total_m: 0.25, L0_m: 25.0, static: true}'
