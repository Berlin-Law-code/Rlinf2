# LBench environment adapter

`LBenchEnv` keeps RLinf/OpenPI and Isaac Sim in separate Python environments.
Each local environment is an external LBench worker connected through a
private Unix-domain socket; the RLinf process imports only the transport and
mapping modules, never Isaac, CUDA, PhysX, or Omniverse.

Required variables for a real evaluation:

```bash
export RLINF_PYTHON=/path/to/rlinf-env/bin/python
export LBENCH_ROOT=/path/to/LBench_multi_env
export LBENCH_ISAAC_PYTHON=/path/to/isaac-env/bin/python
export LBENCH_TASK_CONFIG=/path/to/task_config.yaml
export LBENCH_NORM_STATS=/path/to/norm_stats.json
export LBENCH_SFT_CHECKPOINT=/path/to/checkpoints/global_step_30000
export LBENCH_ISAAC_GPUS=4
bash evaluations/run_eval.sh lbench lbench_openpi_pi05_eval \
  env.eval.isaac_cuda_visible_devices="[$LBENCH_ISAAC_GPUS]"
```

`LBENCH_ISAAC_PYTHON` must point to a different environment from
`RLINF_PYTHON`. Its launcher path is intentionally kept without resolving
venv symlinks. The external worker puts that interpreter's `bin` first on
`PATH`, disables the user site, sets its own `PYTHONPATH` to `LBENCH_ROOT`, and
does not inherit the RLinf virtual environment's `PYTHONHOME`, `VIRTUAL_ENV`,
or Conda-prefix markers. Packages needed by Isaac must therefore be installed
in the Isaac environment itself. System driver/CUDA variables remain shared
and GPU isolation is controlled separately with `LBENCH_ISAAC_GPUS`.

## Pi0.5 SFT and PPO contract

`pi05_lbench_joint` uses three cameras (`obs_camera`, `obs_camera_2`, and
`realsense`), an 8-D state, an 8-D absolute joint target, and model horizon 50.
SFT uses all 50 targets; eval/PPO execute the first 20 and then replan. Generate
the one normalization file shared by all phases before training:

```bash
export LBENCH_LEROBOT_REPO=/path/to/lbench_lerobot_v21
export LBENCH_NORM_DIR=/path/to/openpi_assets/lbench_joint
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_lbench_joint \
  --repo-id "$LBENCH_LEROBOT_REPO" \
  --output-dir "$LBENCH_NORM_DIR"
export LBENCH_NORM_STATS=$LBENCH_NORM_DIR/norm_stats.json
python toolkits/lbench_data_dry_run.py \
  --repo-id "$LBENCH_LEROBOT_REPO" --norm-stats "$LBENCH_NORM_STATS"
```

Use the exact `Writing stats to:` path printed by the command. The SFT worker
copies that file into both the actor directory and its parent
checkpoint under the canonical asset id, and writes `openpi_assets.json`.
Consequently either checkpoint root is directly readable by RLinf eval/PPO.

The current `rlinf-openpi==0.1.1` installation uses `lerobot==0.3.3`, so its
training dataset must use the LeRobot v2.1 physical layout (`tasks.jsonl`,
`episodes.jsonl`, and per-episode parquet files). Collector option
`lbench_joint` selects only the LBench feature names; historical `lbench_v3`
remains an alias and does not force a v3.0 physical layout.

The base SFT config is full-parameter. For a reduced-memory, single-GPU
feasibility run, add `actor.model.openpi.train_expert_only=true`,
`actor.micro_batch_size=1`, and `actor.global_batch_size=1`. This freezes the
PaliGemma VLM but still trains the action expert/projections. The M8 SFT smoke
adds that override automatically. LBench PPO also freezes the VLM, matching
RLinf's Pi0.5 PPO baseline; it is not expected to fit end-to-end with parallel
Isaac workers on one 32 GB GPU.

```bash
export PI05_BASE_CHECKPOINT=/path/to/openpi_pi05_base_pytorch
python examples/sft/train_vla_sft.py --config-name lbench_sft_openpi_pi05

export LBENCH_SFT_CHECKPOINT=/path/to/checkpoints/global_step_30000
python examples/embodiment/train_embodied_agent.py \
  --config-name lbench_ppo_openpi_pi05
```

The PPO environment has `auto_reset=false` and `ignore_terminations=false`.
Its per-subaction mask includes the terminal action and excludes every later
slot. `loss_mask_sum` gives each scheduled environment equal total loss weight;
combined with global round-robin assignment, this is task-balanced when each
task has one or the same number of environment slots.

The external-worker step protocol also reports `executed_action` and
`transition_executed`. This matters when a recovery controller replaces the
policy command: episode collectors store the recovered 8-D joint action, while
PPO masks that off-policy sub-action from the policy loss. Control-only rescue
notifications that do not advance `world.step` are excluded from episode
lengths, rewards, PPO masks, and LeRobot/pickle frames. `transition_mask`
therefore means “a physical state transition exists”; `action_mask` is stricter
and means “the policy action itself produced that transition.” Older LBench
workers without these fields retain the historical normal-step behavior, but
recovery-enabled runs require the updated protocol.

Evaluation also uses `auto_reset=false`: one environment slot contributes one
episode per `rollout_epoch`, so short tasks cannot produce more samples than
long tasks. Episode counters, seeds, and ids advance across explicit rollout
resets. `04_eight_env_eval.sh --eval-rollout-epochs 8` therefore evaluates all
eight tasks once at each of seeds 0 through 7 (64 episodes total).

The default PPO recipe uses `libero_sparse_success`, which matches RLinf's
LIBERO reward contract: the first success transition receives
`algorithm.reward_coef` once, while progress, stages, explicit failures, and
timeouts receive zero reward. SimBox still terminates an explicitly failed task
and the aligned 1200-step horizon remains a truncation; reward parity must not
override native simulator safety or irrecoverability semantics. Alternative
`sparse_success`, `progress_delta`, and `backend` adapters remain available for
ablations. With `auto_reset=false`, RLinf does not bootstrap the final value
into a newly reset task. Primitive rewards are summed over each executed
20-action chunk before chunk-level GAE.

Updated workers also report `success_confirmed`. When present, this boolean is
the authoritative success signal for rewards and episode metrics: a raw
`episode_sr=1` or `reward_components.success=1` is only a candidate until the
configured success hold completes. RLinf falls back to those legacy metrics
only when an older worker omits `success_confirmed`. The complete step info,
including this field, is retained in collected `lbench_metadata.step_info`.

LeRobot policies and RLinf/OpenPI policies share the LBench observation/action
contract, but their checkpoints are **not** declared interchangeable. A
checkpoint converter plus numerical parity test is required before using a
LeRobot-trained checkpoint as an OpenPI initialization.

For a clean OpenPI initialization, download
`gs://openpi-assets/checkpoints/pi05_base` with
`openpi.shared.download.maybe_download` and convert it with
`rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py --config-name
pi05_lbench_joint`. The complete AutoDL command sequence is in
`docs/LBENCH_MULTI_ENV_RUNBOOK_ZH.md`.

## M8 remote acceptance

Keep the model and simulator GPU groups disjoint. The launcher assigns Isaac
GPU ids round-robin to its isolated subprocesses. It also rejects a previously
started Ray cluster by default, because Ray captures GPU visibility when it is
started. Let the launcher create its local Ray instance, or start Ray yourself
with only the RLinf GPUs visible and pass `--allow-existing-ray` after checking
the placement.

```bash
# Run these from the RLinf environment. `ray stop` is unnecessary when no Ray
# service is active.
"$RLINF_PYTHON" -m ray stop

RLINF_GPUS=0 \
  scripts/lbench/remote/01_sft_smoke.sh --dry-run
RLINF_GPUS=0 LBENCH_ISAAC_GPUS=4 \
  scripts/lbench/remote/02_single_env_eval.sh --dry-run
RLINF_GPUS=0,1 LBENCH_ISAAC_GPUS=4,5 \
  scripts/lbench/remote/03_dual_env_ppo_smoke.sh --dry-run
RLINF_GPUS=0,1,2,3 LBENCH_ISAAC_GPUS=4,5,6,7 \
  scripts/lbench/remote/04_eight_env_eval.sh --dry-run
RLINF_GPUS=0,1,2,3 LBENCH_ISAAC_GPUS=4,5,6,7 \
  scripts/lbench/remote/05_eight_env_ppo.sh --dry-run
```

The dual-environment PPO smoke stage deliberately aligns the simulator,
episode, and rollout horizons at 40 primitive steps. This forces both partial
task attempts to truncate and flush through the collection path. Its rewards,
success rate, and collected trajectories are acceptance artifacts only; the
formal eight-environment PPO stage restores the 1200-step task horizon.

Remove `--dry-run` only on a suitable remote host. The stages respectively
perform: five SFT updates, checkpoint reload, and one resumed update; fixed
task/seed single-env evaluation with episode recording; two global PPO steps
with two envs and `rollout_epoch=1`; fixed-seed eight-env evaluation with
episode recording; and 200 global steps of eight-env balanced PPO.

The number of evaluation/training environments must be divisible by the
number of RLinf GPUs because `env`, `rollout`, and `actor` use collocated
one-process-per-GPU placement. The commands above therefore use one RLinf GPU
for one environment, two for the two-environment smoke, and four for eight
environments. Isaac GPU ids are physical ids and remain visible only to the
external Isaac subprocesses.

For an existing single-node Ray service, the equivalent safe startup is:

```bash
CUDA_VISIBLE_DEVICES="$RLINF_GPUS" "$RLINF_PYTHON" -m ray start --head
scripts/lbench/remote/05_eight_env_ppo.sh --allow-existing-ray --dry-run
```

Do not use `--allow-existing-ray` for a cluster that was started while the
Isaac GPUs were visible as Ray resources.

Run the CPU-only protocol/adapter check from the RLinf environment:

```bash
python toolkits/lbench_env_dry_run.py --lbench-root "$LBENCH_ROOT" --num-envs 2
```

Set `env.eval.data_collection.enabled=true` to use RLinf's existing episode
collector. Pickle records retain the raw `lbench_metadata` and use
rank-qualified filenames; LeRobot export uses `rank_<rank>/id_<shard>`
directories. A real OpenPI checkpoint and Isaac workers are not part of this
dry run.
