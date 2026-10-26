#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch single-task PPO with explicit one- to four-GPU placement."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkits.lbench_remote_stage import (  # noqa: E402
    _common_checks,
    _csv,
    _existing_ray_cluster,
    _isaac_runtime_env,
    _visible_gpu_ids,
)
from toolkits.lbench_two_task_ppo import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    checkpoint_contract,
    run_foreground,
    validate_fsdp_precision,
)

CONFIG_NAME = "lbench_ppo_openpi_pi05_drawer_2gpu"
COLLOCATED_CONFIG_NAME = "lbench_ppo_openpi_pi05_drawer_3gpu_collocated"
SEPARATED_CONFIG_NAME = "lbench_ppo_openpi_pi05_drawer_3gpu_separated"
LIBERO_LONG_CONFIG_NAME = "lbench_ppo_openpi_pi05_drawer_4gpu_libero_long"
CLEAN_LIBERO_LONG_CONFIG_NAME = "lbench_ppo_openpi_pi05_clean_4gpu_libero_long"
LIBERO_LONG_PROBE_CONFIG_NAME = "lbench_ppo_openpi_pi05_drawer_3gpu_libero_long_probe"
DEFAULT_TASK = "store_objects_in_the_drawer_0"
CLEAN_TASK = "clean_the_dirt_with_browncloth_lighttable_part0"
FLOWER_TASK = "insert_the_flower_in_vase_part0"
FLOWER_SINGLE_CONFIG_NAME = "lbench_ppo_openpi_pi05_flower_1gpu"
FLOWER_FOUR_CONFIG_NAME = "lbench_ppo_openpi_pi05_flower_4gpu"
PACK_TASK = "pack_in_objects_franka"
PACK_FOUR_CONFIG_NAME = "lbench_ppo_openpi_pi05_pack_4gpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Expose a single task with isolated or collocated GPU placement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        choices=(
            "collocated_1gpu",
            "two_gpu",
            "collocated_3gpu",
            "separated_3gpu",
            "collocated_4gpu_libero_long",
            "libero_long_3gpu_probe",
        ),
        default="two_gpu",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Physical GPU IDs; defaults to consecutive IDs for the chosen layout",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Exact task name; the four-GPU clean task selects its 600-step profile",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--record-diagnostics",
        action="store_true",
        help="Save train/eval transitions, three-camera videos and PPO tensors",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="Completed PPO checkpoint to restore, including optimizer and RNG state",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="PPO iteration limit: 1000 for LIBERO-Long profiles, otherwise 500",
    )
    parser.add_argument(
        "--lr-schedule-steps",
        type=int,
        help="LR decay horizon; resume preserves the source horizon by default",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        help="LIBERO-Long profiles: override micro batch after measuring GPU memory",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        help="Override predicted action length; executed chunk length is unchanged",
    )
    parser.add_argument(
        "--denoise-steps",
        type=int,
        help="Use the same denoising step count for sampling and PPO recomputation",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        help="Positive exploration noise shared by rollout and PPO recomputation",
    )
    parser.add_argument(
        "--ignore-last",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Exclude the final denoising step from SDE exploration; explicit resume override",
    )
    parser.add_argument(
        "--clip-ratio-c",
        type=float,
        help="Enable PPO dual clipping with a finite coefficient greater than one",
    )
    parser.add_argument(
        "--value-head-hidden-sizes",
        type=int,
        nargs="+",
        help="Explicit Critic widths, e.g. 1024 512 256 for LIBERO-Long",
    )
    parser.add_argument(
        "--smoke-test-steps",
        type=int,
        help="Run this many additional rounds, preserving the configured LR horizon",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Fresh output directory, also when restoring a checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files/config only; report GPU availability without launching",
    )
    args = parser.parse_args(argv)
    four_gpu = args.layout == "collocated_4gpu_libero_long"
    probe = args.layout == "libero_long_3gpu_probe"
    single = args.layout == "collocated_1gpu"
    libero_long = four_gpu or probe or single
    if args.gpus is None:
        count = (
            1 if single else (4 if four_gpu else (2 if args.layout == "two_gpu" else 3))
        )
        args.gpus = ",".join(str(index) for index in range(count))
    if args.max_steps is None:
        args.max_steps = 1000 if libero_long else 500
    if args.micro_batch_size is not None:
        if not libero_long:
            parser.error("--micro-batch-size is supported only by LIBERO-Long profiles")
        if args.micro_batch_size <= 0:
            parser.error("--micro-batch-size must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.action_horizon is not None and args.action_horizon <= 0:
        parser.error("--action-horizon must be positive")
    if args.denoise_steps is not None and args.denoise_steps < 2:
        parser.error("--denoise-steps must be at least 2 for SDE sampling")
    if args.noise_level is not None and (
        not math.isfinite(args.noise_level) or args.noise_level <= 0
    ):
        parser.error("--noise-level must be finite and positive for PPO sampling")
    if args.clip_ratio_c is not None and (
        not math.isfinite(args.clip_ratio_c) or args.clip_ratio_c <= 1
    ):
        parser.error("--clip-ratio-c must be finite and greater than one")
    if args.value_head_hidden_sizes is not None and any(
        width <= 0 for width in args.value_head_hidden_sizes
    ):
        parser.error("--value-head-hidden-sizes must contain positive integers")
    if args.lr_schedule_steps is not None and args.lr_schedule_steps <= 0:
        parser.error("--lr-schedule-steps must be positive")
    if (probe or single) and args.smoke_test_steps is None:
        args.smoke_test_steps = 1
    if (
        args.smoke_test_steps is not None
        and not 0 < args.smoke_test_steps <= args.max_steps
    ):
        parser.error(
            "--smoke-test-steps must be positive and no greater than --max-steps"
        )
    return args


def resume_checkpoint_contract(path: Path, task: str, layout: str) -> dict:
    """Require a completed local PPO checkpoint and its original launch manifest."""
    path = path.expanduser().resolve()
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if match is None:
        raise ValueError("--resume-dir must point to a global_step_<N> directory")
    step = int(match.group(1))
    marker = path / ".checkpoint_complete"
    if not marker.is_file() or marker.read_text().strip() != str(step):
        raise ValueError(f"PPO checkpoint has no matching completion marker: {path}")
    for relative in (
        "actor/dcp_checkpoint/.metadata",
        "actor/model_state_dict/full_weights.pt",
    ):
        item = path / relative
        if not item.is_file() or item.stat().st_size == 0:
            raise ValueError(f"PPO checkpoint is missing {relative}: {path}")
    shards = list((path / "actor/dcp_checkpoint").glob("*.distcp"))
    if not shards or any(item.stat().st_size == 0 for item in shards):
        raise ValueError(
            f"PPO checkpoint has missing or empty distributed shards: {path}"
        )
    manifest = path.parents[2] / "launch_plan.json"
    if not manifest.is_file():
        raise ValueError(
            f"Resume requires the checkpoint's launch_plan.json: {manifest}"
        )
    source = json.loads(manifest.read_text())
    if source.get("task") != task or source.get("layout") != layout:
        raise ValueError("Resume task and GPU layout must match the source run")
    return {"checkpoint": str(path), "step": step, "source_plan": source}


def validate_resume_config(
    source: dict, current: dict, *, ignore_last_override: bool | None = None
) -> None:
    """Keep model, optimizer, and rollout semantics stable during recovery."""

    # This sampling-only experiment changes neither parameter shapes nor the
    # optimizer. The runner collects fresh trajectories after restoring. Permit
    # only the explicitly requested flag; retain every other resume check.
    if ignore_last_override is not None:
        if (
            current["actor"]["model"]["openpi"].get("ignore_last")
            is not ignore_last_override
        ):
            raise ValueError(
                "Resume ignore_last override must match the requested flag"
            )
        source = copy.deepcopy(source)
        source["actor"]["model"]["openpi"]["ignore_last"] = ignore_last_override

    def get(config: dict, dotted: str):
        for key in dotted.split("."):
            config = config[key]
        return config

    for field in (
        "actor.model",
        "actor.optim",
        "actor.global_batch_size",
        "actor.fsdp_config.sharding_strategy",
        "algorithm.update_epoch",
        "algorithm.clip_ratio_c",
        "algorithm.gamma",
        "algorithm.gae_lambda",
        "algorithm.logprob_type",
        "algorithm.ppo_update_guard.loss_weighting",
        "env.train.env_type",
        "env.train.task_indices",
        "env.train.total_num_envs",
        "env.train.rollout_epoch",
        "env.train.max_episode_steps",
        "env.train.max_steps_per_rollout_epoch",
        "rollout.model",
        "rollout.pipeline_stage_num",
    ):
        if get(source, field) != get(current, field):
            raise ValueError(f"Resume must preserve the source configuration: {field}")


def hydra_overrides(
    gpus: list[str], task_index: int, output: Path, max_steps: int
) -> list[str]:
    """Pin models to the first GPU and eight training envs to the second."""
    return [
        f"cluster.component_placement={{actor\\,env\\,rollout:{gpus[0]}}}",
        f"env.train.isaac_cuda_visible_devices=[{gpus[min(1, len(gpus) - 1)]}]",
        f"env.eval.isaac_cuda_visible_devices=[{gpus[0]}]",
        f"env.train.task_indices=[{task_index}]",
        f"env.eval.task_indices=[{task_index}]",
        f"runner.logger.log_path={json.dumps(str(output))}",
        f"runner.max_steps={max_steps}",
        f"runner.max_epochs={max_steps}",
        "runner.resume_dir=null",
    ]


def compose_config(overrides: list[str], config_name: str = CONFIG_NAME) -> dict:
    """Compose the exact launch configuration without starting Ray or Isaac."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(
        version_base="1.1", config_dir=str(ROOT / "examples/embodiment/config")
    ):
        cfg = compose(config_name=config_name, overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def validate_training_schedule(
    config: dict, actor_world_size: int = 1, starting_step: int = 0
) -> dict:
    """Validate real minibatch arithmetic and report upper bounds before launch."""

    def positive_int(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        return value

    train, actor = config["env"]["train"], config["actor"]
    envs = positive_int(train["total_num_envs"], "env.train.total_num_envs")
    rounds = positive_int(train["rollout_epoch"], "env.train.rollout_epoch")
    horizon = positive_int(
        train["max_steps_per_rollout_epoch"], "env.train.max_steps_per_rollout_epoch"
    )
    chunk = positive_int(actor["model"]["num_action_chunks"], "num_action_chunks")
    batch = positive_int(actor["global_batch_size"], "actor.global_batch_size")
    micro = positive_int(actor["micro_batch_size"], "actor.micro_batch_size")
    epochs = positive_int(config["algorithm"]["update_epoch"], "algorithm.update_epoch")
    limit = positive_int(config["runner"]["max_steps"], "runner.max_steps")
    if type(starting_step) is not int or not 0 <= starting_step < limit:
        raise ValueError("The training limit must exceed the restored PPO step")
    remaining = limit - starting_step
    world = positive_int(actor_world_size, "actor_world_size")
    chunks_per_episode, remainder = divmod(horizon, chunk)
    episodes = envs * rounds
    samples = episodes * chunks_per_episode
    if remainder or samples % batch or batch % (micro * world) or envs % world:
        raise ValueError(
            "The rollout horizon, global batch and micro batch must divide the "
            "rollout exactly across all actor ranks"
        )
    warmup = actor["optim"].get("critic_warmup_steps", 0)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(
            "actor.optim.critic_warmup_steps must be a non-negative integer"
        )
    zero_limit = positive_int(
        config["runner"]["max_consecutive_zero_success"],
        "runner.max_consecutive_zero_success",
    )
    updates = samples // batch * epochs
    episode_cap = positive_int(
        train["max_episode_steps"], "env.train.max_episode_steps"
    )
    real_chunks = (episode_cap + chunk - 1) // chunk
    return {
        "train_episodes_per_iteration": episodes,
        "starting_step": starting_step,
        "remaining_iterations": remaining,
        "planned_train_episodes": episodes * remaining,
        "planned_chunk_samples_per_iteration": samples,
        "global_batch_size": batch,
        "micro_batch_size": micro,
        "chunk_steps_per_rollout_epoch": chunks_per_episode,
        "max_episode_steps": episode_cap,
        "terminal_padding_chunk_steps": max(chunks_per_episode - real_chunks, 0),
        "gradient_accumulation": batch // micro // world,
        "optimizer_updates_per_iteration": updates,
        "planned_optimizer_updates": updates * remaining,
        "optimizer_updates_are_upper_bounds": True,
        "critic_warmup_successful_optimizer_steps": warmup,
        "minimum_iterations_for_critic_warmup": (warmup + updates - 1) // updates,
        "planned_actor_updates_after_warmup_max": max(
            updates * remaining - (warmup if starting_step == 0 else 0), 0
        ),
        "max_consecutive_zero_success": zero_limit,
        "training_may_stop_early": True,
    }


def build_plan(args: argparse.Namespace) -> dict:
    """Check checkpoint assets, task identity and independent training GPU."""
    gpus = _csv(args.gpus)
    four_gpu = args.layout == "collocated_4gpu_libero_long"
    probe = args.layout == "libero_long_3gpu_probe"
    single = args.layout == "collocated_1gpu"
    libero_long = four_gpu or probe or single
    collocated = args.layout == "collocated_3gpu" or libero_long
    separated = args.layout == "separated_3gpu"
    multi_gpu = collocated or separated
    count = 1 if single else (4 if four_gpu else (3 if multi_gpu else 2))
    config_name = {
        "collocated_1gpu": FLOWER_SINGLE_CONFIG_NAME,
        "two_gpu": CONFIG_NAME,
        "collocated_3gpu": COLLOCATED_CONFIG_NAME,
        "separated_3gpu": SEPARATED_CONFIG_NAME,
        "collocated_4gpu_libero_long": LIBERO_LONG_CONFIG_NAME,
        "libero_long_3gpu_probe": LIBERO_LONG_PROBE_CONFIG_NAME,
    }[args.layout]
    if single and args.task != FLOWER_TASK:
        raise ValueError("The single-GPU validation profile requires the flower task")
    if four_gpu and args.task == CLEAN_TASK:
        config_name = CLEAN_LIBERO_LONG_CONFIG_NAME
    if four_gpu and args.task == FLOWER_TASK:
        config_name = FLOWER_FOUR_CONFIG_NAME
    if four_gpu and args.task == PACK_TASK:
        config_name = PACK_FOUR_CONFIG_NAME
    if len(gpus) != count or any(not gpu.isdecimal() for gpu in gpus):
        raise ValueError(
            f"--gpus requires {count} distinct non-negative physical GPU IDs"
        )
    gpus = [str(int(gpu)) for gpu in gpus]
    if len(set(gpus)) != count:
        raise ValueError(f"--gpus must select {count} different physical GPUs")
    if collocated and gpus != sorted(gpus, key=int):
        raise ValueError("Collocated physical GPU IDs must be in ascending rank order")
    if separated and gpus[1:] != sorted(gpus[1:], key=int):
        raise ValueError("Sampling GPU IDs must be in ascending rank order")
    actor_gpus = gpus[:2] if probe else (gpus if collocated else [gpus[0]])
    rollout_gpus = gpus if probe else (gpus[1:] if separated else actor_gpus)
    model_gpus = sorted(gpus, key=int) if multi_gpu else actor_gpus
    isaac_gpus = gpus if collocated else gpus[1:]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = (
        (
            args.output
            or Path(
                f"/root/autodl-tmp/outputs/lbench_ppo_single_task_{args.layout}_{timestamp}"
            )
        )
        .expanduser()
        .resolve()
    )
    if output.exists():
        raise FileExistsError(f"Choose a fresh --output directory: {output}")
    resume = (
        resume_checkpoint_contract(args.resume_dir, args.task, args.layout)
        if args.resume_dir is not None
        else None
    )
    starting_step = resume["step"] if resume else 0
    stop_step = (
        starting_step + args.smoke_test_steps
        if args.smoke_test_steps is not None
        else args.max_steps
    )
    if not starting_step < stop_step <= args.max_steps:
        raise ValueError("The requested additional rounds must fit below --max-steps")
    lr_steps = args.lr_schedule_steps
    if lr_steps is None:
        lr_steps = (
            resume["source_plan"]["config"]["actor"]["optim"]["total_training_steps"]
            if resume
            else args.max_steps
        )
    contract = checkpoint_contract(
        args.checkpoint.expanduser().resolve(), allowed_action_horizons=(20, 50)
    )
    os.environ.update(
        LBENCH_SFT_CHECKPOINT=contract["checkpoint"],
        LBENCH_NORM_STATS=contract["norm_stats"],
        CUDA_VISIBLE_DEVICES=",".join(model_gpus),
        RLINF_GPUS=",".join(model_gpus),
        LBENCH_ISAAC_GPUS=",".join(isaac_gpus),
        EMBODIED_PATH=str(ROOT / "examples/embodiment"),
    )
    sharing = {"shared_gpu_ids": tuple(isaac_gpus)} if multi_gpu else {}
    resources = _common_checks(
        "single-eval", model_gpus, isaac_gpus, allow_no_gpu=args.dry_run, **sharing
    )
    names = resources["task_names"]
    if args.task not in names:
        raise ValueError(f"Unknown LBench task {args.task!r}; available={names}")
    task_index = names.index(args.task)
    runtime_env = _isaac_runtime_env(Path(resources["isaac_python"]))
    os.environ.update(runtime_env)
    overrides = hydra_overrides(gpus, task_index, output, stop_step)
    if resume:
        overrides[-1] = f"runner.resume_dir={json.dumps(resume['checkpoint'])}"
    if (
        args.smoke_test_steps is not None
        or libero_long
        or resume
        or args.lr_schedule_steps
    ):
        overrides.append(f"actor.optim.total_training_steps={lr_steps}")
    if args.micro_batch_size is not None:
        overrides.append(f"actor.micro_batch_size={args.micro_batch_size}")
    if args.action_horizon is not None:
        overrides.append(f"++actor.model.openpi.action_horizon={args.action_horizon}")
    if args.denoise_steps is not None:
        overrides.extend(
            [
                f"actor.model.num_steps={args.denoise_steps}",
                f"actor.model.openpi.num_steps={args.denoise_steps}",
            ]
        )
    if args.noise_level is not None:
        overrides.append(f"actor.model.openpi.noise_level={args.noise_level}")
    if args.ignore_last is not None:
        overrides.append(
            f"++actor.model.openpi.ignore_last={str(args.ignore_last).lower()}"
        )
    if args.clip_ratio_c is not None:
        overrides.append(f"algorithm.clip_ratio_c={args.clip_ratio_c}")
    if args.value_head_hidden_sizes is not None:
        widths = ",".join(str(width) for width in args.value_head_hidden_sizes)
        overrides.append(f"++actor.model.openpi.value_head_hidden_sizes=[{widths}]")
    if args.record_diagnostics:
        for mode in ("train", "eval"):
            prefix = f"++env.{mode}.data_collection"
            overrides.extend(
                [
                    f"{prefix}.enabled=true",
                    f"{prefix}.export_format=diagnostic",
                    f"{prefix}.save_dir={json.dumps(str(output / 'recordings' / mode))}",
                    f"{prefix}.only_success=false",
                    f"{prefix}.fps=30",
                    f"{prefix}.max_pending_exports=1",
                    f"{prefix}.diagnostic_image_interval=20",
                ]
            )
        overrides.append(
            f"++actor.rollout_diagnostics_dir={json.dumps(str(output / 'recordings' / 'ppo'))}"
        )
    if multi_gpu:
        # Explicit hardware-to-process maps work with noncontiguous GPU IDs;
        # ordering matches the Isaac env_rank device list below.
        placement = ",".join(f"{gpu}:{rank}" for rank, gpu in enumerate(rollout_gpus))
        component_map = (
            f'actor\\,env\\,rollout:"{placement}"'
            if collocated
            else f'actor:{gpus[0]},env\\,rollout:"{placement}"'
        )
        if probe:
            actor_placement = ",".join(
                f"{gpu}:{rank}" for rank, gpu in enumerate(actor_gpus)
            )
            component_map = f'actor:"{actor_placement}",env\\,rollout:"{placement}"'
        overrides[:3] = [
            f"cluster.component_placement={{{component_map}}}",
            f"env.train.isaac_cuda_visible_devices=[{','.join(isaac_gpus)}]",
            f"env.eval.isaac_cuda_visible_devices=[{','.join(isaac_gpus)}]",
        ]
        config = compose_config(overrides, config_name)
    else:
        config = compose_config(overrides)
    predicted = config["actor"]["model"]["openpi"].get("action_horizon")
    executed = config["actor"]["model"]["num_action_chunks"]
    if contract["manifest"]["action_horizon"] == 20 and predicted != 20:
        raise ValueError("An adapted 20-step SFT checkpoint requires action_horizon=20")
    if predicted is not None and predicted < executed:
        raise ValueError(
            "Predicted action horizon must cover the executed action chunk"
        )
    fsdp_precision = validate_fsdp_precision(config)
    from omegaconf import OmegaConf

    from rlinf.algorithms.ppo_update_guard import validate_ppo_update_guard_config
    from rlinf.models.embodiment.openpi.precision import (
        validate_strict_ppo_precision_config,
    )

    strict_precision = validate_strict_ppo_precision_config(
        config["actor"]["model"], fsdp_config=config["actor"]["fsdp_config"]
    )
    if not strict_precision:
        raise ValueError(
            "The single-task PPO profile requires strict_ppo_precision=true"
        )
    update_guard = validate_ppo_update_guard_config(OmegaConf.create(config))
    if not update_guard["enabled"]:
        raise ValueError(
            "The single-task PPO profile requires ppo_update_guard.enabled=true"
        )
    if resume:
        validate_resume_config(
            resume["source_plan"]["config"],
            config,
            ignore_last_override=args.ignore_last,
        )
    schedule = validate_training_schedule(config, len(actor_gpus), starting_step)
    train, evaluation = config["env"]["train"], config["env"]["eval"]
    evaluation_enabled = config["runner"]["val_check_interval"] > 0
    if multi_gpu and (
        config["rollout"]["pipeline_stage_num"] != 1
        or train["isaac_gpu_assignment"] != "env_rank"
        or train["total_num_envs"] % len(isaac_gpus)
    ):
        raise ValueError(
            "Multi-GPU profiles require pipeline_stage_num=1, env_rank GPU "
            "assignment and evenly divided environments"
        )
    evaluation_envs = evaluation["total_num_envs"] if evaluation_enabled else 0
    serial_eval_pools = multi_gpu and evaluation_enabled
    if serial_eval_pools:
        if (
            isinstance(evaluation_envs, bool)
            or not isinstance(evaluation_envs, int)
            or evaluation_envs < 1
            or evaluation_envs % len(isaac_gpus)
            or evaluation["isaac_gpu_assignment"] != "env_rank"
            or train.get("offload_before_eval") is not True
            or train.get("lazy_start") is not True
            or evaluation.get("lazy_start") is not True
            or evaluation.get("enable_offload") is not True
            or train["auto_reset"]
            or evaluation["auto_reset"]
            or config["runner"].get("overlap_env_bootstrap", False)
        ):
            raise ValueError(
                "Multi-GPU automatic evaluation requires evenly divided eval envs, "
                "env_rank assignment, lazy_start for both pools, "
                "train.offload_before_eval=true, eval.enable_offload=true, "
                "auto_reset=false and no overlapping bootstrap"
            )
    from rlinf.utils.eval_initial_noise import build_eval_noise_plan

    build_eval_noise_plan(
        OmegaConf.create(config),
        env_world_size=len(isaac_gpus) if multi_gpu else 1,
        rollout_world_size=len(rollout_gpus),
    )
    visible = _visible_gpu_ids()
    command = [
        sys.executable,
        str(ROOT / "examples/embodiment/train_embodied_agent.py"),
        "--config-name",
        config_name,
        *overrides,
    ]
    return {
        "dry_run": args.dry_run,
        "output": str(output),
        "task": args.task,
        "task_index": task_index,
        "layout": args.layout,
        "smoke_test_steps": args.smoke_test_steps,
        "resume_checkpoint": resume["checkpoint"] if resume else None,
        "ignore_last_override": args.ignore_last,
        "learning_rate_schedule_steps": config["actor"]["optim"][
            "total_training_steps"
        ],
        "actor_gpus": actor_gpus,
        "rollout_gpus": rollout_gpus,
        "training_isaac_gpus": isaac_gpus,
        "actor_world_size": len(actor_gpus),
        "rollout_world_size": len(rollout_gpus),
        "env_world_size": len(isaac_gpus) if multi_gpu else 1,
        "actor_gpu": None if collocated else gpus[0],
        "rollout_gpu": rollout_gpus[0] if len(rollout_gpus) == 1 else None,
        "training_isaac_gpu": None if multi_gpu else gpus[1],
        "evaluation_enabled": evaluation_enabled,
        "evaluation_interval_rounds": config["runner"]["val_check_interval"],
        "evaluate_before_training": config["runner"].get("eval_before_train", False),
        "evaluation_isaac_gpu": gpus[0]
        if evaluation_enabled and not multi_gpu
        else None,
        "evaluation_isaac_gpus": isaac_gpus
        if serial_eval_pools
        else ([gpus[0]] if evaluation_enabled else []),
        "evaluation_uses_serial_env_pools": serial_eval_pools,
        "evaluation_shares_model_gpu": evaluation_enabled,
        "resident_isaac_processes_by_gpu": {
            gpu: max(train["total_num_envs"], evaluation_envs) // len(isaac_gpus)
            if gpu in isaac_gpus
            else 0
            for gpu in gpus
        }
        if multi_gpu
        else {
            gpus[0]: evaluation_envs,
            gpus[1]: train["total_num_envs"],
        },
        "resident_isaac_processes": max(train["total_num_envs"], evaluation_envs)
        if serial_eval_pools
        else train["total_num_envs"] + evaluation_envs,
        "training_isaac_processes": train["total_num_envs"],
        "evaluation_isaac_processes": evaluation_envs,
        "visible_gpu_ids": sorted(visible),
        "requested_gpus_available": set(gpus) <= visible,
        "hardware_training_validated": False,
        **schedule,
        "eval_episodes_per_task": evaluation_envs * evaluation["rollout_epoch"],
        "checkpoint_contract": contract,
        "fsdp_precision": fsdp_precision,
        "strict_ppo_precision_validated": strict_precision,
        "ppo_update_guard": update_guard,
        "resources": resources,
        "isaac_runtime_env": runtime_env,
        "config": config,
        "command": command,
    }


def main(argv: list[str] | None = None) -> None:
    """Validate then start foreground PPO only when --dry-run is absent."""
    args = parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if _existing_ray_cluster():
        raise RuntimeError(
            "An existing Ray cluster is reachable; launch this isolated recipe after that job has ended and its cluster is stopped"
        )
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "launch_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    from omegaconf import OmegaConf

    OmegaConf.save(OmegaConf.create(plan["config"]), output / "resolved_config.yaml")
    print(f"PPO output: {output}", flush=True)
    print(
        f"Task: {args.task}; completed rounds {plan['starting_step']} -> "
        f"{plan['config']['runner']['max_steps']}, "
        f"{plan['train_episodes_per_iteration']} episodes/iteration, "
        f"{plan['planned_train_episodes']} planned episodes maximum",
        flush=True,
    )
    print(
        f"GPUs: actor={plan['actor_gpus']}, rollout={plan['rollout_gpus']}, "
        f"training Isaac={plan['training_isaac_gpus']}",
        flush=True,
    )
    print(
        f"Evaluation during training: {plan['evaluation_enabled']}; "
        f"save every {plan['config']['runner']['save_interval']} iterations; "
        f"keep latest {plan['config']['runner']['max_saved_checkpoints']} checkpoints",
        flush=True,
    )
    print(
        f"Update limit: {plan['optimizer_updates_per_iteration']} optimizer calls/iteration; "
        f"critic warmup: {plan['critic_warmup_successful_optimizer_steps']} successful steps; "
        f"save and stop after {plan['max_consecutive_zero_success']} zero-success iterations",
        flush=True,
    )
    run_foreground(plan["command"], output)


if __name__ == "__main__":
    main()
