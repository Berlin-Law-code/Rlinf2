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

"""Validate and launch a foreground, SFT-aligned two-task LBench PPO trial."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

CONFIG_NAME = "lbench_ppo_openpi_pi05_two_tasks_3gpu"
DEFAULT_TASKS = ("stack_bowls", "store_objects_in_the_drawer_0")
DEFAULT_CHECKPOINT = Path(
    "/root/autodl-tmp/outputs/lbench_sft_lerobot_aligned_fp32_4gpu_memfix_004/"
    "lbench_sft_pi05_lerobot_aligned_4gpu/checkpoints/global_step_30000"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse deliberate trial controls; inherited generic exports are ignored."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus", default="0,1,2", help="Physical actor,rollout,Isaac IDs"
    )
    parser.add_argument("--tasks", nargs=2, default=list(DEFAULT_TASKS))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--max-steps", type=int, default=20, help="PPO rollout/update iterations"
    )
    parser.add_argument(
        "--output", type=Path, help="New output directory (never resumed)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check assets and config only; report unavailable GPUs without launching",
    )
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    return args


def checkpoint_contract(
    checkpoint: Path, *, allowed_action_horizons: tuple[int, ...] = (50,)
) -> dict:
    """Check the real checkpoint metadata and FP32 tensor headers, without copies."""
    import torch

    from rlinf.models.embodiment.openpi.checkpoint_assets import (
        read_checkpoint_manifest,
        resolve_checkpoint_norm_stats,
    )

    manifest = read_checkpoint_manifest(checkpoint)
    compatibility = manifest.get("compatibility", {})
    if (
        manifest.get("config_name") != "pi05_lbench_joint"
        or manifest.get("action_horizon") not in allowed_action_horizons
        or manifest.get("action_dim") != 8
        or compatibility.get("lerobot_compatible") is not True
        or compatibility.get("parameter_precision") != "float32"
    ):
        raise ValueError(
            "This trial requires an aligned FP32 LBench SFT checkpoint manifest"
        )
    stats = resolve_checkpoint_norm_stats(checkpoint, None)
    if stats is None:
        raise ValueError(
            "The checkpoint must bundle its training normalization statistics"
        )
    weights = next(
        (
            path
            for path in (
                checkpoint / "actor/model_state_dict/full_weights.pt",
                checkpoint / "model_state_dict/full_weights.pt",
            )
            if path.is_file()
        ),
        None,
    )
    if weights is None:
        raise FileNotFoundError(f"No full_weights.pt in {checkpoint}")
    # mmap reads tensor metadata without materializing a second 13 GB model.
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint must contain a non-empty model state dictionary")
    tensors = {
        key: value for key, value in state.items() if isinstance(value, torch.Tensor)
    }
    invalid = [
        key
        for key, value in tensors.items()
        if value.is_floating_point() and value.dtype != torch.float32
    ]
    if not tensors or invalid:
        raise ValueError(f"Checkpoint parameters must already be FP32: {invalid[:5]}")
    return {
        "checkpoint": str(checkpoint),
        "weights": str(weights),
        "norm_stats": str(stats),
        "tensor_count": len(tensors),
        "manifest": manifest,
    }


def hydra_overrides(
    gpus: list[str], task_indices: list[int], output: Path, max_steps: int
) -> list[str]:
    """Use the same explicit physical GPU mapping for preflight and execution."""
    tasks = json.dumps(task_indices, separators=(",", ":"))
    return [
        f"cluster.component_placement={{actor:{gpus[0]},env\\,rollout:{gpus[1]}}}",
        f"env.train.isaac_cuda_visible_devices=[{gpus[2]}]",
        f"env.eval.isaac_cuda_visible_devices=[{gpus[2]}]",
        f"env.train.task_indices={tasks}",
        f"env.eval.task_indices={tasks}",
        f"runner.logger.log_path={json.dumps(str(output))}",
        f"runner.max_steps={max_steps}",
        f"runner.max_epochs={max_steps}",
        "runner.resume_dir=null",
    ]


def compose_config(overrides: list[str]) -> dict:
    """Compose Hydra without validate_cfg, which would start a Ray cluster."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(
        version_base="1.1", config_dir=str(ROOT / "examples/embodiment/config")
    ):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def validate_fsdp_precision(config: dict) -> dict[str, str]:
    """Resolve the actor's FP32 settings with the actual FSDP dtype parser."""
    import torch

    from rlinf.config import torch_dtype_from_precision

    mixed_precision = config["actor"]["fsdp_config"]["mixed_precision"]
    resolved = {}
    for field in ("param_dtype", "reduce_dtype", "buffer_dtype"):
        value = mixed_precision[field]
        path = f"actor.fsdp_config.mixed_precision.{field}"
        try:
            dtype = torch_dtype_from_precision(value)
        except ValueError as exc:
            raise ValueError(f"Invalid FSDP precision {path}={value!r}: {exc}") from exc
        if dtype != torch.float32:
            raise ValueError(f"The aligned PPO profile requires FP32: {path}={value!r}")
        resolved[field] = str(dtype)
    return resolved


def build_plan(args: argparse.Namespace) -> dict:
    """Validate all local assets; dry runs have no training or output side effects."""
    gpus = _csv(args.gpus)
    if len(gpus) != 3 or any(not gpu.isdecimal() for gpu in gpus):
        raise ValueError("--gpus requires three distinct non-negative physical GPU IDs")
    gpus = [str(int(gpu)) for gpu in gpus]
    if len(set(gpus)) != 3:
        raise ValueError("--gpus must select three different physical GPUs")
    if len(args.tasks) != 2 or len(set(args.tasks)) != 2:
        raise ValueError("Select exactly two different task names")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = (
        (
            args.output
            or Path(f"/root/autodl-tmp/outputs/lbench_ppo_two_tasks_3gpu_{timestamp}")
        )
        .expanduser()
        .resolve()
    )
    if output.exists():
        raise FileExistsError(f"Choose a fresh --output directory: {output}")
    contract = checkpoint_contract(args.checkpoint.expanduser().resolve())
    os.environ.update(
        LBENCH_SFT_CHECKPOINT=contract["checkpoint"],
        LBENCH_NORM_STATS=contract["norm_stats"],
        CUDA_VISIBLE_DEVICES=",".join(gpus[:2]),
        LBENCH_ISAAC_GPUS=gpus[2],
        RLINF_GPUS=",".join(gpus[:2]),
        EMBODIED_PATH=str(ROOT / "examples/embodiment"),
    )
    resources = _common_checks(
        "dual-ppo", gpus[:2], gpus[2:], allow_no_gpu=args.dry_run
    )
    names = resources["task_names"]
    unknown = [name for name in args.tasks if name not in names]
    if unknown:
        raise ValueError(f"Unknown LBench tasks {unknown}; available={names}")
    indices = [names.index(name) for name in args.tasks]
    runtime_env = _isaac_runtime_env(Path(resources["isaac_python"]))
    os.environ.update(runtime_env)
    overrides = hydra_overrides(gpus, indices, output, args.max_steps)
    config = compose_config(overrides)
    fsdp_precision = validate_fsdp_precision(config)
    visible = _visible_gpu_ids()
    command = [
        sys.executable,
        str(ROOT / "examples/embodiment/train_embodied_agent.py"),
        "--config-name",
        CONFIG_NAME,
        *overrides,
    ]
    return {
        "dry_run": args.dry_run,
        "output": str(output),
        "tasks": args.tasks,
        "task_indices": indices,
        "actor_gpu": gpus[0],
        "rollout_gpu": gpus[1],
        "isaac_gpu": gpus[2],
        "visible_gpu_ids": sorted(visible),
        "requested_gpus_available": set(gpus) <= visible,
        "hardware_training_validated": False,
        "resident_isaac_processes": 4,
        "train_episodes_per_iteration": 8,
        "optimizer_updates_per_iteration": 30,
        "eval_episodes_per_task": 10,
        "checkpoint_contract": contract,
        "fsdp_precision": fsdp_precision,
        "resources": resources,
        "isaac_runtime_env": runtime_env,
        "config": config,
        "command": command,
    }


def run_foreground(command: list[str], output: Path) -> None:
    """Keep the child in the terminal's process group and tee its training log."""
    with (output / "train.log").open("x", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                    log.flush()
                return_code = process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main(argv: list[str] | None = None) -> None:
    """Only a non-dry invocation starts PPO, after all checks complete."""
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
        f"Tasks: {', '.join(args.tasks)}; {args.max_steps} iterations, 8 training episodes/iteration",
        flush=True,
    )
    run_foreground(plan["command"], output)


if __name__ == "__main__":
    main()
