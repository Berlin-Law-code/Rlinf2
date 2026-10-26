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

"""Plan or run the five resource-dependent LBench M8 acceptance stages."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCENARIOS = ("sft-smoke", "single-eval", "dual-ppo", "eval-8", "ppo-8")


def _csv(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("GPU list must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"GPU list contains duplicates: {result}")
    return result


def _env_path(name: str, kind: str = "path") -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required environment variable {name} is not set")
    expanded = Path(value).expanduser()
    # Do not resolve Python launcher symlinks: venv/bin/python intentionally
    # points at a base executable while retaining a distinct sys.prefix.
    path = (
        Path(os.path.abspath(str(expanded)))
        if kind == "executable"
        else expanded.resolve()
    )
    valid = path.is_dir() if kind == "dir" else path.is_file()
    if kind == "executable":
        valid = path.is_file() and os.access(path, os.X_OK)
    if not valid:
        raise FileNotFoundError(f"{name} is not a valid {kind}: {path}")
    return path


def _dataset_root(repo: str) -> Path:
    direct = Path(repo).expanduser()
    if (direct / "meta" / "info.json").is_file():
        return direct.resolve()
    home = Path(
        os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")
    ).expanduser()
    cached = home / repo
    if (cached / "meta" / "info.json").is_file():
        return cached.resolve()
    raise FileNotFoundError(
        f"LBENCH_LEROBOT_REPO is not materialized as a LeRobot dataset: {repo}"
    )


def _visible_gpu_ids() -> set[str]:
    command = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()
    return {line.strip() for line in output.splitlines() if line.strip()}


def _existing_ray_cluster() -> bool:
    """Return whether the local Ray CLI can reach an already-running cluster."""

    try:
        result = subprocess.run(
            ["ray", "status"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _scenario_artifact_paths(scenario: str, output: Path) -> list[Path]:
    """Return paths that a fresh launcher run may create or overwrite."""

    if scenario == "sft-smoke":
        return [output / "m8_sft_smoke"]
    if scenario in {"single-eval", "eval-8"}:
        return [output / scenario]
    if scenario == "dual-ppo":
        return [output / "m8_dual_ppo_smoke", output / "episodes/train"]
    return [output / "lbench_ppo_openpi_pi05"]


def _check_fresh_output(scenario: str, output: Path) -> None:
    conflicts = [
        path
        for path in _scenario_artifact_paths(scenario, output)
        if path.exists() and (not path.is_dir() or any(path.iterdir()))
    ]
    if conflicts:
        raise FileExistsError(
            "refusing to mix a new M8 run with existing artifacts: "
            f"{[str(path) for path in conflicts]}. Choose a fresh --output "
            "directory; use direct Hydra resume commands for an interrupted run."
        )


def _validate_task_config(path: Path, *, require_multitask: bool) -> dict:
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    tasks = config.get("tasks") if isinstance(config, dict) else None
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"LBENCH_TASK_CONFIG has no non-empty tasks list: {path}")
    if require_multitask and len(tasks) < 2:
        raise ValueError(
            "this M8 stage requires a joint multi-task YAML (at least two task "
            "entries); build one with LBench_multi_env/scripts/eval/"
            "build_multitask_task_config.py"
        )
    names = []
    prompts = []
    for task in tasks:
        data = task.get("data", {}) if isinstance(task, dict) else {}
        names.append(
            str(
                (data.get("task_dir") if isinstance(data, dict) else None)
                or (task.get("name") if isinstance(task, dict) else None)
                or ""
            )
        )
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(
            "LBENCH_TASK_CONFIG effective task names (data.task_dir/name) must "
            "be non-empty and unique"
        )
    required_cameras = {"obs_camera", "obs_camera_2", "realsense"}
    for index, task in enumerate(tasks):
        cameras = task.get("cameras", []) if isinstance(task, dict) else []
        camera_names = {
            str(camera.get("name", ""))
            for camera in cameras
            if isinstance(camera, dict)
        }
        missing = sorted(required_cameras - camera_names)
        if missing:
            raise ValueError(
                f"LBENCH_TASK_CONFIG task {index} ({names[index]!r}) is missing "
                f"required cameras: {missing}"
            )
        data = task.get("data", {}) if isinstance(task, dict) else {}
        instruction = (
            data.get("language_instruction") if isinstance(data, dict) else None
        )
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                f"LBENCH_TASK_CONFIG task {index} ({names[index]!r}) has no "
                "string language_instruction"
            )
        prompts.append(instruction.strip())
    if len(set(prompts)) != len(prompts):
        raise ValueError(
            "LBENCH_TASK_CONFIG language instructions must be unique across tasks"
        )
    return {
        "task_count": len(tasks),
        "task_names": names,
        "task_prompts": prompts,
    }


def _validate_dataset_prompt_alignment(task_contract: dict) -> dict:
    """Ensure SFT prompts and live simulator prompts describe the same tasks."""

    from toolkits.lbench_data_dry_run import (
        _dataset_task_prompts,
        _storage_format,
        _validate_metadata,
    )

    repo = os.environ.get("LBENCH_LEROBOT_REPO")
    if not repo:
        raise ValueError(
            "M8 eval/PPO acceptance requires LBENCH_LEROBOT_REPO so the "
            "runtime task prompts can be checked against SFT data"
        )
    dataset = _dataset_root(repo)
    metadata = _validate_metadata(dataset)
    storage_format = _storage_format(metadata.get("codebase_version"))
    dataset_prompts = _dataset_task_prompts(dataset, storage_format)
    runtime_prompts = list(task_contract["task_prompts"])
    missing_at_runtime = sorted(set(dataset_prompts) - set(runtime_prompts))
    unseen_by_sft = sorted(set(runtime_prompts) - set(dataset_prompts))
    if missing_at_runtime or unseen_by_sft:
        raise ValueError(
            "LBench SFT dataset prompts and simulator language instructions "
            "do not match: "
            f"dataset_only={missing_at_runtime}, runtime_only={unseen_by_sft}"
        )
    return {
        "prompt_dataset": str(dataset),
        "prompt_storage_format": storage_format,
        "dataset_task_prompts": dataset_prompts,
        "runtime_task_prompts": runtime_prompts,
        "task_prompts_aligned": True,
    }


def _common_checks(
    scenario: str,
    rlinf_gpus: list[str],
    isaac_gpus: list[str],
    allow_no_gpu: bool,
    allow_shared_python: bool = False,
    shared_gpu_ids: tuple[str, ...] = (),
):
    from toolkits.lbench_data_dry_run import (
        _storage_format,
        _v21_task_episode_counts,
        _validate_metadata,
        validate_lerobot_runtime,
        validate_norm_stats,
    )

    if not (ROOT / "rlinf").is_dir():
        raise FileNotFoundError(f"RLinf root is invalid: {ROOT}")
    visible = _visible_gpu_ids()
    requested = set(rlinf_gpus)
    if scenario != "sft-smoke":
        requested |= set(isaac_gpus)
        overlap = set(rlinf_gpus) & set(isaac_gpus)
        if overlap - set(shared_gpu_ids):
            raise ValueError(f"RLinf and Isaac GPU sets overlap: {sorted(overlap)}")
        if set(shared_gpu_ids) - overlap:
            raise ValueError("Explicit shared_gpu_ids must belong to both GPU sets")
    if not allow_no_gpu and not requested <= visible:
        raise RuntimeError(
            f"requested GPUs {sorted(requested)} are not all visible; "
            f"visible={sorted(visible)}"
        )

    norm = _env_path("LBENCH_NORM_STATS")
    validate_norm_stats(norm)
    if scenario == "sft-smoke":
        base = _env_path("PI05_BASE_CHECKPOINT", "dir")
        has_base_weights = (
            bool(list(base.glob("*.safetensors")))
            or (base / "model_state_dict/full_weights.pt").is_file()
        )
        if not has_base_weights:
            raise FileNotFoundError(
                f"PI05_BASE_CHECKPOINT has no model weights: {base}"
            )
        repo = os.environ.get("LBENCH_LEROBOT_REPO")
        if not repo:
            raise ValueError(
                "required environment variable LBENCH_LEROBOT_REPO is not set"
            )
        dataset = _dataset_root(repo)
        metadata = _validate_metadata(dataset)
        lerobot_runtime = validate_lerobot_runtime(metadata)
        codebase_version = str(metadata.get("codebase_version", ""))
        storage_format = _storage_format(codebase_version)
        task_episode_counts = (
            _v21_task_episode_counts(dataset)
            if storage_format == "v2.1"
            else None
        )
        for key in ("total_episodes", "total_frames", "total_tasks"):
            if int(metadata.get(key, 0)) < 1:
                raise ValueError(f"LeRobot metadata {key} must be positive")
        if int(metadata["total_tasks"]) != 8:
            raise ValueError(
                "the joint LBench M8 SFT stage requires exactly 8 dataset tasks, "
                f"got {metadata['total_tasks']}"
            )
        return {
            "norm_stats": str(norm),
            "base_checkpoint": str(base),
            "dataset": str(dataset),
            "dataset_codebase_version": codebase_version,
            "dataset_storage_format": storage_format,
            "dataset_task_episode_counts": task_episode_counts,
            "lerobot_runtime": lerobot_runtime,
        }
    lbench = _env_path("LBENCH_ROOT", "dir")
    if not (lbench / "simbox_eval").is_dir():
        raise FileNotFoundError(f"LBENCH_ROOT lacks simbox_eval: {lbench}")
    isaac = _env_path("LBENCH_ISAAC_PYTHON", "executable")
    rlinf_python = Path(os.path.abspath(sys.executable))
    isaac_prefix = isaac.parent.parent if isaac.parent.name == "bin" else isaac.parent
    rlinf_prefix = (
        rlinf_python.parent.parent
        if rlinf_python.parent.name == "bin"
        else rlinf_python.parent
    )
    shared_python = isaac == rlinf_python or isaac_prefix == rlinf_prefix
    if shared_python and not allow_shared_python:
        raise ValueError(
            "LBENCH_ISAAC_PYTHON points to the RLinf launcher environment. "
            "Use a separate Isaac environment, or pass --allow-shared-python "
            "only for a lightweight fake-backend check."
        )
    task = _env_path("LBENCH_TASK_CONFIG")
    task_contract = _validate_task_config(
        task, require_multitask=scenario in {"dual-ppo", "eval-8", "ppo-8"}
    )
    if scenario in {"eval-8", "ppo-8"} and task_contract["task_count"] != 8:
        raise ValueError(
            f"{scenario} requires exactly 8 LBench tasks, got "
            f"{task_contract['task_count']}"
        )
    checkpoint = _env_path("LBENCH_SFT_CHECKPOINT", "dir")
    weight_candidates = (
        "actor/model_state_dict/full_weights.pt",
        "model_state_dict/full_weights.pt",
    )
    has_checkpoint_weights = bool(list(checkpoint.glob("*.safetensors"))) or any(
        (checkpoint / name).is_file() for name in weight_candidates
    )
    if not has_checkpoint_weights:
        raise FileNotFoundError(
            f"LBENCH_SFT_CHECKPOINT has no model weights: {checkpoint}"
        )
    prompt_contract = _validate_dataset_prompt_alignment(task_contract)
    return {
        "norm_stats": str(norm),
        "lbench_root": str(lbench),
        "isaac_python": str(isaac),
        "rlinf_python": str(rlinf_python),
        "python_environments_isolated": not shared_python,
        "task_config": str(task),
        **task_contract,
        **prompt_contract,
        "checkpoint": str(checkpoint),
    }


def _hydra_override_devices(devices: list[str]) -> str:
    return "env.train.isaac_cuda_visible_devices=[" + ",".join(devices) + "]"


def _isaac_runtime_env(isaac_python: Path) -> dict[str, str]:
    """Resolve Isaac runtime paths from the explicitly selected interpreter.

    ``evaluations/run_eval.sh`` supplies generic standalone-install placeholders
    when these variables are absent.  Those placeholders shadow the paths that
    the pip/Conda ``isaacsim`` package would otherwise discover for itself and
    make ``from isaacsim import SimulationApp`` return ``None``.  Prefer the
    package colocated with ``LBENCH_ISAAC_PYTHON`` so an RLinf parent environment
    cannot select an unrelated Isaac installation.
    """

    interpreter_prefix = (
        isaac_python.parent.parent
        if isaac_python.parent.name == "bin"
        else isaac_python.parent
    )
    candidates = sorted(
        (interpreter_prefix / "lib").glob("python*/site-packages/isaacsim")
    )
    for isaac_path in candidates:
        exp_path = isaac_path / "apps"
        carb_app_path = isaac_path / "kit"
        if isaac_path.is_dir() and exp_path.is_dir() and carb_app_path.is_dir():
            runtime_env = {
                "ISAAC_PATH": str(isaac_path.resolve()),
                "EXP_PATH": str(exp_path.resolve()),
                "CARB_APP_PATH": str(carb_app_path.resolve()),
            }
            # pydrake bundles another libgomp.  Loading it after Isaac's many
            # native extensions can exhaust static TLS slots.  Pass this as an
            # inert LBench variable here; the RLinf bridge applies LD_PRELOAD
            # only to external Isaac workers, not to model processes.
            libgomp = interpreter_prefix / "lib" / "libgomp.so.1"
            if libgomp.is_file():
                runtime_env["LBENCH_ISAAC_LD_PRELOAD"] = str(libgomp.resolve())
            return runtime_env

    configured = {
        "ISAAC_PATH": os.environ.get("ISAAC_PATH", ""),
        "EXP_PATH": os.environ.get("EXP_PATH", ""),
        "CARB_APP_PATH": os.environ.get("CARB_APP_PATH", ""),
    }
    if all(
        value and Path(value).expanduser().is_dir() for value in configured.values()
    ):
        return {
            key: str(Path(value).expanduser().resolve())
            for key, value in configured.items()
        }
    raise FileNotFoundError(
        "could not locate a complete Isaac runtime beside "
        f"LBENCH_ISAAC_PYTHON={isaac_python}; expected an isaacsim package with "
        "apps/ and kit/, or valid ISAAC_PATH/EXP_PATH/CARB_APP_PATH variables"
    )


def _target_global_batch(scenario: str, world_size: int) -> int | None:
    if scenario == "sft-smoke":
        return world_size
    if scenario == "dual-ppo":
        return math.lcm(2, world_size)
    if scenario == "ppo-8":
        return math.lcm(32, world_size)
    return None


def _hydra_component_placement(scenario: str, rlinf_gpus: list[str]) -> str:
    """Pin RLinf workers to the physical GPU ranks selected by the launcher.

    RLinf's hardware probe enumerates every accelerator visible to the Ray node,
    so setting ``CUDA_VISIBLE_DEVICES`` only on the driver is not sufficient to
    change a config whose component placement is ``all``.  In particular, a
    one-environment evaluation on a two-GPU host would otherwise create two env
    processes and fail the environment-count divisibility check.
    """

    try:
        hardware_ranks = sorted(int(value) for value in rlinf_gpus)
    except ValueError as exc:
        raise ValueError(
            f"RLinf GPU ids must be integer hardware ranks: {rlinf_gpus}"
        ) from exc
    if not hardware_ranks or len(hardware_ranks) != len(set(hardware_ranks)):
        raise ValueError(f"RLinf GPU ids must be non-empty and unique: {rlinf_gpus}")

    components = (
        r"env\,rollout"
        if scenario in {"single-eval", "eval-8"}
        else r"actor\,env\,rollout"
    )
    # Commas delimit both the component key and rank-map entries, so Hydra's
    # override grammar requires them to be escaped inside the inline mapping.
    rank_map = r"\,".join(str(rank) for rank in hardware_ranks)
    return f"cluster.component_placement={{{components}:{rank_map}}}"


def _hydra_overrides(
    scenario: str,
    rlinf_gpus: list[str],
    isaac_gpus: list[str],
    output: Path,
    collection_format: str = "pickle",
    eval_rollout_epochs: int = 1,
) -> list[str]:
    """Build the exact overrides shared by preflight composition and launch."""

    world_size = len(rlinf_gpus)
    if scenario == "sft-smoke":
        return [
            _hydra_component_placement(scenario, rlinf_gpus),
            "runner.max_steps=5",
            "runner.save_interval=5",
            f"runner.logger.log_path={output}",
            "runner.logger.experiment_name=m8_sft_smoke",
            "actor.micro_batch_size=1",
            f"actor.global_batch_size={_target_global_batch(scenario, world_size)}",
            # The acceptance smoke is intentionally runnable on a reduced-memory
            # model profile. Full-parameter SFT remains available from the base
            # config by omitting this override on a suitably sized host.
            "actor.model.openpi.train_expert_only=true",
        ]
    if scenario in {"single-eval", "eval-8"}:
        count = 1 if scenario == "single-eval" else 8
        overrides = [
            _hydra_component_placement(scenario, rlinf_gpus),
            f"env.eval.total_num_envs={count}",
            f"env.eval.rollout_epoch={eval_rollout_epochs}",
            "env.eval.task_indices=[0]" if count == 1 else "env.eval.task_indices=null",
            (
                "env.eval.fixed_seeds=[0]"
                if count == 1
                else "env.eval.fixed_seeds=[0,1,2,3,4,5,6,7]"
            ),
            "env.eval.isaac_cuda_visible_devices=[" + ",".join(isaac_gpus) + "]",
            "env.eval.data_collection.enabled=true",
            f"env.eval.data_collection.export_format={collection_format}",
            f"runner.logger.log_path={output / scenario}",
        ]
        if collection_format == "lerobot":
            overrides.append("env.eval.data_collection.lerobot_schema=lbench_joint")
        return overrides
    count = 2 if scenario == "dual-ppo" else 8
    overrides = [
        _hydra_component_placement(scenario, rlinf_gpus),
        f"env.train.total_num_envs={count}",
        f"env.eval.total_num_envs={count}",
        "env.train.rollout_epoch=1"
        if scenario == "dual-ppo"
        else "env.train.rollout_epoch=8",
        "runner.max_steps=2" if scenario == "dual-ppo" else "runner.max_steps=200",
        (
            "env.train.max_episode_steps=40"
            if scenario == "dual-ppo"
            else "env.train.max_episode_steps=1200"
        ),
        (
            "env.train.max_steps_per_rollout_epoch=40"
            if scenario == "dual-ppo"
            else "env.train.max_steps_per_rollout_epoch=1200"
        ),
        "actor.micro_batch_size=1",
        f"actor.global_batch_size={_target_global_batch(scenario, world_size)}",
        (
            "env.train.data_collection.enabled=true"
            if scenario == "dual-ppo"
            else "env.train.data_collection.enabled=false"
        ),
        f"env.train.data_collection.export_format={collection_format}",
        _hydra_override_devices(isaac_gpus),
        "env.eval.isaac_cuda_visible_devices=[" + ",".join(isaac_gpus) + "]",
        f"runner.logger.log_path={output}",
        (
            "runner.logger.experiment_name=m8_dual_ppo_smoke"
            if scenario == "dual-ppo"
            else "runner.logger.experiment_name=lbench_ppo_openpi_pi05"
        ),
    ]
    if collection_format == "lerobot":
        overrides.append("env.train.data_collection.lerobot_schema=lbench_joint")
    return overrides


def _compose_and_validate(
    scenario: str,
    env_count: int,
    world_size: int,
    overrides: list[str] | None = None,
) -> dict:
    """Fully compose Hydra defaults and validate the M8 shape/resource contract."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    os.environ.setdefault("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    if scenario == "sft-smoke":
        config_dir = ROOT / "examples/sft/config"
        config_name = "lbench_sft_openpi_pi05"
    elif scenario in {"single-eval", "eval-8"}:
        config_dir = ROOT / "evaluations/lbench"
        config_name = "lbench_openpi_pi05_eval"
    else:
        config_dir = ROOT / "examples/embodiment/config"
        config_name = "lbench_ppo_openpi_pi05"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    OmegaConf.to_container(cfg, resolve=True)

    model = cfg.actor.model if "actor" in cfg else cfg.rollout.model
    if str(model.openpi.config_name) != "pi05_lbench_joint":
        raise ValueError("M8 requires openpi.config_name=pi05_lbench_joint")
    if int(model.openpi.action_horizon) != 50:
        raise ValueError("M8 requires model action_horizon=50")
    execute_horizon = int(model.num_action_chunks)
    expected_execute_horizon = 50 if scenario == "sft-smoke" else 20
    if execute_horizon != expected_execute_horizon:
        raise ValueError(
            f"M8 {scenario} requires execute horizon={expected_execute_horizon}"
        )
    if (
        int(model.action_dim) != 8
        or int(model.openpi.action_env_dim) != 8
        or int(model.openpi.action_chunk) != execute_horizon
        or int(model.openpi.num_images_in_input) != 3
        or not bool(model.use_proprio)
    ):
        raise ValueError(
            "M8 requires three images, proprioception, and an 8-D LBench "
            "action contract with openpi.action_chunk=num_action_chunks"
        )
    env_cfg = None
    chunk_steps_per_rollout_epoch = None
    if scenario != "sft-smoke":
        env_cfg = (
            cfg.env.eval if scenario in {"single-eval", "eval-8"} else cfg.env.train
        )
        camera_mapping = OmegaConf.to_container(
            env_cfg.camera_mapping, resolve=True
        )
        expected_camera_mapping = {
            "main": "obs_camera",
            "wrist": "obs_camera_2",
            "additional": "realsense",
        }
        if camera_mapping != expected_camera_mapping:
            raise ValueError(
                "M8 runtime camera mapping must match the SFT feature contract: "
                f"expected={expected_camera_mapping}, got={camera_mapping}"
            )
        episode_horizon = int(env_cfg.max_episode_steps)
        rollout_horizon = int(env_cfg.max_steps_per_rollout_epoch)
        backend_horizon = int(env_cfg.backend_config.simbox_eval_config.max_step)
        if backend_horizon != episode_horizon:
            raise ValueError(
                "LBench simulator and RL episode horizons must match: "
                f"backend max_step={backend_horizon}, "
                f"max_episode_steps={episode_horizon}"
            )
        expected_horizon = 40 if scenario == "dual-ppo" else 1200
        if (
            episode_horizon != expected_horizon
            or rollout_horizon != expected_horizon
        ):
            raise ValueError(
                f"{scenario} requires aligned episode/rollout horizons of "
                f"{expected_horizon}, got episode={episode_horizon}, "
                f"rollout={rollout_horizon}"
            )
        # EnvWorker converts the primitive-step window to policy/chunk steps
        # with integer division. Reject a remainder instead of silently running
        # fewer simulator steps than the configured episode horizon.
        if execute_horizon < 1 or rollout_horizon % execute_horizon:
            raise ValueError(
                "LBench rollout horizon must be an exact multiple of the "
                "executed action horizon: "
                f"{rollout_horizon} % {execute_horizon} != 0"
            )
        chunk_steps_per_rollout_epoch = rollout_horizon // execute_horizon
    if scenario in {"dual-ppo", "ppo-8"}:
        if not bool(model.openpi.train_expert_only):
            raise ValueError(
                "LBench Pi0.5 PPO must freeze the PaliGemma VLM with "
                "actor.model.openpi.train_expert_only=true"
            )
        if not bool(cfg.actor.fsdp_config.use_orig_params):
            raise ValueError(
                "expert-only LBench PPO requires actor.fsdp_config.use_orig_params=true"
            )
        if bool(cfg.env.train.auto_reset) or bool(cfg.env.train.ignore_terminations):
            raise ValueError(
                "LBench PPO requires auto_reset=false and ignore_terminations=false"
            )
        if not bool(cfg.algorithm.balanced_rl):
            raise ValueError("LBench PPO requires algorithm.balanced_rl=true")
        if int(cfg.algorithm.update_epoch) != 4:
            raise ValueError(
                "LBench Pi0.5 PPO must retain the RLinf LIBERO-Long Pi0.5 "
                "update_epoch=4 baseline"
            )
        reward_adapter = cfg.env.train.reward_adapter
        if str(reward_adapter.type) != "libero_sparse_success":
            raise ValueError(
                "LBench baseline PPO must use the LIBERO-compatible "
                "reward_adapter.type=libero_sparse_success"
            )
        success_bonus = float(reward_adapter.success_bonus)
        reward_coef = float(cfg.algorithm.reward_coef)
        if not math.isclose(success_bonus, reward_coef, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "LIBERO-compatible LBench success_bonus must equal "
                f"algorithm.reward_coef: {success_bonus} != {reward_coef}"
            )
    if scenario != "sft-smoke" and env_count % world_size:
        raise ValueError(
            f"{scenario} uses {env_count} environments, which is not divisible "
            f"by the {world_size} RLinf env/rollout ranks. Choose a divisor of "
            f"{env_count} in RLINF_GPUS."
        )
    target_global_batch = _target_global_batch(scenario, world_size)
    if target_global_batch is not None:
        global_batch = int(cfg.actor.global_batch_size)
        if global_batch != target_global_batch or global_batch % world_size:
            raise ValueError(
                f"global_batch_size={global_batch} does not match the M8 target "
                f"{target_global_batch} for world_size={world_size}"
            )
    if overrides is not None and scenario != "sft-smoke":
        assert env_cfg is not None
        if int(env_cfg.total_num_envs) != env_count:
            raise ValueError("Hydra did not apply the requested environment count")
        expected_checkpoint = Path(os.environ["LBENCH_SFT_CHECKPOINT"]).resolve()
        configured_checkpoint = Path(str(model.model_path)).expanduser().resolve()
        if configured_checkpoint != expected_checkpoint:
            raise ValueError(
                "composed model checkpoint does not match LBENCH_SFT_CHECKPOINT: "
                f"{configured_checkpoint} != {expected_checkpoint}"
            )
        if scenario in {"single-eval", "eval-8"} and not bool(
            cfg.env.eval.data_collection.enabled
        ):
            raise ValueError("M8 evaluation must enable episode data collection")
        if scenario in {"single-eval", "eval-8"} and bool(cfg.env.eval.auto_reset):
            raise ValueError(
                "LBench evaluation requires auto_reset=false so every env slot "
                "contributes one episode per rollout epoch"
            )
        if scenario == "dual-ppo" and not bool(cfg.env.train.data_collection.enabled):
            raise ValueError("M8 dual PPO smoke must exercise episode data collection")
        if scenario == "ppo-8" and bool(cfg.env.train.data_collection.enabled):
            raise ValueError(
                "M8 formal PPO must not archive all camera frames by default"
            )
    if overrides is not None:
        if scenario == "sft-smoke" and not bool(model.openpi.train_expert_only):
            raise ValueError(
                "the M8 SFT smoke must use the reduced-memory expert-only profile"
            )
        if (
            scenario == "sft-smoke"
            and bool(model.openpi.train_expert_only)
            and not bool(cfg.actor.fsdp_config.use_orig_params)
        ):
            raise ValueError(
                "expert-only LBench SFT requires actor.fsdp_config.use_orig_params=true"
            )
        expected_max_steps = {
            "sft-smoke": 5,
            "dual-ppo": 2,
            "ppo-8": 200,
        }.get(scenario)
        if (
            expected_max_steps is not None
            and int(cfg.runner.max_steps) != expected_max_steps
        ):
            raise ValueError(
                f"{scenario} requires runner.max_steps={expected_max_steps}"
            )
    rollout_epoch = (
        None
        if scenario == "sft-smoke"
        else int(
            cfg.env.eval.rollout_epoch
            if scenario in {"single-eval", "eval-8"}
            else cfg.env.train.rollout_epoch
        )
    )
    max_primitive_env_steps_per_global_step = (
        env_count * rollout_epoch * int(env_cfg.max_steps_per_rollout_epoch)
        if scenario in {"dual-ppo", "ppo-8"}
        and rollout_epoch is not None
        and env_cfg is not None
        else None
    )
    return {
        "config": config_name,
        "component_placement": OmegaConf.to_container(
            cfg.cluster.component_placement, resolve=True
        ),
        "action_horizon": int(model.openpi.action_horizon),
        "execute_horizon": execute_horizon,
        "train_expert_only": bool(model.openpi.train_expert_only),
        "rollout_epoch": rollout_epoch,
        "episode_horizon": (
            None if env_cfg is None else int(env_cfg.max_episode_steps)
        ),
        "rollout_horizon": (
            None if env_cfg is None else int(env_cfg.max_steps_per_rollout_epoch)
        ),
        "chunk_steps_per_rollout_epoch": chunk_steps_per_rollout_epoch,
        "planned_eval_episode_attempts": (
            env_count * rollout_epoch
            if scenario in {"single-eval", "eval-8"}
            and rollout_epoch is not None
            else None
        ),
        "max_primitive_env_steps_per_global_step": (
            max_primitive_env_steps_per_global_step
        ),
        "max_primitive_env_steps_total": (
            max_primitive_env_steps_per_global_step * int(cfg.runner.max_steps)
            if max_primitive_env_steps_per_global_step is not None
            else None
        ),
        "global_batch_size": target_global_batch,
        "reward_adapter": (
            None
            if scenario not in {"dual-ppo", "ppo-8"}
            else str(cfg.env.train.reward_adapter.type)
        ),
    }


def _commands(
    scenario: str,
    rlinf_gpus: list[str],
    isaac_gpus: list[str],
    output: Path,
    collection_format: str = "pickle",
    eval_rollout_epochs: int = 1,
):
    python = sys.executable
    overrides = _hydra_overrides(
        scenario,
        rlinf_gpus,
        isaac_gpus,
        output,
        collection_format,
        eval_rollout_epochs,
    )
    if scenario == "sft-smoke":
        common = [
            python,
            str(ROOT / "examples/sft/train_vla_sft.py"),
            "--config-name",
            "lbench_sft_openpi_pi05",
            *overrides,
        ]
        checkpoint = output / "m8_sft_smoke/checkpoints/global_step_5"
        reload_cmd = [
            "runner.max_steps=6" if item == "runner.max_steps=5" else item
            for item in common
            if item != "runner.save_interval=5"
        ]
        reload_cmd += [f"runner.resume_dir={checkpoint}", "runner.save_interval=-1"]
        return [common, reload_cmd]
    if scenario in {"single-eval", "eval-8"}:
        command = [
            "bash",
            str(ROOT / "evaluations/run_eval.sh"),
            "lbench",
            "lbench_openpi_pi05_eval",
            *overrides,
        ]
        return [command]
    command = [
        python,
        str(ROOT / "examples/embodiment/train_embodied_agent.py"),
        "--config-name",
        "lbench_ppo_openpi_pi05",
        *overrides,
    ]
    return [command]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-no-gpu", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--allow-shared-python", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--allow-existing-ray",
        action="store_true",
        help="Allow attaching to a Ray cluster already started with the correct GPU mask.",
    )
    parser.add_argument("--rlinf-gpus", default=os.environ.get("RLINF_GPUS", "0"))
    parser.add_argument(
        "--isaac-gpus", default=os.environ.get("LBENCH_ISAAC_GPUS", "1")
    )
    parser.add_argument("--output", type=Path, default=ROOT / "../results/lbench_m8")
    parser.add_argument(
        "--collection-format",
        choices=("pickle", "lerobot"),
        default="pickle",
        help=(
            "Episode archive format for collection-enabled stages. 'lerobot' "
            "uses the LBench joint feature schema and the storage version of "
            "the installed LeRobot writer; formal ppo-8 collection remains "
            "disabled regardless of this option."
        ),
    )
    parser.add_argument(
        "--eval-rollout-epochs",
        type=int,
        default=1,
        help=(
            "Number of explicit one-episode rounds for single-eval/eval-8 "
            "(passing 8 to eval-8 uses seeds 0..7 for every task)."
        ),
    )
    args = parser.parse_args()
    if args.scenario == "sft-smoke" and args.collection_format != "pickle":
        parser.error("--collection-format applies only to eval/PPO stages")
    if args.eval_rollout_epochs < 1:
        parser.error("--eval-rollout-epochs must be positive")
    if args.scenario not in {"single-eval", "eval-8"} and args.eval_rollout_epochs != 1:
        parser.error("--eval-rollout-epochs applies only to evaluation stages")
    rlinf_gpus = _csv(args.rlinf_gpus)
    isaac_gpus = _csv(args.isaac_gpus)
    resources = _common_checks(
        args.scenario,
        rlinf_gpus,
        isaac_gpus,
        args.allow_no_gpu,
        args.allow_shared_python,
    )
    isaac_runtime_env: dict[str, str] = {}
    if args.scenario != "sft-smoke":
        isaac_runtime_env = _isaac_runtime_env(
            Path(str(resources["isaac_python"]))
        )
        resources["isaac_runtime_env"] = isaac_runtime_env
    env_count = {
        "sft-smoke": len(rlinf_gpus),
        "single-eval": 1,
        "dual-ppo": 2,
        "eval-8": 8,
        "ppo-8": 8,
    }[args.scenario]
    output = args.output.expanduser().resolve()
    overrides = _hydra_overrides(
        args.scenario,
        rlinf_gpus,
        isaac_gpus,
        output,
        args.collection_format,
        args.eval_rollout_epochs,
    )
    config = _compose_and_validate(args.scenario, env_count, len(rlinf_gpus), overrides)
    commands = _commands(
        args.scenario,
        rlinf_gpus,
        isaac_gpus,
        output,
        args.collection_format,
        args.eval_rollout_epochs,
    )
    existing_ray = _existing_ray_cluster()
    if existing_ray and not args.allow_existing_ray and not args.dry_run:
        raise RuntimeError(
            "an existing Ray cluster is reachable. It may have captured a GPU "
            "mask that includes Isaac GPUs. Stop it first, or restart it with "
            "CUDA_VISIBLE_DEVICES=$RLINF_GPUS. Pass --allow-existing-ray only "
            "after verifying that isolation."
        )
    report = {
        "scenario": args.scenario,
        "dry_run": args.dry_run,
        "world_size": len(rlinf_gpus),
        "rlinf_gpus": rlinf_gpus,
        "isaac_gpus": [] if args.scenario == "sft-smoke" else isaac_gpus,
        "resources": resources,
        "config": config,
        "existing_ray_cluster": existing_ray,
        "eval_rollout_epochs": (
            args.eval_rollout_epochs
            if args.scenario in {"single-eval", "eval-8"}
            else None
        ),
        "commands": commands,
    }
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return
    _check_fresh_output(args.scenario, output)
    run_env = os.environ.copy()
    run_env["CUDA_VISIBLE_DEVICES"] = ",".join(rlinf_gpus)
    run_env["RLINF_GPUS"] = ",".join(rlinf_gpus)
    if args.scenario != "sft-smoke":
        run_env["LBENCH_ISAAC_GPUS"] = ",".join(isaac_gpus)
        run_env.update(isaac_runtime_env)
    run_env["EMBODIED_PATH"] = str(ROOT / "examples/embodiment")
    run_env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), run_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    output.mkdir(parents=True, exist_ok=True)
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=run_env, check=True)


if __name__ == "__main__":
    main()
