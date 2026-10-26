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

"""CPU contracts for the two-task PPO profile and foreground launcher."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.distributed.fsdp import MixedPrecision

from rlinf.config import torch_dtype_from_precision
from rlinf.envs.lbench.lbench_env import LBenchEnv
from rlinf.utils.metric_utils import compute_evaluate_metrics
from toolkits import lbench_two_task_ppo as launcher

ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = "lbench_ppo_openpi_pi05_two_tasks_3gpu"
TASKS = ("stack_bowls", "store_objects_in_the_drawer_0")
TASK_INDICES = (5, 6)
LBENCH_ROOT = Path(
    os.environ.get("LBENCH_ROOT") or ROOT.parent / "LBench_multi_env"
).expanduser()


@pytest.fixture
def profile(monkeypatch, tmp_path):
    """Compose the actual profile without opening models or simulators."""
    values = {
        "EMBODIED_PATH": str(ROOT / "examples" / "embodiment"),
        "LBENCH_ROOT": str(LBENCH_ROOT),
        "LBENCH_ISAAC_PYTHON": sys.executable,
        "LBENCH_TASK_CONFIG": str(
            LBENCH_ROOT / "configs" / "simbox" / "lbench_joint_8tasks.yaml"
        ),
        "LBENCH_SFT_CHECKPOINT": str(tmp_path / "sft"),
        "LBENCH_NORM_STATS": str(tmp_path / "norm_stats.json"),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(
        config_dir=str(ROOT / "examples" / "embodiment" / "config"),
        version_base="1.1",
    ):
        cfg = compose(config_name=CONFIG_NAME)
    OmegaConf.resolve(cfg)
    return cfg


def test_profile_places_actor_and_rollout_on_separate_visible_devices(profile):
    placement = OmegaConf.to_container(profile.cluster.component_placement)
    assert set(placement) == {"actor", "env,rollout"}
    assert str(placement["actor"]) == "0"
    assert str(placement["env,rollout"]) == "1"
    assert profile.cluster.num_nodes == 1
    assert profile.rollout.pipeline_stage_num == 1
    for env_cfg in (profile.env.train, profile.env.eval):
        assert list(env_cfg.isaac_cuda_visible_devices) == [2]
        assert env_cfg.total_num_envs == 2
        assert env_cfg.group_size == 1
        assert env_cfg.get("enable_offload", False) is False
    # EnvWorker initializes both pools when periodic evaluation is enabled.
    assert profile.runner.val_check_interval > 0
    assert profile.env.train.total_num_envs + profile.env.eval.total_num_envs == 4


def test_profile_preserves_sft_action_and_normalization_contract(profile):
    model = profile.actor.model
    # OpenPI retains the generic dtype=None setting and owns parameter casting.
    assert model.precision is None
    assert model.openpi.parameter_precision == "float32"
    # Exercise the same conversion used by FSDPStrategy. OpenPI's precision
    # label is not itself a valid FSDP precision spelling.
    dtype_fields = ("param_dtype", "reduce_dtype", "buffer_dtype")
    dtypes = {
        field: torch_dtype_from_precision(
            profile.actor.fsdp_config.mixed_precision[field]
        )
        for field in dtype_fields
    }
    assert set(dtypes.values()) == {torch.float32}
    mixed_precision = MixedPrecision(**dtypes)
    assert all(
        getattr(mixed_precision, field) == torch.float32 for field in dtype_fields
    )
    assert profile.actor.fsdp_config.amp_autocast.enabled is False
    assert profile.rollout.model.precision == model.precision
    assert profile.rollout.model.model_path == model.model_path
    assert model.num_steps == model.openpi.num_steps == 10
    assert model.num_action_chunks == model.openpi.action_chunk == 20
    assert model.action_dim == model.openpi.action_env_dim == 8
    assert model.openpi.action_horizon == 50
    assert model.openpi.num_images_in_input == 3
    assert model.openpi_data.lerobot_compatible is True
    assert profile.runner.resume_dir is None


@pytest.mark.parametrize("field", ["param_dtype", "reduce_dtype", "buffer_dtype"])
def test_fsdp_preflight_rejects_openpi_precision_spelling(profile, field):
    config = OmegaConf.to_container(profile, resolve=True)
    config["actor"]["fsdp_config"]["mixed_precision"][field] = "float32"
    with pytest.raises(ValueError, match=field):
        launcher.validate_fsdp_precision(config)


def test_profile_rollout_batch_and_periodic_evaluation_are_complete(profile):
    train, evaluation = profile.env.train, profile.env.eval
    for env_cfg in (train, evaluation):
        assert list(env_cfg.task_indices) == list(TASK_INDICES)
        assert env_cfg.max_episode_steps == env_cfg.max_steps_per_rollout_epoch == 1200
        assert env_cfg.auto_reset is False
        assert env_cfg.ignore_terminations is False
        assert env_cfg.task_sampling == "balanced_round_robin"
        assert env_cfg.data_collection.enabled is False
        assert env_cfg.video_cfg.save_video is False
    assert train.rollout_epoch == 4
    assert train.reward_adapter.type == "libero_sparse_success"
    assert evaluation.rollout_epoch == 10
    assert list(evaluation.fixed_seeds) == list(range(10))
    assert list(train.fixed_seeds) == []
    assert train.seed == 1000
    assert profile.actor.global_batch_size == 32
    assert profile.actor.micro_batch_size == 1
    assert profile.algorithm.update_epoch == 2
    chunks_per_episode, remainder = divmod(
        train.max_steps_per_rollout_epoch, profile.actor.model.num_action_chunks
    )
    assert remainder == 0
    rollout_batch = train.total_num_envs * train.rollout_epoch * chunks_per_episode
    assert rollout_batch == 480
    assert rollout_batch % profile.actor.global_batch_size == 0
    assert profile.actor.global_batch_size % profile.actor.micro_batch_size == 0
    assert evaluation.total_num_envs * evaluation.rollout_epoch == 2 * 10
    assert profile.runner.max_steps == 20
    assert profile.runner.val_check_interval == profile.runner.save_interval == 5


class _ResetOnlyClient:
    """Fake transport; task scheduling and EpisodeSpec remain production code."""

    def __init__(self, task_names, num_envs):
        self.task_names = task_names
        self.num_envs = num_envs
        self.specs = {}
        self.raw = []

    def list_tasks(self):
        return self.task_names

    def reset(self, env_ids=None, episode_specs=None):
        assert env_ids is None or list(env_ids) == list(range(self.num_envs))
        self.specs = dict(episode_specs)
        self.raw = []
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        for env_id in range(self.num_envs):
            spec = self.specs[env_id]
            self.raw.append(
                {
                    "robot": {
                        "context": {
                            "env_id": env_id,
                            "worker_id": spec.worker_id,
                            "episode_id": spec.episode_id,
                            "task_index": spec.task_index,
                        },
                        "joints_state": {"positions": [0.0] * 9},
                        "sensors": {
                            name: {"rgb": image}
                            for name in ("obs_camera", "obs_camera_2", "realsense")
                        },
                        "instruction": self.task_names[spec.task_index],
                        "metric": {"episode_sr": 0.0, "current_stage_id": 0},
                    }
                }
            )
        return [{} for _ in self.raw], copy.deepcopy(self.raw)

    def step(self, commands):
        infos = []
        for env_id, raw in enumerate(self.raw):
            success = env_id == 0
            raw["robot"]["metric"]["episode_sr"] = float(success)
            infos.append({"reward_components": {"success": float(success)}})
        return (
            [{} for _ in self.raw],
            [0.0] * self.num_envs,
            [True] * self.num_envs,
            [False] * self.num_envs,
            infos,
            copy.deepcopy(self.raw),
        )

    def deactivate(self, env_ids):
        pass

    def close(self):
        pass


@pytest.fixture
def make_env(profile):
    task_path = Path(profile.env.eval.backend_config.simbox_eval_config.task_cfg_path)
    if not task_path.is_file():
        pytest.skip("Set LBENCH_ROOT to run the external LBench CPU protocol checks")
    tasks = OmegaConf.load(task_path).tasks
    task_names = [str(task.data.task_dir) for task in tasks]
    assert tuple(task_names[index] for index in TASK_INDICES) == TASKS
    environments = []

    def factory(mode):
        cfg = profile.env[mode]
        client = _ResetOnlyClient(task_names, cfg.total_num_envs)
        env = LBenchEnv(
            cfg=cfg,
            num_envs=cfg.total_num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        environments.append(env)
        return env, client

    yield factory
    for env in environments:
        env.close()


def _reset_rounds(env, client, rounds):
    scheduled = []
    for _ in range(rounds):
        env.reset()
        scheduled.extend(client.specs[index] for index in range(env.num_envs))
    env.update_reset_state_ids()
    return scheduled


def test_actual_eval_scheduler_repeats_identical_task_seed_coverage(profile, make_env):
    env, client = make_env("eval")
    first = _reset_rounds(env, client, profile.env.eval.rollout_epoch)
    second = _reset_rounds(env, client, profile.env.eval.rollout_epoch)

    def pairs(specs):
        return [(spec.task_index, spec.seed) for spec in specs]

    expected = {(task, seed) for task in TASK_INDICES for seed in range(10)}
    assert len(first) == len(second) == len(expected) == 20
    assert set(pairs(first)) == set(pairs(second)) == expected
    assert pairs(first) == pairs(second)
    assert len({spec.episode_id for spec in first + second}) == 40


def test_actual_train_scheduler_balances_tasks_and_advances_seeds(profile, make_env):
    env, client = make_env("train")
    first = _reset_rounds(env, client, profile.env.train.rollout_epoch)
    second = _reset_rounds(env, client, profile.env.train.rollout_epoch)
    for batch in (first, second):
        assert Counter(spec.task_index for spec in batch) == {5: 4, 6: 4}
    assert len({(spec.task_index, spec.seed) for spec in first + second}) == 16
    assert len({spec.episode_id for spec in first + second}) == 16


def test_actual_episode_metrics_keep_task_success_denominators_separate(make_env):
    env, _ = make_env("eval")
    env.reset()
    _, _, terminated, truncated, infos = env.step(torch.zeros((2, 8)))
    assert terminated.tolist() == [True, True]
    assert not truncated.any()
    metrics = compute_evaluate_metrics([infos["episode"]])
    assert float(metrics[f"task/{TASKS[0]}/success"]) == 1.0
    assert float(metrics[f"task/{TASKS[1]}/success"]) == 0.0
    assert float(metrics["success_once"]) == 0.5
    assert metrics["num_trajectories"] == 2
    for task in TASKS:
        assert float(metrics[f"task/{task}/count"]) == 1.0


@pytest.fixture
def checkpoint(tmp_path):
    """Write a tiny real torch checkpoint, including its bundled norm assets."""
    root = tmp_path / "checkpoint"
    weights = root / "actor" / "model_state_dict" / "full_weights.pt"
    weights.parent.mkdir(parents=True)
    torch.save({"expert.weight": torch.ones(2, dtype=torch.float32)}, weights)
    stats = root / "lbench" / "norm_stats.json"
    stats.parent.mkdir()
    stats.write_text(
        json.dumps({"norm_stats": {"actions": {"mean": [0.0], "std": [1.0]}}})
    )
    manifest = {
        "config_name": "pi05_lbench_joint",
        "action_horizon": 50,
        "action_dim": 8,
        "norm_stats": ["lbench/norm_stats.json"],
        "compatibility": {
            "lerobot_compatible": True,
            "parameter_precision": "float32",
        },
    }
    (root / "openpi_assets.json").write_text(json.dumps(manifest))
    return root


@pytest.mark.parametrize("invalid", ["manifest", "tensor_header"])
def test_launcher_rejects_old_or_mislabeled_bfloat16_checkpoint(checkpoint, invalid):
    if invalid == "manifest":
        manifest_path = checkpoint / "openpi_assets.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["compatibility"]["parameter_precision"] = "openpi"
        manifest_path.write_text(json.dumps(manifest))
    else:
        torch.save(
            {"expert.weight": torch.ones(2, dtype=torch.bfloat16)},
            checkpoint / "actor/model_state_dict/full_weights.pt",
        )
    with pytest.raises(ValueError, match="FP32"):
        launcher.checkpoint_contract(checkpoint)


def test_adapted_h20_checkpoint_requires_an_explicitly_supported_horizon(checkpoint):
    manifest_path = checkpoint / "openpi_assets.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["action_horizon"] = 20
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="FP32"):
        launcher.checkpoint_contract(checkpoint)
    contract = launcher.checkpoint_contract(
        checkpoint, allowed_action_horizons=(20, 50)
    )
    assert contract["manifest"]["action_horizon"] == 20
    assert contract["norm_stats"] == str(checkpoint / "lbench/norm_stats.json")


@pytest.mark.parametrize("gpus", ["0,0,2", "0,00,2", "0,1", "0,1,-1"])
def test_launcher_rejects_invalid_gpu_assignment_before_loading_assets(gpus):
    args = launcher.parse_args(["--gpus", gpus, "--dry-run"])
    with pytest.raises(ValueError, match="GPU"):
        launcher.build_plan(args)


def test_launcher_rejects_duplicate_tasks_before_loading_assets():
    args = launcher.parse_args(["--tasks", TASKS[0], TASKS[0], "--dry-run"])
    with pytest.raises(ValueError, match="two different task"):
        launcher.build_plan(args)


def test_launcher_requires_fresh_output_directory(tmp_path):
    args = launcher.parse_args(["--output", str(tmp_path), "--dry-run"])
    with pytest.raises(FileExistsError, match="fresh"):
        launcher.build_plan(args)


@pytest.fixture
def fake_machine(monkeypatch, profile):
    """Replace only hardware/Isaac inspection, keeping checkpoint and Hydra real."""
    names = [f"unselected_task_{index}" for index in range(8)]
    for index, task in zip(TASK_INDICES, TASKS, strict=True):
        names[index] = task

    def check_resources(scenario, actor_gpus, isaac_gpus, *, allow_no_gpu):
        assert scenario == "dual-ppo"
        assert len(actor_gpus) == 2 and len(isaac_gpus) == 1
        assert allow_no_gpu is True
        return {"task_names": names, "isaac_python": sys.executable}

    monkeypatch.setattr(launcher, "_common_checks", check_resources)
    monkeypatch.setattr(launcher, "_visible_gpu_ids", lambda: {"0"})
    monkeypatch.setattr(launcher, "_isaac_runtime_env", lambda path: {})
    # build_plan deliberately overwrites stale generic exports. Restore them
    # after each test through monkeypatch's environment snapshot.
    for key in (
        "LBENCH_SFT_CHECKPOINT",
        "LBENCH_NORM_STATS",
        "CUDA_VISIBLE_DEVICES",
        "LBENCH_ISAAC_GPUS",
        "RLINF_GPUS",
        "EMBODIED_PATH",
    ):
        monkeypatch.setenv(key, os.environ.get(key, "stale"))


def test_launcher_dry_run_is_read_only_and_composes_explicit_overrides(
    monkeypatch, capsys, checkpoint, tmp_path, fake_machine
):
    output = tmp_path / "new output"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("LBENCH_SFT_CHECKPOINT", "/stale/checkpoint")
    monkeypatch.setenv("LBENCH_NORM_STATS", "/stale/stats")
    monkeypatch.setattr(
        launcher, "run_foreground", lambda *args: pytest.fail("dry-run launched")
    )
    monkeypatch.setattr(
        launcher,
        "_existing_ray_cluster",
        lambda: pytest.fail("dry-run contacted Ray"),
    )
    launcher.main(
        [
            "--dry-run",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--tasks",
            TASKS[1],
            TASKS[0],
            "--max-steps",
            "7",
        ]
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["fsdp_precision"] == {
        "param_dtype": "torch.float32",
        "reduce_dtype": "torch.float32",
        "buffer_dtype": "torch.float32",
    }
    assert not output.exists()
    assert plan["requested_gpus_available"] is False
    assert plan["task_indices"] == [6, 5]
    assert plan["resident_isaac_processes"] == 4
    assert plan["config"]["runner"]["max_steps"] == 7
    assert plan["config"]["runner"]["logger"]["log_path"] == str(output)
    assert plan["config"]["cluster"]["component_placement"] == {
        "actor": 0,
        "env,rollout": 1,
    }
    assert plan["config"]["env"]["train"]["task_indices"] == [6, 5]
    assert plan["config"]["env"]["eval"]["isaac_cuda_visible_devices"] == [2]
    assert plan["checkpoint_contract"]["tensor_count"] == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert os.environ["LBENCH_SFT_CHECKPOINT"] == str(checkpoint)
    assert os.environ["LBENCH_NORM_STATS"] == str(checkpoint / "lbench/norm_stats.json")


def test_launcher_rejects_unknown_task(checkpoint, tmp_path, fake_machine):
    args = launcher.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "new"),
            "--tasks",
            TASKS[0],
            "unknown_task",
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="Unknown LBench tasks"):
        launcher.build_plan(args)


def test_launcher_refuses_regular_run_when_requested_gpus_are_missing(
    monkeypatch, checkpoint, tmp_path, profile
):
    from toolkits import lbench_remote_stage

    monkeypatch.setattr(lbench_remote_stage, "_visible_gpu_ids", lambda: {"0"})
    for key in ("CUDA_VISIBLE_DEVICES", "LBENCH_ISAAC_GPUS", "RLINF_GPUS"):
        monkeypatch.setenv(key, os.environ.get(key, "stale"))
    monkeypatch.setattr(
        launcher, "run_foreground", lambda *args: pytest.fail("missing GPUs launched")
    )
    output = tmp_path / "uncreated"
    with pytest.raises(RuntimeError, match="not all visible"):
        launcher.main(["--checkpoint", str(checkpoint), "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_foreground_child_keeps_terminal_group_and_tees_output(
    tmp_path, capsys, exit_code
):
    command = [
        sys.executable,
        "-c",
        "import os, sys; print('group=' + str(os.getpgrp()), flush=True); "
        f"sys.exit({exit_code})",
    ]
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError) as failure:
            launcher.run_foreground(command, tmp_path)
        assert failure.value.returncode == exit_code
    else:
        launcher.run_foreground(command, tmp_path)
    expected = f"group={os.getpgrp()}\n"
    assert (tmp_path / "train.log").read_text() == expected
    assert capsys.readouterr().out == expected


def test_shell_launcher_overrides_stale_exports_and_forwards_arguments(tmp_path):
    stub = tmp_path / "stub-python"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), "
        "'env': {k: os.environ[k] for k in "
        "['LBENCH_ROOT', 'LBENCH_TASK_CONFIG', 'LBENCH_ISAAC_PYTHON']}}))\n"
    )
    stub.chmod(0o755)
    child_env = os.environ.copy()
    child_env.update(
        RLINF_ROOT=str(ROOT),
        RLINF_PYTHON=str(stub),
        LBENCH_ROOT="/stale/lbench",
        LBENCH_TASK_CONFIG="/stale/task.yaml",
        LBENCH_ISAAC_PYTHON="/stale/python",
        LBENCH_PPO_ROOT="/selected/lbench",
        LBENCH_PPO_TASK_CONFIG="/selected/task.yaml",
        LBENCH_PPO_ISAAC_PYTHON="/selected/python",
    )
    argv = ["--dry-run", "--output", str(tmp_path / "output with spaces")]
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/lbench/remote/07_ppo_two_tasks_3gpu.sh"), *argv],
        env=child_env,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["argv"] == ["toolkits/lbench_two_task_ppo.py", *argv]
    assert payload["cwd"] == str(ROOT)
    assert payload["env"] == {
        "LBENCH_ROOT": "/selected/lbench",
        "LBENCH_TASK_CONFIG": "/selected/task.yaml",
        "LBENCH_ISAAC_PYTHON": "/selected/python",
    }
