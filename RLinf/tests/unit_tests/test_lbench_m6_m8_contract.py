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

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import toolkits.lbench_remote_stage as remote_stage
from toolkits.lbench_remote_stage import (
    _commands,
    _compose_and_validate,
    _hydra_overrides,
)

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "lbench_data_dry_run", ROOT / "toolkits/lbench_data_dry_run.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_DATA = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DATA)


@pytest.fixture(autouse=True)
def lbench_env(monkeypatch, tmp_path):
    (tmp_path / "meta").mkdir()
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [8],
            "names": ["state"],
        },
        "observation.images.obs_camera": {
            "dtype": "image",
            "shape": [3, 8, 8],
            "names": ["channel", "height", "width"],
        },
        "observation.images.obs_camera_2": {
            "dtype": "image",
            "shape": [3, 8, 8],
            "names": ["channel", "height", "width"],
        },
        "observation.images.realsense": {
            "dtype": "image",
            "shape": [3, 8, 8],
            "names": ["channel", "height", "width"],
        },
        "action": {"dtype": "float32", "shape": [8], "names": ["action"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    (tmp_path / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 30,
                "total_episodes": 8,
                "total_frames": 400,
                "total_tasks": 8,
                "total_chunks": 1,
                "chunks_size": 1000,
                "data_path": (
                    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
                ),
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    task_names = [f"mock task {index}" for index in range(8)]
    (tmp_path / "meta/tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": index, "task": task}) + "\n"
            for index, task in enumerate(task_names)
        ),
        encoding="utf-8",
    )
    (tmp_path / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "length": 50, "tasks": [task]}) + "\n"
            for index, task in enumerate(task_names)
        ),
        encoding="utf-8",
    )
    (tmp_path / "meta/episodes_stats.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "stats": {}}) + "\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    for index in range(8):
        (tmp_path / f"data/chunk-000/episode_{index:06d}.parquet").touch()
    (tmp_path / "base").mkdir()
    (tmp_path / "base/model.safetensors").touch()
    (tmp_path / "sft").mkdir()
    (tmp_path / "sft/model.safetensors").touch()
    norm = {
        "state": {
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
            "q01": [-1.0] * 7 + [0.0],
            "q99": [1.0] * 8,
        },
        "actions": {
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
            "q01": [-1.0] * 7 + [0.0],
            "q99": [1.0] * 8,
        },
    }
    (tmp_path / "norm_stats.json").write_text(json.dumps(norm), encoding="utf-8")
    (tmp_path / "task.yaml").write_text(
        "tasks:\n"
        "  - name: mock_task\n"
        "    cameras:\n"
        "      - {name: obs_camera}\n"
        "      - {name: obs_camera_2}\n"
        "      - {name: realsense}\n"
        "    data:\n"
        "      language_instruction: do the mock task\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    monkeypatch.setenv("LBENCH_LEROBOT_REPO", str(tmp_path))
    monkeypatch.setenv("LBENCH_NORM_STATS", str(tmp_path / "norm_stats.json"))
    monkeypatch.setenv("PI05_BASE_CHECKPOINT", str(tmp_path / "base"))
    monkeypatch.setenv("LBENCH_SFT_CHECKPOINT", str(tmp_path / "sft"))
    monkeypatch.setenv("LBENCH_ROOT", "/home/berlin/flywheel/LBench_multi_env")
    monkeypatch.setenv("LBENCH_ISAAC_PYTHON", "/bin/python3")
    monkeypatch.setenv("LBENCH_TASK_CONFIG", str(tmp_path / "task.yaml"))


def test_m6_sft_eval_and_ppo_configs_share_shape_contract(tmp_path):
    def compose(scenario, env_count):
        overrides = _hydra_overrides(scenario, ["0"], ["1"], tmp_path)
        return _compose_and_validate(scenario, env_count, 1, overrides)

    sft = compose("sft-smoke", 1)
    evaluate = compose("single-eval", 1)
    ppo = compose("dual-ppo", 2)
    assert (
        sft["action_horizon"]
        == evaluate["action_horizon"]
        == ppo["action_horizon"]
        == 50
    )
    assert sft["execute_horizon"] == 50
    assert evaluate["execute_horizon"] == ppo["execute_horizon"] == 20
    assert sft["train_expert_only"] is True
    assert ppo["train_expert_only"] is True
    assert ppo["episode_horizon"] == ppo["rollout_horizon"] == 40
    assert evaluate["chunk_steps_per_rollout_epoch"] == 60
    assert ppo["chunk_steps_per_rollout_epoch"] == 2
    assert ppo["max_primitive_env_steps_per_global_step"] == 80
    assert ppo["max_primitive_env_steps_total"] == 160
    assert ppo["reward_adapter"] == "libero_sparse_success"


def test_norm_stats_contract_rejects_non_finite_and_wrong_dim(tmp_path):
    valid = {
        "state": {
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
            "q01": [-1.0] * 7 + [0.0],
            "q99": [1.0] * 8,
        },
        "actions": {
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
            "q01": [-1.0] * 7 + [0.0],
            "q99": [1.0] * 8,
        },
    }
    path = tmp_path / "norm.json"
    path.write_text(json.dumps(valid), encoding="utf-8")
    assert _DATA.validate_norm_stats(path) == {"state": [8], "actions": [8]}
    valid["actions"]["q01"] = [0.0] * 7
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ValueError, match="8-D"):
        _DATA.validate_norm_stats(path)


def test_norm_stats_contract_accepts_openpi_serialization(tmp_path):
    stats = {
        key: {
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
            "q01": [0.0] * 8,
            "q99": [1.0] * 8,
        }
        for key in ("state", "actions")
    }
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": stats}), encoding="utf-8")
    assert _DATA.validate_norm_stats(path) == {"state": [8], "actions": [8]}


def test_lbench_metadata_accepts_current_v21_layout(tmp_path):
    info = _DATA._validate_metadata(tmp_path)
    assert info["codebase_version"] == "v2.1"
    assert _DATA._v21_task_episode_counts(tmp_path) == {
        f"mock task {index}": 1 for index in range(8)
    }


def test_lbench_runtime_rejects_storage_format_mismatch(monkeypatch):
    monkeypatch.setattr(
        _DATA,
        "_lerobot_runtime",
        lambda: {
            "package_version": "0.3.3",
            "codebase_version": "v2.1",
            "storage_format": "v2.1",
        },
    )
    assert (
        _DATA.validate_lerobot_runtime({"codebase_version": "v2.1"})["package_version"]
        == "0.3.3"
    )
    with pytest.raises(ValueError, match="dataset/runtime mismatch"):
        _DATA.validate_lerobot_runtime({"codebase_version": "v3.0"})


def test_m8_plans_cover_required_acceptance_sequence(tmp_path):
    expected = {
        "sft-smoke": ("runner.max_steps=5", 2),
        "single-eval": ("env.eval.total_num_envs=1", 1),
        "dual-ppo": ("runner.max_steps=2", 1),
        "eval-8": ("env.eval.total_num_envs=8", 1),
        "ppo-8": ("env.train.total_num_envs=8", 1),
    }
    for scenario, (needle, command_count) in expected.items():
        commands = _commands(scenario, ["0"], ["1"], tmp_path)
        assert len(commands) == command_count
        assert any(needle in argument for command in commands for argument in command)
    sft_commands = _commands("sft-smoke", ["0"], ["1"], tmp_path)
    assert "actor.model.openpi.train_expert_only=true" in sft_commands[0]
    assert any("runner.resume_dir=" in value for value in sft_commands[1])
    assert "runner.max_steps=6" in sft_commands[1]
    assert "runner.max_steps=200" in _commands("ppo-8", ["0"], ["1"], tmp_path)[0]
    dual_ppo_command = _commands("dual-ppo", ["0"], ["1"], tmp_path)[0]
    assert "env.train.max_episode_steps=40" in dual_ppo_command
    assert "env.train.max_steps_per_rollout_epoch=40" in dual_ppo_command


def test_eval_composition_loads_checked_checkpoint_and_collects_data(tmp_path):
    overrides = _hydra_overrides("single-eval", ["0"], ["1"], tmp_path)
    result = _compose_and_validate("single-eval", 1, 1, overrides)
    assert result["component_placement"] == {"env,rollout": 0}
    assert result["execute_horizon"] == 20
    assert "env.eval.data_collection.enabled=true" in overrides
    assert result["train_expert_only"] is True


def test_remote_stage_pins_component_placement_to_requested_gpus(tmp_path):
    single = _hydra_overrides("single-eval", ["0"], ["1"], tmp_path)
    assert r"cluster.component_placement={env\,rollout:0}" in single

    multi = _hydra_overrides("eval-8", ["0", "2"], ["1"], tmp_path)
    assert r"cluster.component_placement={env\,rollout:0\,2}" in multi

    ppo = _hydra_overrides("dual-ppo", ["0"], ["1"], tmp_path)
    assert r"cluster.component_placement={actor\,env\,rollout:0}" in ppo


def test_remote_stage_derives_isaac_runtime_from_selected_python(tmp_path, monkeypatch):
    prefix = tmp_path / "isaac-env"
    python = prefix / "bin/python"
    python.parent.mkdir(parents=True)
    python.touch(mode=0o755)
    isaac_path = prefix / "lib/python3.11/site-packages/isaacsim"
    (isaac_path / "apps").mkdir(parents=True)
    (isaac_path / "kit").mkdir()
    libgomp = prefix / "lib/libgomp.so.1"
    libgomp.touch()

    # Invalid parent-process placeholders must not shadow the selected Isaac
    # interpreter's colocated runtime.
    monkeypatch.setenv("ISAAC_PATH", "/path/to/isaac-sim")
    monkeypatch.setenv("EXP_PATH", "/path/to/isaac-sim/apps")
    monkeypatch.setenv("CARB_APP_PATH", "/path/to/isaac-sim/kit")

    assert remote_stage._isaac_runtime_env(python) == {
        "ISAAC_PATH": str(isaac_path),
        "EXP_PATH": str(isaac_path / "apps"),
        "CARB_APP_PATH": str(isaac_path / "kit"),
        "LBENCH_ISAAC_LD_PRELOAD": str(libgomp),
    }


def test_eight_env_eval_can_run_balanced_seed_rounds(tmp_path):
    overrides = _hydra_overrides(
        "eval-8", ["0"], ["1"], tmp_path, eval_rollout_epochs=8
    )
    result = _compose_and_validate("eval-8", 8, 1, overrides)
    assert result["rollout_epoch"] == 8
    assert result["planned_eval_episode_attempts"] == 64
    assert "env.eval.rollout_epoch=8" in overrides


def test_eval_can_emit_the_lbench_joint_feature_schema(tmp_path):
    overrides = _hydra_overrides(
        "single-eval", ["0"], ["1"], tmp_path, collection_format="lerobot"
    )
    result = _compose_and_validate("single-eval", 1, 1, overrides)
    assert result["execute_horizon"] == 20
    assert "env.eval.data_collection.export_format=lerobot" in overrides
    assert "env.eval.data_collection.lerobot_schema=lbench_joint" in overrides


def test_m8_rejects_env_count_not_divisible_by_world_size(tmp_path):
    overrides = _hydra_overrides("dual-ppo", ["0", "1", "2"], ["3"], tmp_path)
    with pytest.raises(ValueError, match="not divisible"):
        _compose_and_validate("dual-ppo", 2, 3, overrides)


def test_m8_baseline_rejects_non_libero_reward_adapter(tmp_path):
    overrides = _hydra_overrides("dual-ppo", ["0"], ["1"], tmp_path)
    overrides.append("env.train.reward_adapter.type=progress_delta")
    with pytest.raises(ValueError, match="libero_sparse_success"):
        _compose_and_validate("dual-ppo", 2, 1, overrides)


def test_m8_resource_check_rejects_gpu_overlap(monkeypatch):
    monkeypatch.setattr(remote_stage, "_visible_gpu_ids", lambda: {"0", "1"})
    resources = remote_stage._common_checks("sft-smoke", ["0"], ["1"], False)
    assert (Path(resources["dataset"]) / "meta/info.json").is_file()
    with pytest.raises(ValueError, match="overlap"):
        remote_stage._common_checks("dual-ppo", ["0"], ["0"], False)


def test_explicit_shared_gpu_allowance_does_not_permit_other_overlap(monkeypatch):
    monkeypatch.setattr(remote_stage, "_visible_gpu_ids", lambda: {"0", "1", "2"})
    with pytest.raises(ValueError, match="overlap"):
        remote_stage._common_checks(
            "dual-ppo", ["0", "1"], ["0", "1"], False, shared_gpu_ids=("0",)
        )
    with pytest.raises(ValueError, match="both GPU sets"):
        remote_stage._common_checks(
            "dual-ppo", ["0"], ["1"], False, shared_gpu_ids=("2",)
        )
