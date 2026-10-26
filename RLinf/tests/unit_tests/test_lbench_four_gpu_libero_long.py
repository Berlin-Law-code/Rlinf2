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

"""CPU launch checks for the four-GPU drawer and clean configurations."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from toolkits import lbench_single_task_ppo as launcher

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = "collocated_4gpu_libero_long"


def make_resume_checkpoint(tmp_path):
    """Build a complete-marker fixture using a real composed source launch plan."""
    source = tmp_path / "source run"
    plan = launcher.build_plan(
        launcher.parse_args(
            [
                "--layout",
                LAYOUT,
                "--task",
                launcher.CLEAN_TASK,
                "--dry-run",
                "--output",
                str(source),
            ]
        )
    )
    checkpoint = source / "lbench_ppo_clean_4gpu_libero_long/checkpoints/global_step_42"
    for relative in (
        "actor/dcp_checkpoint/.metadata",
        "actor/dcp_checkpoint/__0_0.distcp",
        "actor/model_state_dict/full_weights.pt",
    ):
        item = checkpoint / relative
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_bytes(b"structural fixture only")
    (checkpoint / ".checkpoint_complete").write_text("42\n")
    (source / "launch_plan.json").write_text(json.dumps(plan))
    return checkpoint


def test_prediction_horizon_override_preserves_sampling_and_update_budget(
    isolated_machine, tmp_path
):
    common = ["--layout", LAYOUT, "--task", launcher.CLEAN_TASK, "--dry-run"]
    original = launcher.build_plan(
        launcher.parse_args(common + ["--output", str(tmp_path / "original")])
    )
    changed = launcher.build_plan(
        launcher.parse_args(
            common + ["--action-horizon", "20", "--output", str(tmp_path / "short")]
        )
    )
    cfg = changed["config"]
    assert cfg["actor"]["model"]["openpi"]["action_horizon"] == 20
    assert cfg["actor"]["model"]["openpi"]["action_chunk"] == 20
    assert cfg["actor"]["model"]["num_steps"] == 10
    assert cfg["runner"]["resume_dir"] is None
    for key in (
        "train_episodes_per_iteration",
        "optimizer_updates_per_iteration",
        "planned_optimizer_updates",
    ):
        assert changed[key] == original[key]
    assert cfg["algorithm"] == original["config"]["algorithm"]
    assert cfg["actor"]["optim"] == original["config"]["actor"]["optim"]
    assert launcher.compose_config(changed["command"][4:], changed["command"][3]) == cfg


def test_clean_dual_clip_and_critic_overrides_round_trip(isolated_machine, tmp_path):
    args = launcher.parse_args(
        [
            "--layout",
            LAYOUT,
            "--task",
            launcher.CLEAN_TASK,
            "--dry-run",
            "--action-horizon",
            "20",
            "--clip-ratio-c",
            "3",
            "--value-head-hidden-sizes",
            "1024",
            "512",
            "256",
            "--max-steps",
            "100",
            "--lr-schedule-steps",
            "1000",
            "--smoke-test-steps",
            "1",
            "--output",
            str(tmp_path / "trial"),
        ]
    )
    plan = launcher.build_plan(args)
    cfg = plan["config"]
    assert cfg["algorithm"]["clip_ratio_c"] == 3
    assert cfg["actor"]["model"]["openpi"]["value_head_hidden_sizes"] == [
        1024,
        512,
        256,
    ]
    assert cfg["actor"]["model"]["openpi"]["action_horizon"] == 20
    assert cfg["actor"]["model"]["num_action_chunks"] == 20
    assert cfg["actor"]["model"]["num_steps"] == 10
    assert cfg["actor"]["optim"]["lr"] == 1e-6
    assert cfg["actor"]["optim"]["total_training_steps"] == 1000
    assert cfg["runner"]["max_steps"] == 1
    assert cfg["runner"]["resume_dir"] is None
    assert plan["planned_optimizer_updates"] == 32
    assert launcher.compose_config(plan["command"][4:], plan["command"][3]) == cfg


@pytest.mark.parametrize(
    "option",
    [
        ["--clip-ratio-c", "1"],
        ["--clip-ratio-c", "nan"],
        ["--clip-ratio-c", "inf"],
        ["--value-head-hidden-sizes", "1024", "0"],
    ],
)
def test_invalid_critic_and_dual_clip_arguments(option):
    with pytest.raises(SystemExit):
        launcher.parse_args(option)


@pytest.mark.parametrize(
    "option,field",
    [
        (["--clip-ratio-c", "3"], "algorithm.clip_ratio_c"),
        (["--value-head-hidden-sizes", "1024", "512", "256"], "actor.model"),
    ],
)
def test_resume_preserves_critic_and_dual_clip(
    isolated_machine, tmp_path, option, field
):
    checkpoint = make_resume_checkpoint(tmp_path)
    args = launcher.parse_args(
        [
            "--layout",
            LAYOUT,
            "--task",
            launcher.CLEAN_TASK,
            "--dry-run",
            "--resume-dir",
            str(checkpoint),
            "--output",
            str(tmp_path / "changed"),
            *option,
        ]
    )
    with pytest.raises(ValueError, match=field):
        launcher.build_plan(args)


def test_prediction_horizon_cannot_be_shorter_than_execution(
    isolated_machine, tmp_path
):
    args = launcher.parse_args(
        [
            "--layout",
            LAYOUT,
            "--task",
            launcher.CLEAN_TASK,
            "--dry-run",
            "--action-horizon",
            "10",
            "--output",
            str(tmp_path / "short"),
        ]
    )
    with pytest.raises(ValueError, match="cover the executed"):
        launcher.build_plan(args)


def test_resume_rejects_changed_prediction_horizon(isolated_machine, tmp_path):
    checkpoint = make_resume_checkpoint(tmp_path)
    args = launcher.parse_args(
        [
            "--layout",
            LAYOUT,
            "--task",
            launcher.CLEAN_TASK,
            "--dry-run",
            "--resume-dir",
            str(checkpoint),
            "--action-horizon",
            "20",
            "--output",
            str(tmp_path / "changed_resume"),
        ]
    )
    with pytest.raises(ValueError, match="actor.model"):
        launcher.build_plan(args)


def test_resume_one_round_preserves_lr_and_restores_step(isolated_machine, tmp_path):
    checkpoint = make_resume_checkpoint(tmp_path)
    output = tmp_path / "recovery run"
    args = launcher.parse_args(
        [
            "--layout",
            LAYOUT,
            "--task",
            launcher.CLEAN_TASK,
            "--resume-dir",
            str(checkpoint),
            "--max-steps",
            "100",
            "--smoke-test-steps",
            "1",
            "--dry-run",
            "--output",
            str(output),
        ]
    )
    plan = launcher.build_plan(args)
    assert plan["starting_step"] == 42
    assert plan["remaining_iterations"] == 1
    assert plan["planned_train_episodes"] == 64
    assert plan["planned_optimizer_updates"] == 32
    assert plan["config"]["runner"]["max_steps"] == 43
    assert plan["learning_rate_schedule_steps"] == 1000
    assert plan["config"]["runner"]["resume_dir"] == str(checkpoint)
    assert plan["config"]["runner"]["eval_before_train"] is True
    assert (
        launcher.compose_config(plan["command"][4:], plan["command"][3])
        == plan["config"]
    )
    assert not output.exists()


def test_short_fresh_budget_can_preserve_long_lr_horizon(isolated_machine, tmp_path):
    plan = launcher.build_plan(
        launcher.parse_args(
            [
                "--layout",
                LAYOUT,
                "--task",
                launcher.CLEAN_TASK,
                "--max-steps",
                "100",
                "--lr-schedule-steps",
                "1000",
                "--dry-run",
                "--output",
                str(tmp_path / "fresh"),
            ]
        )
    )
    assert plan["config"]["runner"]["max_steps"] == 100
    assert plan["learning_rate_schedule_steps"] == 1000
    assert plan["starting_step"] == 0


@pytest.mark.parametrize("limit,extra", [(42, None), (43, 2)])
def test_resume_rejects_empty_or_excess_budget(
    isolated_machine, tmp_path, limit, extra
):
    checkpoint = make_resume_checkpoint(tmp_path)
    argv = [
        "--layout",
        LAYOUT,
        "--task",
        launcher.CLEAN_TASK,
        "--resume-dir",
        str(checkpoint),
        "--max-steps",
        str(limit),
        "--dry-run",
        "--output",
        str(tmp_path / "recovery"),
    ]
    if extra:
        argv.extend(["--smoke-test-steps", str(extra)])
    with pytest.raises(ValueError, match="additional rounds"):
        launcher.build_plan(launcher.parse_args(argv))


@pytest.mark.parametrize("problem", ["incomplete", "wrong_task", "changed_lr"])
def test_resume_rejects_incomplete_or_incompatible_source(
    isolated_machine, tmp_path, problem
):
    checkpoint = make_resume_checkpoint(tmp_path)
    argv = [
        "--layout",
        LAYOUT,
        "--task",
        launcher.CLEAN_TASK,
        "--resume-dir",
        str(checkpoint),
        "--max-steps",
        "100",
        "--dry-run",
        "--output",
        str(tmp_path / "recovery"),
    ]
    if problem == "incomplete":
        (checkpoint / ".checkpoint_complete").unlink()
        message = "completion marker"
    elif problem == "wrong_task":
        manifest = checkpoint.parents[2] / "launch_plan.json"
        source = json.loads(manifest.read_text())
        source["task"] = launcher.DEFAULT_TASK
        manifest.write_text(json.dumps(source))
        message = "task and GPU layout"
    else:
        argv.extend(["--lr-schedule-steps", "100"])
        message = "actor.optim"
    with pytest.raises(ValueError, match=message):
        launcher.build_plan(launcher.parse_args(argv))


@pytest.fixture
def isolated_machine(monkeypatch, tmp_path):
    # build_plan exports these for Hydra; restore them between tests so later
    # client tests cannot inherit a synthetic GPU placement or checkpoint.
    for key in (
        "LBENCH_SFT_CHECKPOINT",
        "LBENCH_NORM_STATS",
        "CUDA_VISIBLE_DEVICES",
        "RLINF_GPUS",
        "LBENCH_ISAAC_GPUS",
    ):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    for key, value in {
        "LBENCH_ROOT": ROOT.parent / "LBench_multi_env",
        "LBENCH_TASK_CONFIG": ROOT.parent
        / "LBench_multi_env/configs/simbox/lbench_joint_8tasks.yaml",
        "LBENCH_ISAAC_PYTHON": sys.executable,
        "EMBODIED_PATH": ROOT / "examples/embodiment",
    }.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(
        launcher,
        "checkpoint_contract",
        lambda path, **kwargs: {
            "checkpoint": str(path),
            "norm_stats": str(tmp_path / "norm_stats.json"),
            "manifest": {"action_horizon": 50},
        },
    )
    resource_calls = []

    def common_checks(scenario, models, environments, **kwargs):
        resource_calls.append((models, environments, kwargs))
        return {
            "task_names": [launcher.CLEAN_TASK]
            + [
                {1: launcher.PACK_TASK, 3: launcher.FLOWER_TASK}.get(
                    index, f"unused_{index}"
                )
                for index in range(1, 6)
            ]
            + [launcher.DEFAULT_TASK],
            "isaac_python": sys.executable,
        }

    monkeypatch.setattr(launcher, "_common_checks", common_checks)
    monkeypatch.setattr(launcher, "_isaac_runtime_env", lambda _: {})
    monkeypatch.setattr(launcher, "_visible_gpu_ids", lambda: {"0", "1", "2"})
    monkeypatch.setattr(
        launcher, "run_foreground", lambda *_: pytest.fail("CPU preflight launched")
    )
    monkeypatch.setattr(
        launcher, "_existing_ray_cluster", lambda: pytest.fail("CPU preflight used Ray")
    )
    return resource_calls


@pytest.mark.parametrize("micro", [1, 4, 32, 64])
def test_four_gpu_preflight_matches_actual_command_and_global_batch(
    isolated_machine, tmp_path, micro
):
    output = tmp_path / "new run"
    args = launcher.parse_args(
        [
            "--layout",
            LAYOUT,
            "--dry-run",
            "--gpus",
            "1,3,5,7",
            "--micro-batch-size",
            str(micro),
            "--smoke-test-steps",
            "5",
            "--output",
            str(output),
        ]
    )
    plan = launcher.build_plan(args)
    cfg = plan["config"]
    assert not output.exists()
    assert args.checkpoint == launcher.DEFAULT_CHECKPOINT
    assert cfg["runner"]["resume_dir"] is None
    assert cfg["runner"]["max_steps"] == cfg["runner"]["max_epochs"] == 5
    assert cfg["actor"]["optim"]["total_training_steps"] == 1000
    assert cfg["actor"]["optim"]["critic_warmup_steps"] == 0
    assert cfg["actor"]["optim"]["lr"] == 1e-6
    assert cfg["actor"]["global_batch_size"] == 256
    assert plan["gradient_accumulation"] == 64 // micro
    assert plan["train_episodes_per_iteration"] == 64
    assert plan["planned_train_episodes"] == 320
    assert plan["planned_chunk_samples_per_iteration"] == 3072
    assert plan["optimizer_updates_per_iteration"] == 48
    assert plan["planned_actor_updates_after_warmup_max"] == 240
    assert plan["terminal_padding_chunk_steps"] == 3
    assert plan["max_episode_steps"] == 900
    assert not plan["requested_gpus_available"]
    assert not plan["hardware_training_validated"]
    assert not plan["evaluation_enabled"]
    assert plan["resident_isaac_processes_by_gpu"] == dict.fromkeys(
        ["1", "3", "5", "7"], 4
    )
    assert cfg["cluster"]["component_placement"] == {
        "actor,env,rollout": "1:0,3:1,5:2,7:3"
    }
    assert plan["env_world_size"] == plan["actor_world_size"] == 4
    models, environments, resource_kwargs = isolated_machine[0]
    assert models == environments == ["1", "3", "5", "7"]
    assert resource_kwargs == {
        "allow_no_gpu": True,
        "shared_gpu_ids": ("1", "3", "5", "7"),
    }
    assert launcher.compose_config(plan["command"][4:], plan["command"][3]) == cfg


def test_four_gpu_defaults_keep_sft_assets_and_task_interface(
    isolated_machine, tmp_path
):
    args = launcher.parse_args(
        ["--layout", LAYOUT, "--dry-run", "--output", str(tmp_path / "new")]
    )
    assert args.gpus == "0,1,2,3"
    assert args.max_steps == 1000
    cfg = launcher.build_plan(args)["config"]
    model = cfg["actor"]["model"]
    assert model["action_dim"] == 8
    assert model["num_action_chunks"] == model["openpi"]["action_chunk"] == 20
    assert model["num_steps"] == model["openpi"]["num_steps"] == 10
    assert model["openpi"]["action_horizon"] == 50
    assert model["openpi"]["num_images_in_input"] == 3
    assert model["openpi"]["strict_ppo_precision"]
    assert cfg["actor"]["micro_batch_size"] == 32
    assert cfg["algorithm"]["ppo_update_guard"]["loss_weighting"] == "libero_trajectory"
    for key in ("clip_ratio_c", "clip_log_ratio_min", "clip_log_ratio_max"):
        assert cfg["algorithm"][key] is None
    assert cfg["env"]["eval"]["max_steps_per_rollout_epoch"] == 900
    train = cfg["env"]["train"]
    assert train["total_num_envs"] == 16
    assert train["startup_batch_size"] == 2
    assert train["startup_timeout_s"] == 1200
    assert train["backend_config"]["simbox_eval_config"]["cpu_thread_limit"] == 4
    assert train["rollout_epoch"] == 4
    assert train["max_episode_steps"] == 900
    assert train["backend_config"]["simbox_eval_config"]["max_step"] == 900
    assert train["max_steps_per_rollout_epoch"] == 960
    assert not train["auto_reset"]


def test_clean_four_gpu_plan_uses_task_horizon_and_original_sft(
    isolated_machine, tmp_path
):
    output = tmp_path / "clean run"
    args = launcher.parse_args(
        [
            "--layout",
            LAYOUT,
            "--task",
            launcher.CLEAN_TASK,
            "--smoke-test-steps",
            "1",
            "--dry-run",
            "--output",
            str(output),
        ]
    )
    plan = launcher.build_plan(args)
    cfg = plan["config"]
    assert not output.exists()
    assert plan["command"][3] == launcher.CLEAN_LIBERO_LONG_CONFIG_NAME
    assert launcher.compose_config(plan["command"][4:], plan["command"][3]) == cfg
    assert plan["task"] == launcher.CLEAN_TASK
    assert plan["task_index"] == 0
    assert args.checkpoint == launcher.DEFAULT_CHECKPOINT
    assert args.checkpoint.name == "global_step_30000"
    assert cfg["actor"]["model"]["model_path"] == str(args.checkpoint)
    assert cfg["rollout"]["model"]["model_path"] == str(args.checkpoint)
    assert cfg["runner"]["resume_dir"] is None
    assert cfg["runner"]["ckpt_path"] is None
    assert cfg["runner"]["max_steps"] == cfg["runner"]["max_epochs"] == 1
    assert cfg["actor"]["optim"]["total_training_steps"] == 1000
    assert (
        cfg["runner"]["logger"]["experiment_name"]
        == "lbench_ppo_clean_4gpu_libero_long"
    )
    assert cfg["actor"]["micro_batch_size"] == 32
    assert cfg["actor"]["global_batch_size"] == 256
    assert cfg["actor"]["optim"]["lr"] == 1e-6
    assert cfg["actor"]["enable_offload"] and cfg["rollout"]["enable_offload"]
    assert cfg["actor"]["model"]["num_action_chunks"] == 20
    assert cfg["actor"]["model"]["num_steps"] == 10
    guard = cfg["algorithm"]["ppo_update_guard"]
    assert guard["enabled"] is True
    assert guard["enforce_kl_limit"] is False
    assert guard["loss_weighting"] == "libero_trajectory"
    assert guard["max_initial_logratio"] == 0.01
    assert plan["ppo_update_guard"]["enforce_kl_limit"] is False
    train, evaluation = cfg["env"]["train"], cfg["env"]["eval"]
    for environment in (train, evaluation):
        assert environment["task_indices"] == [0]
        assert environment["max_episode_steps"] == 600
        assert environment["backend_config"]["simbox_eval_config"]["max_step"] == 600
        assert "clean" in environment["run_id"]
    assert train["max_steps_per_rollout_epoch"] == 640
    assert evaluation["max_steps_per_rollout_epoch"] == 600
    assert train["total_num_envs"] == 16
    assert train["rollout_epoch"] == 4
    assert plan["resident_isaac_processes_by_gpu"] == dict.fromkeys(
        ["0", "1", "2", "3"], 4
    )
    assert plan["gradient_accumulation"] == 2
    assert plan["train_episodes_per_iteration"] == 64
    assert plan["planned_chunk_samples_per_iteration"] == 2048
    assert plan["terminal_padding_chunk_steps"] == 2
    assert plan["optimizer_updates_per_iteration"] == 32
    assert plan["planned_actor_updates_after_warmup_max"] == 32
    assert plan["evaluation_enabled"]
    assert plan["evaluation_interval_rounds"] == 10
    assert plan["evaluate_before_training"]
    assert plan["eval_episodes_per_task"] == 32
    assert plan["evaluation_uses_serial_env_pools"]
    assert plan["resident_isaac_processes"] == 16
    assert plan["training_isaac_processes"] == 16
    assert plan["evaluation_isaac_processes"] == 16
    assert plan["evaluation_isaac_gpus"] == ["0", "1", "2", "3"]
    assert evaluation["fixed_seeds"] == list(range(32))
    assert evaluation["total_num_envs"] == 16
    assert evaluation["rollout_epoch"] == 2
    assert train["offload_before_eval"] and train["lazy_start"]
    assert not train["enable_offload"]
    assert evaluation["lazy_start"] and evaluation["enable_offload"]
    assert cfg["rollout"]["eval_initial_noise_seed"] == 42
    assert cfg["rollout"]["eval_repeat_initial_noise"]
    assert cfg["runner"]["save_interval"] == 1
    assert not plan["hardware_training_validated"]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("train", "offload_before_eval", False),
        ("train", "lazy_start", False),
        ("eval", "lazy_start", False),
        ("eval", "enable_offload", False),
        ("eval", "total_num_envs", 6),
        ("eval", "auto_reset", True),
    ],
)
def test_clean_eval_preflight_rejects_overlapping_or_uneven_pools(
    isolated_machine, tmp_path, monkeypatch, section, field, value
):
    compose = launcher.compose_config

    def changed_config(*args, **kwargs):
        cfg = compose(*args, **kwargs)
        cfg["env"][section][field] = value
        return cfg

    monkeypatch.setattr(launcher, "compose_config", changed_config)
    args = launcher.parse_args(
        [
            "--layout",
            LAYOUT,
            "--task",
            launcher.CLEAN_TASK,
            "--dry-run",
            "--output",
            str(tmp_path / "new"),
        ]
    )
    with pytest.raises(ValueError, match="Multi-GPU automatic evaluation requires"):
        launcher.build_plan(args)


def test_selecting_clean_does_not_change_drawer_profile(isolated_machine, tmp_path):
    def plan_for(task, directory):
        return launcher.build_plan(
            launcher.parse_args(
                [
                    "--layout",
                    LAYOUT,
                    "--task",
                    task,
                    "--dry-run",
                    "--output",
                    str(tmp_path / directory),
                ]
            )
        )

    before = plan_for(launcher.DEFAULT_TASK, "drawer")
    plan_for(launcher.CLEAN_TASK, "clean")
    after = plan_for(launcher.DEFAULT_TASK, "drawer")
    assert before == after
    cfg = after["config"]
    assert after["command"][3] == launcher.LIBERO_LONG_CONFIG_NAME
    assert after["task_index"] == 6
    assert cfg["env"]["train"]["max_episode_steps"] == 900
    assert cfg["env"]["train"]["max_steps_per_rollout_epoch"] == 960
    assert after["ppo_update_guard"].get("enforce_kl_limit", True) is True
    assert after["optimizer_updates_per_iteration"] == 48


@pytest.mark.parametrize("micro", [3, 128])
def test_four_gpu_unaligned_microbatch_rejected_before_launch(
    isolated_machine, tmp_path, micro
):
    args = launcher.parse_args(
        [
            "--layout",
            LAYOUT,
            "--dry-run",
            "--micro-batch-size",
            str(micro),
            "--output",
            str(tmp_path / "new"),
        ]
    )
    with pytest.raises(ValueError, match="divide"):
        launcher.build_plan(args)
    assert not (tmp_path / "new").exists()


def test_three_gpu_probe_keeps_2048_with_two_actor_ranks(isolated_machine, tmp_path):
    args = launcher.parse_args(
        [
            "--layout",
            "libero_long_3gpu_probe",
            "--dry-run",
            "--micro-batch-size",
            "2",
            "--output",
            str(tmp_path / "new"),
        ]
    )
    assert args.gpus == "0,1,2"
    assert args.max_steps == 1000
    assert args.smoke_test_steps == 1
    plan = launcher.build_plan(args)
    cfg = plan["config"]
    assert cfg["runner"]["max_steps"] == 1
    assert cfg["actor"]["optim"]["total_training_steps"] == 1000
    assert cfg["cluster"]["component_placement"] == {
        "actor": "0:0,1:1",
        "env,rollout": "0:0,1:1,2:2",
    }
    assert plan["actor_gpus"] == ["0", "1"]
    assert plan["rollout_gpus"] == plan["training_isaac_gpus"] == ["0", "1", "2"]
    assert plan["resident_isaac_processes_by_gpu"] == dict.fromkeys(["0", "1", "2"], 8)
    assert plan["train_episodes_per_iteration"] == 192
    assert plan["planned_chunk_samples_per_iteration"] == 18432
    assert plan["global_batch_size"] == 2048
    assert plan["gradient_accumulation"] == 512
    assert plan["optimizer_updates_per_iteration"] == 36
    assert launcher.compose_config(plan["command"][4:], plan["command"][3]) == cfg


@pytest.mark.parametrize("prestart", ["0", "1"])
def test_four_gpu_shell_passes_layout_and_thread_limits(
    tmp_path, prestart, monkeypatch
):
    monkeypatch.delenv("LBENCH_PPO_RAY_PRESTART_WORKERS", raising=False)
    stub = tmp_path / "stub-python"
    stub.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'threads': os.environ['OMP_NUM_THREADS'], "
        "'prestart': os.environ['RAY_prestart_worker_first_driver'], "
        "'enable_prestart': os.environ['RAY_enable_worker_prestart']}))\n"
    )
    stub.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/lbench/remote/11_ppo_drawer_4gpu_libero_long.sh"),
            "--dry-run",
        ],
        env={
            **os.environ,
            "RLINF_PYTHON": str(stub),
            "LBENCH_PPO_CPU_THREADS": "1",
            **({"LBENCH_PPO_RAY_PRESTART_WORKERS": "1"} if prestart == "1" else {}),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["argv"] == [
        "toolkits/lbench_single_task_ppo.py",
        "--dry-run",
        "--layout",
        LAYOUT,
    ]
    assert payload["threads"] == "1"
    assert payload["prestart"] == payload["enable_prestart"] == prestart


@pytest.mark.parametrize("task_override", [None, launcher.DEFAULT_TASK])
def test_clean_shell_keeps_explicit_arguments(tmp_path, task_override):
    stub = tmp_path / "stub-python"
    stub.write_text(
        f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    stub.chmod(0o755)
    arguments = ["--dry-run", "--max-steps", "75", "--micro-batch-size", "16"]
    if task_override is not None:
        arguments += ["--task", task_override]
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/lbench/remote/13_ppo_clean_4gpu_libero_long.sh"),
            *arguments,
        ],
        env={**os.environ, "RLINF_PYTHON": str(stub)},
        capture_output=True,
        text=True,
        check=True,
    )
    forwarded = json.loads(result.stdout)
    assert forwarded == [
        "toolkits/lbench_single_task_ppo.py",
        "--task",
        launcher.CLEAN_TASK,
        *arguments,
        "--layout",
        LAYOUT,
    ]
    args = launcher.parse_args(forwarded[1:])
    assert args.task == (task_override or launcher.CLEAN_TASK)
    assert args.layout == LAYOUT
    assert args.max_steps == 75
    assert args.micro_batch_size == 16


def test_clean_shell_help_identifies_its_default_task():
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/lbench/remote/13_ppo_clean_4gpu_libero_long.sh"),
            "--help",
        ],
        env={**os.environ, "RLINF_PYTHON": sys.executable},
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"Default task: {launcher.CLEAN_TASK}" in result.stdout
    assert "600 real episode steps" in result.stdout
    assert "--smoke-test-steps" in result.stdout


def test_diagnostic_launch_preserves_training_and_records_both_outcomes(
    isolated_machine, tmp_path
):
    common = [
        "--layout",
        LAYOUT,
        "--task",
        launcher.CLEAN_TASK,
        "--dry-run",
        "--output",
        str(tmp_path / "recording"),
    ]
    original = launcher.build_plan(launcher.parse_args(common))
    recorded = launcher.build_plan(
        launcher.parse_args(common + ["--record-diagnostics"])
    )
    assert recorded["config"]["algorithm"] == original["config"]["algorithm"]
    assert recorded["config"]["actor"]["optim"] == original["config"]["actor"]["optim"]
    for mode in ("train", "eval"):
        cfg = recorded["config"]["env"][mode]["data_collection"]
        assert cfg["enabled"] and not cfg["only_success"]
        assert cfg["export_format"] == "diagnostic"
        assert cfg["max_pending_exports"] == 1
    assert (
        launcher.compose_config(recorded["command"][4:], recorded["command"][3])
        == recorded["config"]
    )


@pytest.mark.parametrize(
    "task,task_index,cap,window,updates,config_name",
    [
        (launcher.FLOWER_TASK, 3, 600, 640, 32, launcher.FLOWER_FOUR_CONFIG_NAME),
        (launcher.PACK_TASK, 1, 960, 960, 48, launcher.PACK_FOUR_CONFIG_NAME),
    ],
)
def test_four_gpu_task_validation_uses_matching_budget_and_final_eval(
    isolated_machine, tmp_path, task, task_index, cap, window, updates, config_name
):
    from rlinf.utils.runner_utils import check_progress

    plan = launcher.build_plan(
        launcher.parse_args(
            [
                "--layout",
                LAYOUT,
                "--task",
                task,
                "--smoke-test-steps",
                "1",
                "--record-diagnostics",
                "--dry-run",
                "--output",
                str(tmp_path / "validation4"),
            ]
        )
    )
    cfg = plan["config"]
    assert (
        plan["actor_world_size"]
        == plan["rollout_world_size"]
        == plan["env_world_size"]
        == 4
    )
    assert plan["resident_isaac_processes_by_gpu"] == {str(i): 4 for i in range(4)}
    assert plan["planned_train_episodes"] == 64
    assert plan["planned_optimizer_updates"] == updates
    assert plan["command"][3] == config_name
    assert plan["config"]["env"]["train"]["max_steps_per_rollout_epoch"] == window
    assert plan["gradient_accumulation"] == 2
    assert plan["eval_episodes_per_task"] == 32
    assert cfg["env"]["eval"]["fixed_seeds"] == list(range(32))
    assert plan["learning_rate_schedule_steps"] == 1000
    assert cfg["runner"]["max_steps"] == 1
    assert cfg["runner"]["resume_dir"] is None
    assert cfg["runner"]["eval_before_train"]
    assert check_progress(1, 1, cfg["runner"]["val_check_interval"], 1, 1.0) == (
        True,
        True,
        True,
    )
    for mode in ("train", "eval"):
        assert cfg["env"][mode]["task_indices"] == [task_index]
        assert cfg["env"][mode]["max_episode_steps"] == cap
        assert cfg["env"][mode]["data_collection"]["enabled"]
    model = cfg["actor"]["model"]
    assert model["openpi"]["action_horizon"] == model["num_action_chunks"] == 20
    assert model["num_steps"] == model["openpi"]["num_steps"] == 5
    assert model["openpi"]["noise_level"] == 0.25
    assert model["openpi"]["ignore_last"] is True
    assert model["openpi"]["value_head_hidden_sizes"] == [1024, 512, 256]
    assert cfg["algorithm"]["clip_ratio_c"] == 3
    assert launcher.compose_config(plan["command"][4:], plan["command"][3]) == cfg


def test_single_gpu_flower_trial_has_bounded_budget_and_serial_pools(
    isolated_machine, tmp_path
):
    args = launcher.parse_args(
        [
            "--layout",
            "collocated_1gpu",
            "--task",
            launcher.FLOWER_TASK,
            "--record-diagnostics",
            "--dry-run",
            "--output",
            str(tmp_path / "flower"),
        ]
    )
    plan = launcher.build_plan(args)
    cfg = plan["config"]
    assert (
        plan["actor_gpus"]
        == plan["rollout_gpus"]
        == plan["training_isaac_gpus"]
        == ["0"]
    )
    assert plan["actor_world_size"] == plan["env_world_size"] == 1
    assert plan["resident_isaac_processes_by_gpu"] == {"0": 4}
    assert plan["evaluation_uses_serial_env_pools"]
    assert isolated_machine[0][2]["shared_gpu_ids"] == ("0",)
    assert plan["remaining_iterations"] == 1
    assert plan["planned_train_episodes"] == 16
    assert plan["planned_optimizer_updates"] == 8
    assert plan["gradient_accumulation"] == 8
    assert plan["eval_episodes_per_task"] == 8
    assert plan["learning_rate_schedule_steps"] == 1000
    assert cfg["runner"]["resume_dir"] is None
    assert (
        cfg["env"]["train"]["task_indices"] == cfg["env"]["eval"]["task_indices"] == [3]
    )
    assert cfg["env"]["train"]["max_episode_steps"] == 600
    assert cfg["env"]["train"]["offload_before_eval"]
    assert cfg["env"]["eval"]["fixed_seeds"] == list(range(8))
    assert cfg["actor"]["model"]["openpi"]["action_horizon"] == 20
    assert cfg["actor"]["model"]["num_action_chunks"] == 20
    assert cfg["actor"]["model"]["num_steps"] == 10
    assert cfg["actor"]["model"]["openpi"]["value_head_hidden_sizes"] == [
        1024,
        512,
        256,
    ]
    assert cfg["algorithm"]["clip_ratio_c"] == 3
    assert cfg["actor"]["optim"]["lr"] == 1e-6
    assert cfg["actor"]["enable_offload"] and cfg["rollout"]["enable_offload"]
    assert cfg["env"]["train"]["data_collection"]["enabled"]
    assert launcher.compose_config(plan["command"][4:], plan["command"][3]) == cfg


@pytest.mark.parametrize("extra", [["--gpus", "0,1"], ["--task", launcher.CLEAN_TASK]])
def test_single_gpu_profile_rejects_wrong_devices_or_task(
    isolated_machine, tmp_path, extra
):
    args = launcher.parse_args(
        [
            "--layout",
            "collocated_1gpu",
            "--task",
            launcher.FLOWER_TASK,
            "--dry-run",
            "--output",
            str(tmp_path / "invalid"),
            *extra,
        ]
    )
    with pytest.raises(ValueError):
        launcher.build_plan(args)


def test_denoising_override_changes_only_model_step_counts(isolated_machine, tmp_path):
    import copy

    common = [
        "--layout",
        "collocated_1gpu",
        "--task",
        launcher.FLOWER_TASK,
        "--dry-run",
        "--output",
        str(tmp_path / "denoising"),
    ]
    original = launcher.build_plan(launcher.parse_args(common))
    changed = launcher.build_plan(
        launcher.parse_args(common + ["--denoise-steps", "5"])
    )
    expected = copy.deepcopy(original["config"])
    expected["actor"]["model"]["num_steps"] = 5
    expected["actor"]["model"]["openpi"]["num_steps"] = 5
    assert changed["config"] == expected
    assert changed["planned_train_episodes"] == 16
    assert changed["planned_optimizer_updates"] == 8
    assert (
        launcher.compose_config(changed["command"][4:], changed["command"][3])
        == expected
    )


@pytest.mark.parametrize("steps", ["0", "1", "-1"])
def test_invalid_denoising_steps_are_rejected(steps):
    with pytest.raises(SystemExit):
        launcher.parse_args(["--denoise-steps", steps])


def test_noise_override_preserves_sft_validation_config(isolated_machine, tmp_path):
    import copy

    common = [
        "--layout",
        "collocated_1gpu",
        "--task",
        launcher.FLOWER_TASK,
        "--denoise-steps",
        "5",
        "--ignore-last",
        "--dry-run",
        "--output",
        str(tmp_path / "noise_trial"),
    ]
    original = launcher.build_plan(launcher.parse_args(common))
    changed = launcher.build_plan(
        launcher.parse_args(common + ["--noise-level", "0.25"])
    )
    expected = copy.deepcopy(original["config"])
    expected["actor"]["model"]["openpi"]["noise_level"] = 0.25
    assert changed["config"] == expected
    assert changed["planned_train_episodes"] == 16
    assert changed["planned_optimizer_updates"] == 8
    assert (
        launcher.compose_config(changed["command"][4:], changed["command"][3])
        == expected
    )


@pytest.mark.parametrize("level", ["0", "-0.25", "nan", "inf"])
def test_invalid_ppo_noise_level_is_rejected(level):
    with pytest.raises(SystemExit):
        launcher.parse_args(["--noise-level", level])


def test_resume_rejects_changed_noise_level(isolated_machine, tmp_path):
    checkpoint = make_resume_checkpoint(tmp_path)
    args = launcher.parse_args(
        [
            "--layout",
            LAYOUT,
            "--task",
            launcher.CLEAN_TASK,
            "--dry-run",
            "--resume-dir",
            str(checkpoint),
            "--noise-level",
            "0.25",
            "--output",
            str(tmp_path / "changed_noise"),
        ]
    )
    with pytest.raises(ValueError, match="actor.model"):
        launcher.build_plan(args)


def test_resume_rejects_changed_denoising_steps(isolated_machine, tmp_path):
    checkpoint = make_resume_checkpoint(tmp_path)
    args = launcher.parse_args(
        [
            "--layout",
            LAYOUT,
            "--task",
            launcher.CLEAN_TASK,
            "--dry-run",
            "--resume-dir",
            str(checkpoint),
            "--denoise-steps",
            "5",
            "--output",
            str(tmp_path / "changed_denoising"),
        ]
    )
    with pytest.raises(ValueError, match="actor.model"):
        launcher.build_plan(args)


@pytest.mark.parametrize(
    "option,expected", [("--ignore-last", True), ("--no-ignore-last", False)]
)
def test_explicit_ignore_last_resume_preserves_other_settings(
    isolated_machine, tmp_path, option, expected
):
    import copy

    checkpoint = make_resume_checkpoint(tmp_path)
    source_file = checkpoint.parents[2] / "launch_plan.json"
    source_text = source_file.read_text()
    common = [
        "--layout",
        LAYOUT,
        "--task",
        launcher.CLEAN_TASK,
        "--dry-run",
        "--resume-dir",
        str(checkpoint),
        "--smoke-test-steps",
        "1",
        "--output",
        str(tmp_path / "resumed"),
    ]
    original = launcher.build_plan(launcher.parse_args(common))
    changed = launcher.build_plan(launcher.parse_args(common + [option]))
    cfg = copy.deepcopy(original["config"])
    cfg["actor"]["model"]["openpi"]["ignore_last"] = expected
    assert changed["config"] == cfg
    assert changed["ignore_last_override"] is expected
    assert changed["starting_step"] == 42
    assert changed["planned_optimizer_updates"] == original["planned_optimizer_updates"]
    assert source_file.read_text() == source_text
    assert launcher.compose_config(changed["command"][4:], changed["command"][3]) == cfg


def test_ignore_last_override_does_not_allow_other_model_changes(
    isolated_machine, tmp_path
):
    checkpoint = make_resume_checkpoint(tmp_path)
    args = launcher.parse_args(
        [
            "--layout",
            LAYOUT,
            "--task",
            launcher.CLEAN_TASK,
            "--dry-run",
            "--resume-dir",
            str(checkpoint),
            "--ignore-last",
            "--denoise-steps",
            "5",
            "--output",
            str(tmp_path / "invalid"),
        ]
    )
    with pytest.raises(ValueError, match="actor.model"):
        launcher.build_plan(args)


def test_ignore_last_resume_change_requires_explicit_override(
    isolated_machine, tmp_path
):
    import copy

    checkpoint = make_resume_checkpoint(tmp_path)
    source = json.loads((checkpoint.parents[2] / "launch_plan.json").read_text())[
        "config"
    ]
    changed = copy.deepcopy(source)
    changed["actor"]["model"]["openpi"]["ignore_last"] = True
    with pytest.raises(ValueError, match="actor.model"):
        launcher.validate_resume_config(source, changed)
    with pytest.raises(ValueError, match="requested flag"):
        launcher.validate_resume_config(source, changed, ignore_last_override=False)
