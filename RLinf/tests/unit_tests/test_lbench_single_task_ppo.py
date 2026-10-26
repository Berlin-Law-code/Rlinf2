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

"""CPU checks for the two-GPU drawer PPO launch and episode allocation."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.distributed.fsdp import MixedPrecision

from rlinf.config import torch_dtype_from_precision
from rlinf.envs.lbench.lbench_env import LBenchEnv
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress

ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = "lbench_ppo_openpi_pi05_drawer_2gpu"
TASK = "store_objects_in_the_drawer_0"
TASK_INDEX = 6
LBENCH_ROOT = Path(
    os.environ.get("LBENCH_ROOT") or ROOT.parent / "LBench_multi_env"
).expanduser()


@pytest.fixture
def profile(monkeypatch, tmp_path):
    for key, value in {
        "EMBODIED_PATH": ROOT / "examples/embodiment",
        "LBENCH_ROOT": LBENCH_ROOT,
        "LBENCH_TASK_CONFIG": LBENCH_ROOT / "configs/simbox/lbench_joint_8tasks.yaml",
        "LBENCH_ISAAC_PYTHON": sys.executable,
        "LBENCH_SFT_CHECKPOINT": tmp_path / "sft",
        "LBENCH_NORM_STATS": tmp_path / "norm_stats.json",
    }.items():
        monkeypatch.setenv(key, str(value))
    with initialize_config_dir(
        config_dir=str(ROOT / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name=CONFIG_NAME)
    OmegaConf.resolve(cfg)
    return cfg


def test_single_task_profile_has_explicit_training_gpu_and_disabled_eval(profile):
    assert OmegaConf.to_container(profile.cluster.component_placement) == {
        "actor,env,rollout": 0
    }
    train, evaluation = profile.env.train, profile.env.eval
    assert list(train.isaac_cuda_visible_devices) == [1]
    assert list(evaluation.isaac_cuda_visible_devices) == [0]
    assert train.total_num_envs == 8
    assert evaluation.total_num_envs == 2
    assert train.rollout_epoch == 4
    assert evaluation.rollout_epoch == 5
    assert profile.rollout.pipeline_stage_num == 1
    assert profile.actor.enable_offload is False
    assert profile.rollout.enable_offload is False
    for env_cfg in (train, evaluation):
        assert list(env_cfg.task_indices) == [TASK_INDEX]
        assert env_cfg.max_episode_steps == env_cfg.max_steps_per_rollout_epoch == 1200
        assert env_cfg.auto_reset is False
        assert env_cfg.ignore_terminations is False
        assert env_cfg.data_collection.enabled is False
        assert env_cfg.video_cfg.save_video is False
        assert env_cfg.get("enable_offload", False) is False
    # Retain independent eval settings without starting the eval worker pool.
    assert profile.runner.val_check_interval == -1
    assert profile.runner.only_eval is False
    assert profile.runner.save_interval == 5
    assert profile.runner.max_saved_checkpoints == 3
    assert profile.runner.max_consecutive_zero_success == 3


def test_single_task_profile_preserves_checkpoint_contract_and_batch_size(profile):
    model = profile.actor.model
    assert model.precision is None
    assert model.openpi.parameter_precision == "float32"
    assert model.openpi.strict_ppo_precision is True
    assert model.openpi_data.lerobot_compatible is True
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
    assert model.openpi.action_horizon == 50
    assert model.action_dim == model.openpi.action_env_dim == 8
    assert model.openpi.num_images_in_input == 3
    assert profile.actor.micro_batch_size == 1
    assert profile.actor.global_batch_size == 480
    assert profile.algorithm.update_epoch == 4
    assert profile.actor.optim.lr == 1e-6
    assert profile.actor.optim.value_lr == 1e-4
    assert profile.actor.optim.critic_warmup_steps == 8
    assert model.openpi.noise_level == 0.5
    assert OmegaConf.to_container(profile.algorithm.ppo_update_guard) == {
        "enabled": True,
        "target_kl": 0.02,
        "max_initial_logratio": 0.01,
        "probe_batch_size": 64,
        "skip_actor_on_zero_reward": True,
    }
    assert profile.env.train.reward_adapter.type == "libero_sparse_success"
    train = profile.env.train
    chunks = (
        train.total_num_envs
        * train.rollout_epoch
        * (train.max_steps_per_rollout_epoch // model.num_action_chunks)
    )
    assert chunks == 1920
    assert chunks % profile.actor.global_batch_size == 0
    assert (
        chunks // profile.actor.global_batch_size * profile.algorithm.update_epoch == 16
    )
    assert profile.runner.max_steps == profile.runner.max_epochs == 500
    assert (
        train.total_num_envs * train.rollout_epoch * profile.runner.max_steps == 16000
    )
    assert profile.runner.resume_dir is None
    assert train.seed == 1000
    assert list(train.fixed_seeds) == []
    assert list(profile.env.eval.fixed_seeds) == list(range(10))


def test_formal_training_never_evaluates_and_saves_every_five_iterations(profile):
    evaluated, saved, finished = [], [], []
    for step in range(1, profile.runner.max_steps + 1):
        run_val, save_model, is_train_end = check_progress(
            step,
            profile.runner.max_steps,
            profile.runner.val_check_interval,
            profile.runner.save_interval,
            limit_val_batches=1.0,
        )
        if run_val:
            evaluated.append(step)
        if save_model:
            saved.append(step)
        if is_train_end:
            finished.append(step)
    assert evaluated == []
    assert saved == list(range(5, 501, 5))
    assert finished == [500]


@pytest.mark.parametrize("val_interval", [-1, 5])
def test_env_worker_initializes_eval_pool_only_when_evaluation_is_enabled(
    monkeypatch, profile, val_interval
):
    from rlinf.workers.env import env_worker as env_worker_module

    def initialize_transport(worker):
        worker._world_size = 1
        worker._group_name = "test-env"

    monkeypatch.setattr(env_worker_module.Worker, "__init__", initialize_transport)
    monkeypatch.setattr(env_worker_module, "Cluster", lambda: None)
    monkeypatch.setattr(
        env_worker_module,
        "HybridComponentPlacement",
        lambda *_: SimpleNamespace(get_world_size=lambda _: 1),
    )
    profile.runner.val_check_interval = val_interval
    worker = object.__new__(env_worker_module.EnvWorker)
    env_worker_module.EnvWorker.__init__(worker, profile)
    initialized = []

    def create_env_pool(*, env_cls, env_cfg, num_envs_per_stage):
        initialized.append((bool(env_cfg.is_eval), num_envs_per_stage))
        return [SimpleNamespace(is_eval=bool(env_cfg.is_eval))]

    # Only replace communication and simulator entry points. The real worker
    # constructor chooses enable_eval; real init_worker selects which pools run.
    monkeypatch.setattr(worker, "broadcast", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, "update_env_cfg", lambda: None)
    monkeypatch.setattr(worker, "_setup_env_and_wrappers", create_env_pool)
    monkeypatch.setattr(worker, "_init_env", lambda: None)
    worker.init_worker()
    if val_interval < 0:
        assert worker.enable_eval is False
        assert worker.eval_env_list == []
        assert initialized == [(False, 8)]
    else:
        assert worker.enable_eval is True
        assert initialized == [(False, 8), (True, 2)]


class _ResetClient:
    """No simulator; LBenchEnv still creates real protocol specs and observations."""

    def __init__(self, names, num_envs):
        self.names = names
        self.num_envs = num_envs
        self.specs = {}

    def list_tasks(self):
        return self.names

    def reset(self, env_ids=None, episode_specs=None):
        self.specs = dict(episode_specs)
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        observations = []
        for env_id in range(self.num_envs):
            spec = self.specs[env_id]
            observations.append(
                {
                    "robot": {
                        "context": {
                            "env_id": env_id,
                            "worker_id": spec.worker_id,
                            "episode_id": spec.episode_id,
                            "task_index": spec.task_index,
                        },
                        "instruction": self.names[spec.task_index],
                        "joints_state": {"positions": [0.0] * 9},
                        "sensors": {
                            name: {"rgb": image}
                            for name in ("obs_camera", "obs_camera_2", "realsense")
                        },
                        "metric": {"episode_sr": 0.0, "current_stage_id": 0},
                    }
                }
            )
        return [{} for _ in observations], copy.deepcopy(observations)

    def close(self):
        pass


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_actual_single_task_scheduler_covers_seeds_without_task_zero_fallback(
    profile, mode
):
    task_path = Path(profile.env.eval.backend_config.simbox_eval_config.task_cfg_path)
    if not task_path.is_file():
        pytest.skip("Set LBENCH_ROOT to run the external LBench CPU protocol checks")
    names = [str(task.data.task_dir) for task in OmegaConf.load(task_path).tasks]
    assert names[TASK_INDEX] == TASK
    cfg = profile.env[mode]
    client = _ResetClient(names, cfg.total_num_envs)
    env = LBenchEnv(
        cfg=cfg,
        num_envs=cfg.total_num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        vector_client=client,
    )
    cycles = []
    episode_infos = []
    try:
        for _ in range(2):
            specs = []
            for _ in range(cfg.rollout_epoch):
                env.reset()
                specs.extend(client.specs[index] for index in range(env.num_envs))
                episode_infos.append(env._episode_info())
            env.update_reset_state_ids()
            cycles.append(specs)
    finally:
        env.close()
    first, second = cycles
    all_specs = first + second
    assert {spec.task_index for spec in all_specs} == {TASK_INDEX}
    assert len({spec.episode_id for spec in all_specs}) == len(all_specs)
    metrics = compute_evaluate_metrics(episode_infos)
    assert metrics["num_trajectories"] == len(all_specs)
    assert float(metrics[f"task/{TASK}/count"]) == len(all_specs)
    assert float(metrics[f"task/{TASK}/success"]) == 0.0
    if mode == "eval":
        assert [spec.seed for spec in first] == list(range(10))
        assert [spec.seed for spec in second] == list(range(10))
    else:
        assert [spec.seed for spec in first] == list(range(1000, 1032))
        assert [spec.seed for spec in second] == list(range(1032, 1064))


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    weights = root / "actor/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    torch.save({"expert.weight": torch.ones(2)}, weights)
    stats = root / "lbench/norm_stats.json"
    stats.parent.mkdir()
    stats.write_text(
        json.dumps({"norm_stats": {"actions": {"mean": [0.0], "std": [1.0]}}})
    )
    (root / "openpi_assets.json").write_text(
        json.dumps(
            {
                "config_name": "pi05_lbench_joint",
                "action_horizon": 50,
                "action_dim": 8,
                "norm_stats": ["lbench/norm_stats.json"],
                "compatibility": {
                    "lerobot_compatible": True,
                    "parameter_precision": "float32",
                },
            }
        )
    )
    return root


@pytest.fixture
def launcher():
    from toolkits import lbench_single_task_ppo

    return lbench_single_task_ppo


@pytest.mark.parametrize("gpus", ["0,0", "0,00", "0", "0,1,2", "0,-1"])
def test_single_task_launcher_rejects_invalid_two_gpu_assignments(launcher, gpus):
    args = launcher.parse_args(["--gpus", gpus, "--dry-run"])
    with pytest.raises(ValueError, match="GPU"):
        launcher.build_plan(args)


def test_single_task_launcher_requires_fresh_output(launcher, tmp_path):
    args = launcher.parse_args(["--output", str(tmp_path), "--dry-run"])
    with pytest.raises(FileExistsError, match="fresh"):
        launcher.build_plan(args)


@pytest.mark.parametrize("horizon", [20, 50])
def test_adapted_checkpoint_keeps_its_prediction_horizon(
    launcher, checkpoint, tmp_path, fake_machine, horizon
):
    manifest_path = checkpoint / "openpi_assets.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["action_horizon"] = 20
    manifest_path.write_text(json.dumps(manifest))
    args = launcher.parse_args(
        [
            "--dry-run",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "new"),
            "--action-horizon",
            str(horizon),
        ]
    )
    if horizon == 20:
        plan = launcher.build_plan(args)
        assert plan["config"]["actor"]["model"]["openpi"]["action_horizon"] == 20
    else:
        with pytest.raises(ValueError, match="adapted 20-step"):
            launcher.build_plan(args)


def test_smoke_test_preserves_full_learning_rate_schedule(
    launcher, checkpoint, tmp_path, fake_machine
):
    args = launcher.parse_args(
        [
            "--dry-run",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "smoke"),
            "--max-steps",
            "500",
            "--smoke-test-steps",
            "2",
        ]
    )
    plan = launcher.build_plan(args)
    assert (
        plan["config"]["runner"]["max_steps"]
        == plan["config"]["runner"]["max_epochs"]
        == 2
    )
    assert plan["learning_rate_schedule_steps"] == 500
    assert plan["config"]["actor"]["optim"]["total_training_steps"] == 500
    assert plan["planned_train_episodes"] == 64
    assert not Path(plan["output"]).exists()


@pytest.mark.parametrize("steps", ["0", "-1", "501"])
def test_smoke_test_rejects_invalid_limits(launcher, steps):
    with pytest.raises(SystemExit):
        launcher.parse_args(["--smoke-test-steps", steps])


@pytest.mark.parametrize("gpus", ["0,1,2", "1,3,7"])
@pytest.mark.parametrize("layout", ["collocated_3gpu", "separated_3gpu"])
def test_three_gpu_launcher_pins_each_environment_shard_to_its_model_gpu(
    launcher, profile, checkpoint, monkeypatch, tmp_path, gpus, layout
):
    from test_placement import create_fake_cluster

    from rlinf.envs.lbench import client as client_module
    from rlinf.utils.placement import HybridComponentPlacement

    devices = gpus.split(",")
    separated = layout == "separated_3gpu"
    actor_devices = devices[:1] if separated else devices
    sampling_devices = devices[1:] if separated else devices
    envs_per_rank = 8
    rollout_rounds = 4 if separated else 3
    episodes = envs_per_rank * len(sampling_devices) * rollout_rounds
    for key in ("CUDA_VISIBLE_DEVICES", "RLINF_GPUS", "LBENCH_ISAAC_GPUS"):
        monkeypatch.setenv(key, os.environ.get(key, "stale"))
    names = [f"unused_{index}" for index in range(8)]
    names[TASK_INDEX] = TASK

    def resources(scenario, model_gpus, isaac_gpus, **kwargs):
        assert model_gpus == devices
        assert isaac_gpus == sampling_devices
        assert kwargs["shared_gpu_ids"] == tuple(sampling_devices)
        return {"task_names": names, "isaac_python": sys.executable}

    monkeypatch.setattr(launcher, "_common_checks", resources)
    monkeypatch.setattr(launcher, "_isaac_runtime_env", lambda _: {})
    monkeypatch.setattr(launcher, "_visible_gpu_ids", lambda: set(devices))
    args = launcher.parse_args(
        [
            "--layout",
            layout,
            "--gpus",
            gpus,
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "new"),
            "--dry-run",
        ]
    )
    plan = launcher.build_plan(args)
    assert not Path(plan["output"]).exists()
    assert plan["actor_gpus"] == actor_devices
    assert plan["rollout_gpus"] == sampling_devices
    assert plan["actor_world_size"] == len(actor_devices)
    assert plan["rollout_world_size"] == plan["env_world_size"] == len(sampling_devices)
    assert plan["resident_isaac_processes_by_gpu"] == {
        gpu: envs_per_rank if gpu in sampling_devices else 0 for gpu in devices
    }
    assert plan["train_episodes_per_iteration"] == episodes
    assert plan["planned_train_episodes"] == episodes * 500
    assert plan["planned_chunk_samples_per_iteration"] == episodes * 45
    assert plan["gradient_accumulation"] == (960 if separated else 360)
    if not separated:
        assert plan["config"]["runner"]["save_interval"] == 1
        assert plan["config"]["runner"]["max_saved_checkpoints"] == 3
    assert plan["optimizer_updates_per_iteration"] == 12
    assert plan["config"]["actor"]["enable_offload"] is (not separated)
    assert plan["config"]["rollout"]["enable_offload"] is (not separated)
    assert plan["config"]["weight_syncer"]["type"] == (
        "patch" if separated else "bucket"
    )
    assert plan["config"]["actor"].get("rollout_state_dict_cpu_offload", False) is (
        not separated
    )
    assert plan["config"]["actor"]["fsdp_config"].get(
        "checkpoint_cpu_staging", False
    ) is (not separated)
    for mode in ("train", "eval"):
        env_config = plan["config"]["env"][mode]
        assert env_config["max_episode_steps"] == 900
        assert (
            env_config["max_steps_per_rollout_epoch"] == env_config["max_episode_steps"]
        )
        assert (
            env_config["backend_config"]["simbox_eval_config"]["max_step"]
            == env_config["max_episode_steps"]
        )
        if not separated:
            assert (
                env_config["backend_config"]["simbox_eval_config"]["cpu_thread_limit"]
                == 8
            )
    assert plan["config"]["algorithm"] == OmegaConf.to_container(profile.algorithm)
    expected_model = OmegaConf.to_container(profile.actor.model)
    expected_model["model_path"] = str(checkpoint)
    expected_model["openpi_data"]["norm_stats_path"] = str(
        checkpoint / "lbench/norm_stats.json"
    )
    assert plan["config"]["actor"]["model"] == expected_model
    assert plan["config"]["actor"]["optim"] == OmegaConf.to_container(
        profile.actor.optim
    )
    expected_fsdp = OmegaConf.to_container(profile.actor.fsdp_config)
    if not separated:
        expected_fsdp["checkpoint_cpu_staging"] = True
    assert plan["config"]["actor"]["fsdp_config"] == expected_fsdp
    assert plan["evaluation_enabled"] is False
    assert plan["hardware_training_validated"] is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == gpus
    assert (
        launcher.SEPARATED_CONFIG_NAME if separated else launcher.COLLOCATED_CONFIG_NAME
    ) in plan["command"]
    cluster = create_fake_cluster(1, int(devices[-1]) + 1)
    placement = HybridComponentPlacement(OmegaConf.create(plan["config"]), cluster)
    for role in ("actor", "env", "rollout"):
        expected_devices = actor_devices if role == "actor" else sampling_devices
        assert placement.get_world_size(role) == len(expected_devices)
        assigned = placement.get_strategy(role).get_placement(cluster)
        assert [item.local_accelerator_rank for item in assigned] == list(
            map(int, expected_devices)
        )

    # Actual reset scheduling must cover distinct scenes across all env ranks.
    specs = []
    for rank in range(len(sampling_devices)):
        client = _ResetClient(names, envs_per_rank)
        env = LBenchEnv(
            cfg=plan["config"]["env"]["train"],
            num_envs=envs_per_rank,
            seed_offset=rank,
            total_num_processes=len(sampling_devices),
            worker_info=None,
            vector_client=client,
        )
        try:
            for _ in range(rollout_rounds):
                env.reset()
                specs.extend(client.specs.values())
        finally:
            env.close()
    assert sorted(spec.seed for spec in specs) == list(range(1000, 1000 + episodes))
    assert len({spec.episode_id for spec in specs}) == episodes

    monkeypatch.setattr(
        client_module,
        "import_lbench_modules",
        lambda _: (
            LBENCH_ROOT,
            None,
            lambda **kw: SimpleNamespace(**kw),
            None,
            lambda count, worker_factory, **kw: [
                worker_factory(i, i) for i in range(count)
            ],
        ),
    )
    all_workers = []
    for rank, gpu in enumerate(sampling_devices):
        children = client_module.create_vector_client(
            plan["config"]["env"]["train"], envs_per_rank, rank * envs_per_rank
        )
        assert {
            child.child_env_overrides["CUDA_VISIBLE_DEVICES"] for child in children
        } == {gpu}
        all_workers.extend(child.worker_id for child in children)
    assert all_workers == list(range(envs_per_rank * len(sampling_devices)))


@pytest.mark.parametrize("gpus", ["0,1", "0,00,1", "2,0,1", "0,1,-1"])
def test_collocated_launcher_rejects_ambiguous_gpu_assignments(launcher, gpus):
    args = launcher.parse_args(
        ["--layout", "collocated_3gpu", "--gpus", gpus, "--dry-run"]
    )
    with pytest.raises(ValueError, match="GPU"):
        launcher.build_plan(args)


def test_schedule_validates_batch_divisibility_by_actor_world_size(launcher, profile):
    cfg = OmegaConf.to_container(profile, resolve=True)
    cfg["env"]["train"]["total_num_envs"] = 6
    cfg["actor"]["global_batch_size"] = 80
    # 1440 rollout slots divide by 80, but 80 does not divide among three actors.
    with pytest.raises(ValueError, match="divide"):
        launcher.validate_training_schedule(cfg, actor_world_size=3)


@pytest.fixture
def fake_machine(monkeypatch, profile, launcher):
    """Keep actual checkpoint/Hydra handling; inspect no GPU or Isaac runtime."""
    names = [f"unused_{index}" for index in range(8)]
    names[TASK_INDEX] = TASK

    def check_resources(scenario, model_gpus, isaac_gpus, *, allow_no_gpu):
        assert scenario == "single-eval"
        assert len(model_gpus) == len(isaac_gpus) == 1
        assert not set(model_gpus) & set(isaac_gpus)
        return {"task_names": names, "isaac_python": sys.executable}

    monkeypatch.setattr(launcher, "_common_checks", check_resources)
    monkeypatch.setattr(launcher, "_isaac_runtime_env", lambda path: {})
    monkeypatch.setattr(launcher, "_visible_gpu_ids", lambda: set())
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "RLINF_GPUS",
        "LBENCH_ISAAC_GPUS",
        "LBENCH_SFT_CHECKPOINT",
        "LBENCH_NORM_STATS",
    ):
        monkeypatch.setenv(key, os.environ.get(key, "stale"))


@pytest.mark.parametrize("gpus", [("3", "7"), ("1", "0")])
@pytest.mark.parametrize("available", [True, False])
def test_single_task_dry_run_composes_role_mapping_and_never_launches(
    launcher, monkeypatch, capsys, checkpoint, tmp_path, fake_machine, gpus, available
):
    output = tmp_path / "new output"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("LBENCH_SFT_CHECKPOINT", "/stale/checkpoint")
    monkeypatch.setenv("LBENCH_NORM_STATS", "/stale/stats")
    monkeypatch.setattr(
        launcher, "_visible_gpu_ids", lambda: set(gpus) if available else set()
    )
    monkeypatch.setattr(
        launcher, "run_foreground", lambda *args: pytest.fail("dry-run launched")
    )
    monkeypatch.setattr(
        launcher, "_existing_ray_cluster", lambda: pytest.fail("dry-run contacted Ray")
    )
    launcher.main(
        [
            "--dry-run",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--gpus",
            ",".join(gpus),
            "--max-steps",
            "9",
        ]
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["fsdp_precision"] == {
        "param_dtype": "torch.float32",
        "reduce_dtype": "torch.float32",
        "buffer_dtype": "torch.float32",
    }
    assert plan["requested_gpus_available"] is available
    assert not output.exists()
    cfg = plan["config"]
    assert cfg["cluster"]["component_placement"] == {"actor,env,rollout": int(gpus[0])}
    assert cfg["env"]["train"]["isaac_cuda_visible_devices"] == [int(gpus[1])]
    assert cfg["env"]["eval"]["isaac_cuda_visible_devices"] == [int(gpus[0])]
    assert cfg["env"]["train"]["task_indices"] == [TASK_INDEX]
    assert cfg["env"]["eval"]["task_indices"] == [TASK_INDEX]
    assert cfg["runner"]["max_steps"] == cfg["runner"]["max_epochs"] == 9
    assert cfg["runner"]["logger"]["log_path"] == str(output)
    assert cfg["runner"]["resume_dir"] is None
    assert plan["task"] == TASK
    assert plan["task_index"] == TASK_INDEX
    assert plan["evaluation_enabled"] is False
    assert plan["evaluation_isaac_gpu"] is None
    assert plan["evaluation_shares_model_gpu"] is False
    assert plan["resident_isaac_processes_by_gpu"] == {gpus[0]: 0, gpus[1]: 8}
    assert plan["resident_isaac_processes"] == 8
    assert plan["train_episodes_per_iteration"] == 32
    assert plan["planned_train_episodes"] == 288
    assert plan["planned_chunk_samples_per_iteration"] == 1920
    assert plan["optimizer_updates_per_iteration"] == 16
    assert plan["gradient_accumulation"] == 480
    assert plan["optimizer_updates_are_upper_bounds"] is True
    assert plan["planned_optimizer_updates"] == 144
    assert plan["critic_warmup_successful_optimizer_steps"] == 8
    assert plan["minimum_iterations_for_critic_warmup"] == 1
    assert plan["planned_actor_updates_after_warmup_max"] == 136
    assert plan["max_consecutive_zero_success"] == 3
    assert plan["training_may_stop_early"] is True
    assert plan["strict_ppo_precision_validated"] is True
    assert plan["ppo_update_guard"]["target_kl"] == 0.02
    assert plan["eval_episodes_per_task"] == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == gpus[0]
    assert os.environ["RLINF_GPUS"] == gpus[0]
    assert os.environ["LBENCH_SFT_CHECKPOINT"] == str(checkpoint)
    assert os.environ["LBENCH_NORM_STATS"] == str(checkpoint / "lbench/norm_stats.json")


def test_formal_launcher_defaults_to_sixteen_thousand_training_episode_limit(
    launcher, checkpoint, tmp_path, fake_machine
):
    args = launcher.parse_args(
        [
            "--dry-run",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "new"),
        ]
    )
    assert args.max_steps == 500
    plan = launcher.build_plan(args)
    assert plan["config"]["runner"]["max_steps"] == 500
    assert plan["planned_train_episodes"] == 16000
    assert plan["planned_optimizer_updates"] == 8000
    assert plan["planned_actor_updates_after_warmup_max"] == 7992
    assert plan["resident_isaac_processes_by_gpu"] == {"0": 0, "1": 8}
    assert plan["resident_isaac_processes"] == 8
    assert plan["evaluation_enabled"] is False
    assert plan["eval_episodes_per_task"] == 0


def test_single_task_default_starts_from_original_sft_not_failed_ppo(launcher):
    args = launcher.parse_args(["--dry-run"])
    assert args.checkpoint == launcher.DEFAULT_CHECKPOINT
    assert "lbench_sft_lerobot_aligned_fp32_4gpu_memfix_004" in str(args.checkpoint)
    assert args.checkpoint.name == "global_step_30000"


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        ("actor.fsdp_config.amp_autocast.enabled", True, "AMP disabled"),
        ("actor.model.openpi.strict_ppo_precision", False, "strict_ppo_precision"),
        ("algorithm.ppo_update_guard.enabled", False, "ppo_update_guard.enabled"),
        ("algorithm.ppo_update_guard.target_kl", 0.0, "target_kl"),
        ("algorithm.ppo_update_guard.probe_batch_size", 0, "probe_batch_size"),
        ("algorithm.ppo_update_guard.skip_actor_on_zero_reward", "yes", "boolean"),
        ("algorithm.ppo_update_guard.misspelled_limit", 0.01, "Unknown"),
        ("algorithm.filter_rewards", True, "no reward filtering"),
        ("actor.global_batch_size", 479, "divide"),
        ("actor.micro_batch_size", 7, "divide"),
        ("env.train.rollout_epoch", 0, "positive integer"),
        ("actor.optim.critic_warmup_steps", -1, "non-negative integer"),
        ("runner.max_consecutive_zero_success", 0, "positive integer"),
    ],
)
def test_single_task_preflight_uses_real_guard_and_schedule_validation(
    launcher,
    monkeypatch,
    profile,
    checkpoint,
    tmp_path,
    fake_machine,
    path,
    value,
    error,
):
    OmegaConf.update(profile, path, value, force_add=True)
    config = OmegaConf.to_container(profile, resolve=True)
    monkeypatch.setattr(launcher, "compose_config", lambda overrides: config)
    monkeypatch.setattr(
        launcher, "run_foreground", lambda *_: pytest.fail("invalid preflight launched")
    )
    output = tmp_path / "must_not_be_created"
    args = launcher.parse_args(
        ["--dry-run", "--checkpoint", str(checkpoint), "--output", str(output)]
    )
    with pytest.raises(ValueError, match=error):
        launcher.build_plan(args)
    assert not output.exists()


@pytest.mark.parametrize("gpus", [("3", "7"), ("1", "0")])
def test_simulator_child_gpu_masks_override_parent_model_gpu(
    monkeypatch, launcher, profile, tmp_path, gpus
):
    from rlinf.envs.lbench import client as lbench_client

    def external_worker(**kwargs):
        return SimpleNamespace(**kwargs)

    def vector_env(num_envs, *, worker_factory, **kwargs):
        return [worker_factory(index, index) for index in range(num_envs)]

    monkeypatch.setattr(
        lbench_client,
        "import_lbench_modules",
        lambda cfg: (LBENCH_ROOT, None, external_worker, None, vector_env),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpus[0])
    overrides = launcher.hydra_overrides(list(gpus), TASK_INDEX, tmp_path / "new", 20)
    cfg = launcher.compose_config(overrides)
    for mode, expected_gpu, count in (("train", gpus[1], 8), ("eval", gpus[0], 2)):
        children = lbench_client.create_vector_client(cfg["env"][mode], count, 0)
        assert len(children) == count
        assert {
            child.child_env_overrides["CUDA_VISIBLE_DEVICES"] for child in children
        } == {expected_gpu}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == gpus[0]


def test_single_task_launcher_rejects_unknown_task(
    launcher, checkpoint, tmp_path, fake_machine
):
    args = launcher.parse_args(
        [
            "--dry-run",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "new"),
            "--task",
            "unknown_task",
        ]
    )
    with pytest.raises(ValueError, match="Unknown LBench task"):
        launcher.build_plan(args)


def test_single_task_regular_run_rejects_missing_gpu_before_creating_output(
    launcher, monkeypatch, profile, checkpoint, tmp_path
):
    from toolkits import lbench_remote_stage

    monkeypatch.setattr(lbench_remote_stage, "_visible_gpu_ids", lambda: {"0"})
    for key in ("CUDA_VISIBLE_DEVICES", "RLINF_GPUS", "LBENCH_ISAAC_GPUS"):
        monkeypatch.setenv(key, os.environ.get(key, "stale"))
    monkeypatch.setattr(
        launcher, "run_foreground", lambda *args: pytest.fail("missing GPUs launched")
    )
    output = tmp_path / "not_created"
    with pytest.raises(RuntimeError, match="not all visible"):
        launcher.main(["--checkpoint", str(checkpoint), "--output", str(output)])
    assert not output.exists()


def test_drawer_shell_forwards_two_gpu_arguments_to_single_task_module(tmp_path):
    stub = tmp_path / "stub-python"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), "
        "'env': {k: os.environ[k] for k in "
        "['LBENCH_ROOT', 'LBENCH_TASK_CONFIG', 'LBENCH_ISAAC_PYTHON', "
        "'LBENCH_LEROBOT_REPO', 'JAX_PLATFORMS']}}))\n"
    )
    stub.chmod(0o755)
    child_env = os.environ.copy()
    child_env.update(
        RLINF_ROOT=str(ROOT),
        RLINF_PYTHON=str(stub),
        LBENCH_ROOT="/stale/root",
        LBENCH_TASK_CONFIG="/stale/task",
        LBENCH_ISAAC_PYTHON="/stale/python",
        LBENCH_LEROBOT_REPO="/deleted/v21",
        LBENCH_PPO_ROOT="/selected/root",
        LBENCH_PPO_TASK_CONFIG="/selected/task",
        LBENCH_PPO_ISAAC_PYTHON="/selected/python",
    )
    child_env.pop("LBENCH_PPO_DATASET", None)
    argv = [
        "--dry-run",
        "--gpus",
        "0,1",
        "--task",
        TASK,
        "--output",
        str(tmp_path / "output with spaces"),
    ]
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/lbench/remote/08_ppo_drawer_2gpu.sh"), *argv],
        env=child_env,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["argv"] == ["toolkits/lbench_single_task_ppo.py", *argv]
    assert payload["cwd"] == str(ROOT)
    assert payload["env"] == {
        "LBENCH_ROOT": "/selected/root",
        "LBENCH_TASK_CONFIG": "/selected/task",
        "LBENCH_ISAAC_PYTHON": "/selected/python",
        "LBENCH_LEROBOT_REPO": "/root/autodl-tmp/datasets/lbench_8tasks_v30_merged_200",
        "JAX_PLATFORMS": "cpu",
    }


def test_collocated_shell_bounds_native_thread_pools(tmp_path):
    stub = tmp_path / "stub-python"
    keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ]
    stub.write_text(
        f"#!{sys.executable}\nimport json, os\n"
        f"print(json.dumps({{k: os.environ[k] for k in {keys!r}}}))\n"
    )
    stub.chmod(0o755)
    env = {**os.environ, **dict.fromkeys(keys, "64"), "RLINF_PYTHON": str(stub)}
    env.pop("LBENCH_PPO_CPU_THREADS", None)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/lbench/remote/09_ppo_drawer_3gpu_collocated.sh"),
            "--dry-run",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == dict.fromkeys(keys, "1")
