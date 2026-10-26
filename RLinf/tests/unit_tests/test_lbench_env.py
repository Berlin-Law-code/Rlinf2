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

import copy
import importlib.util
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.data.schema.embodied_trajectory_builder import (
    EmbodiedLerobotTrajectoryBuilder,
)
from rlinf.data.schema.embodied_types import PolicyOutput
from rlinf.envs import SupportedEnvType, get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.lbench import client as lbench_client
from rlinf.envs.lbench.action import (
    executed_action_to_policy_action,
    map_action,
    map_action_batch,
)
from rlinf.envs.lbench.lbench_env import LBenchEnv
from rlinf.envs.lbench.observation import map_observation_batch
from rlinf.envs.lbench.reward import RewardAdapter, RewardConfig
from rlinf.envs.utils import get_env_attr
from rlinf.envs.wrappers.record_video import RecordVideo
from rlinf.utils.metric_utils import compute_evaluate_metrics

_COLLECT_SPEC = importlib.util.spec_from_file_location(
    "rlinf_lbench_collect_episode_test",
    Path(__file__).resolve().parents[2]
    / "rlinf"
    / "envs"
    / "wrappers"
    / "collect_episode.py",
)
assert _COLLECT_SPEC is not None and _COLLECT_SPEC.loader is not None
_COLLECT_MODULE = importlib.util.module_from_spec(_COLLECT_SPEC)
_COLLECT_SPEC.loader.exec_module(_COLLECT_MODULE)
CollectEpisode = _COLLECT_MODULE.CollectEpisode


LBENCH_ROOT = Path(
    os.environ.get("LBENCH_ROOT")
    or Path(__file__).resolve().parents[3] / "LBench_multi_env"
).expanduser()


def raw_obs(env_id: int, step: int = 0, success: bool = False):
    image = np.full((4, 5, 3), env_id + step, dtype=np.uint8)
    return {
        "robot": {
            "context": {"env_id": env_id, "worker_id": env_id},
            "joints_state": {
                "positions": [0.1 * index for index in range(7)] + [0.0, 0.0],
                "velocities": [99.0] * 9,
            },
            "sensors": {
                "obs_camera": {"rgb": image},
                "obs_camera_2": {"rgb": image + 1},
                "realsense": {"rgb": image + 2},
            },
            "instruction": f"task instruction {env_id}",
            "metric": {
                "step": step,
                "progress": step / 10.0,
                "episode_sr": float(success),
                "current_stage_id": step,
            },
        },
        "filter_scene": {"variation": env_id},
    }


class FakeVectorClient:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.raw = [raw_obs(env_id) for env_id in range(num_envs)]
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.specs = {}
        self.commands = []
        self.active = np.zeros(num_envs, dtype=bool)
        self.deactivated = []
        self.worker_step_counts = np.zeros(num_envs, dtype=np.int64)
        self.closed = False

    def list_tasks(self):
        return ["task_a", "task_b"]

    def reset(self, env_ids=None, episode_specs=None):
        selected = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        self.specs.update(episode_specs or {})
        for env_id in selected:
            self.active[env_id] = True
            self.steps[env_id] = 0
            self.raw[env_id] = raw_obs(env_id)
            context = self.raw[env_id]["robot"]["context"]
            spec = self.specs[env_id]
            context.update(
                {
                    "worker_id": spec.worker_id,
                    "episode_id": spec.episode_id,
                    "task_index": spec.task_index,
                }
            )
        return [{} for _ in range(self.num_envs)], copy.deepcopy(self.raw)

    def step(self, commands):
        self.commands.append(copy.deepcopy(commands))
        terminated = [False] * self.num_envs
        infos = [
            {"env_id": env_id, "inactive": True} for env_id in range(self.num_envs)
        ]
        for env_id, command in enumerate(commands):
            if not self.active[env_id]:
                continue
            self.worker_step_counts[env_id] += 1
            self.steps[env_id] += 1
            done = self.steps[env_id] >= env_id + 1
            self.raw[env_id] = raw_obs(env_id, int(self.steps[env_id]), success=done)
            context = self.raw[env_id]["robot"]["context"]
            context.update(
                {
                    "worker_id": self.specs[env_id].worker_id,
                    "episode_id": self.specs[env_id].episode_id,
                    "task_index": self.specs[env_id].task_index,
                }
            )
            terminated[env_id] = done
            infos[env_id] = {
                "reward_components": {
                    "success": float(done),
                    "current_stage_id": int(self.steps[env_id]),
                },
                "command": command,
            }
        return (
            [{} for _ in range(self.num_envs)],
            [0.25] * self.num_envs,
            terminated,
            [False] * self.num_envs,
            infos,
            copy.deepcopy(self.raw),
        )

    def deactivate(self, env_ids):
        resolved = [int(env_id) for env_id in env_ids]
        self.deactivated.append(resolved)
        for env_id in resolved:
            self.active[env_id] = False

    def close(self):
        self.closed = True


def env_cfg(num_envs=2):
    return {
        "lbench_root": str(LBENCH_ROOT),
        "total_num_envs": num_envs,
        "group_size": 1,
        "auto_reset": True,
        "ignore_terminations": False,
        "max_episode_steps": 5,
        "is_eval": True,
        "seed": 10,
        "run_id": "unit-test",
        "camera_mapping": {
            "main": "obs_camera",
            "wrist": "obs_camera_2",
            "additional": "realsense",
        },
        "gripper_open": [0.0, 0.0],
        "gripper_close": [5.0, 5.0],
        "gripper_threshold": 0.5,
    }


class TestLBenchMapping(unittest.TestCase):
    def test_observation_schema_is_strict_joint_state(self):
        obs = map_observation_batch([raw_obs(0), raw_obs(1)])
        self.assertEqual(obs["main_images"].shape, (2, 4, 5, 3))
        self.assertEqual(obs["main_images"].dtype, torch.uint8)
        self.assertEqual(obs["states"].shape, (2, 8))
        self.assertEqual(obs["states"].dtype, torch.float32)
        np.testing.assert_allclose(obs["states"][0, :7].numpy(), np.arange(7) / 10)
        self.assertEqual(float(obs["states"][0, 7]), 1.0)
        self.assertNotIn(99.0, obs["states"])
        self.assertEqual(obs["task_descriptions"][1], "task instruction 1")

    def test_action_mapping_reuses_lbench_pure_function(self):
        cfg = env_cfg()
        action = np.arange(8, dtype=np.float32) / 10
        mapped = map_action(action, cfg)
        sys.path.insert(0, str(LBENCH_ROOT))
        from simbox_eval.action_mapping import map_joint_action_8d

        expected = map_joint_action_8d(action)
        self.assertEqual(mapped, expected)
        self.assertEqual(
            map_action(np.zeros(8), cfg)["gripper_joint_position"], [5.0, 5.0]
        )
        self.assertEqual(len(map_action_batch(np.stack([action, action]), cfg)), 2)
        with self.assertRaisesRegex(ValueError, r"\[B, 8\]"):
            map_action_batch(np.zeros((2, 7)), cfg)

    def test_observed_gripper_uses_first_joint_training_threshold(self):
        observations = []
        for index, joints in enumerate(
            ((0.0, 5.0), (0.4, 0.8), (0.4001, 0.0), (0.8, 0.0))
        ):
            obs = raw_obs(index)
            obs["robot"]["joints_state"]["positions"][7:] = joints
            observations.append(obs)
        mapped = map_observation_batch(observations)
        self.assertEqual(mapped["states"][:, 7].tolist(), [1.0, 1.0, 0.0, 0.0])
        # Legacy action-target keywords cannot change measured gripper state.
        changed_targets = map_observation_batch(
            observations,
            gripper_open=(5.0, 5.0),
            gripper_close=(0.0, 0.0),
            gripper_threshold=0.99,
        )
        torch.testing.assert_close(mapped["states"], changed_targets["states"])

    def test_observed_gripper_prefers_explicit_binary_robot_state(self):
        for semantic, measured in ((1.0, 0.8), (0.0, 0.0)):
            with self.subTest(semantic=semantic):
                obs = raw_obs(0)
                obs["robot"]["joints_state"]["positions"][7] = measured
                obs["robot"]["states.gripper.position"] = semantic
                self.assertEqual(
                    float(map_observation_batch([obs])["states"][0, 7]), semantic
                )
        for invalid in (0.5, float("nan"), float("inf"), [0.0, 1.0]):
            with self.subTest(invalid=invalid):
                obs["robot"]["states.gripper.position"] = invalid
                with self.assertRaisesRegex(ValueError, "finite binary"):
                    map_observation_batch([obs])

    def test_executed_action_mapping_accepts_policy_and_rescue_commands(self):
        policy_command = {
            "joint_position": np.arange(7, dtype=np.float32).tolist(),
            "gripper_joint_position": [0.0, 0.0],
        }
        np.testing.assert_allclose(
            executed_action_to_policy_action(policy_command, env_cfg()),
            np.concatenate([np.arange(7, dtype=np.float32), [1.0]]),
        )

        rescue_command = {
            "robot": {
                "joint_positions": np.concatenate(
                    [np.full(7, 0.75, dtype=np.float32), [0.04, 0.04]]
                ),
                "joint_indices": np.arange(9),
            }
        }
        np.testing.assert_allclose(
            executed_action_to_policy_action(rescue_command, env_cfg()),
            np.concatenate([np.full(7, 0.75, dtype=np.float32), [1.0]]),
        )
        rescue_command["robot"]["joint_positions"][-2:] = 0.0
        np.testing.assert_allclose(
            executed_action_to_policy_action(rescue_command, env_cfg()),
            np.concatenate([np.full(7, 0.75, dtype=np.float32), [0.0]]),
        )

        normalized_rescue_command = {
            "joint_position": np.full(7, 0.25),
            # Explicit semantic field takes priority over ambiguous/raw scale.
            "gripper_openness": 0.0,
            "gripper_joint_position": [0.04, 0.04],
            "joint_positions": np.concatenate([np.full(7, 0.25), [0.04, 0.04]]),
            "joint_indices": np.arange(9),
        }
        np.testing.assert_allclose(
            executed_action_to_policy_action(normalized_rescue_command, env_cfg()),
            np.concatenate([np.full(7, 0.25), [0.0]]),
        )
        normalized_rescue_command["gripper_openness"] = 1.0
        np.testing.assert_allclose(
            executed_action_to_policy_action(normalized_rescue_command, env_cfg()),
            np.concatenate([np.full(7, 0.25), [1.0]]),
        )

    def test_prepare_actions_validates_policy_shape(self):
        result = prepare_actions(
            np.zeros((2, 3, 8), dtype=np.float64),
            env_type="lbench",
            model_type="openpi",
            num_action_chunks=3,
            action_dim=8,
        )
        self.assertEqual(result.dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            prepare_actions(
                np.zeros((2, 3, 7)),
                env_type="lbench",
                model_type="openpi",
                num_action_chunks=3,
                action_dim=8,
            )
        with self.assertRaisesRegex(ValueError, "action_dim=8"):
            prepare_actions(
                np.zeros((2, 3, 8)),
                env_type="lbench",
                model_type="openpi",
                num_action_chunks=3,
                action_dim=7,
            )

    def test_isaac_gpu_ids_are_assigned_round_robin(self):
        captured = []

        class Worker:
            def __init__(self, **kwargs):
                captured.append(kwargs)

        class Vector:
            def __init__(self, count, worker_factory, **_kwargs):
                for env_id in range(count):
                    worker_factory(env_id, env_id)

        original = lbench_client.import_lbench_modules
        lbench_client.import_lbench_modules = lambda _cfg: (
            LBENCH_ROOT,
            None,
            Worker,
            None,
            Vector,
        )
        try:
            cfg = env_cfg()
            cfg.update(
                {
                    "isaac_python": sys.executable,
                    "isaac_cuda_visible_devices": ["4", "5"],
                }
            )
            lbench_client.create_vector_client(cfg, 3, worker_id_base=2)
        finally:
            lbench_client.import_lbench_modules = original
        self.assertEqual(
            [item["child_env_overrides"]["CUDA_VISIBLE_DEVICES"] for item in captured],
            ["4", "5", "4"],
        )

    def test_explicit_startup_batch_is_forwarded_without_changing_worker_count(self):
        captured = []

        class Vector:
            def __init__(self, count, **kwargs):
                captured.append((count, kwargs))

        cfg = env_cfg()
        cfg.update(isaac_python=sys.executable, startup_batch_size=2)
        modules = (LBENCH_ROOT, None, None, None, Vector)
        with patch.object(lbench_client, "import_lbench_modules", return_value=modules):
            lbench_client.create_vector_client(cfg, 8, worker_id_base=0)
            cfg.pop("startup_batch_size")
            lbench_client.create_vector_client(cfg, 8, worker_id_base=0)
        self.assertEqual(captured[0][0], 8)
        self.assertEqual(captured[0][1]["startup_batch_size"], 2)
        self.assertNotIn("startup_batch_size", captured[1][1])

    def test_isaac_ld_preload_is_scoped_to_external_workers(self):
        captured = []

        class Worker:
            def __init__(self, **kwargs):
                captured.append(kwargs)

        class Vector:
            def __init__(self, count, worker_factory, **_kwargs):
                for env_id in range(count):
                    worker_factory(env_id, env_id)

        original = lbench_client.import_lbench_modules
        lbench_client.import_lbench_modules = lambda _cfg: (
            LBENCH_ROOT,
            None,
            Worker,
            None,
            Vector,
        )
        try:
            cfg = env_cfg()
            cfg.update(
                {
                    "isaac_python": sys.executable,
                    "isaac_ld_preload": "/isaac/lib/libgomp.so.1",
                }
            )
            lbench_client.create_vector_client(cfg, 1, worker_id_base=0)
        finally:
            lbench_client.import_lbench_modules = original
        self.assertEqual(
            captured[0]["child_env_overrides"],
            {"LD_PRELOAD": "/isaac/lib/libgomp.so.1"},
        )

    def test_eval_metrics_support_per_task_nan_mask_and_counts(self):
        metrics = compute_evaluate_metrics(
            [
                {
                    "task/task_a/success": torch.tensor([1.0, float("nan")]),
                    "__sum__/task/task_a/count": torch.tensor([1.0, 0.0]),
                },
                {
                    "task/task_a/success": torch.tensor([0.0]),
                    "__sum__/task/task_a/count": torch.tensor([1.0]),
                },
            ]
        )
        self.assertEqual(float(metrics["task/task_a/success"]), 0.5)
        self.assertEqual(float(metrics["task/task_a/count"]), 2.0)

    def test_timeout_is_penalized_once_but_remains_a_truncation(self):
        adapter = RewardAdapter(
            1,
            RewardConfig(
                kind="progress_delta",
                progress_scale=1.0,
                stage_bonus=0.0,
                success_bonus=1.0,
                failure_penalty=1.0,
            ),
        )
        initial = raw_obs(0, step=0, success=False)
        adapter.reset([0], [initial])
        final = raw_obs(0, step=5, success=False)
        rewards, success, explicit_failure = adapter.adapt(
            [final],
            [{"truncated": True, "reward_components": {"failure": 0.0}}],
            [0.0],
            [True],
            timeout_mask=[True],
        )
        # Progress delta is +0.5 and the one-shot timeout penalty is -1.0.
        self.assertAlmostEqual(float(rewards[0]), -0.5)
        self.assertFalse(success[0])
        self.assertFalse(explicit_failure[0])

    def test_success_takes_precedence_over_a_stale_failure_flag(self):
        adapter = RewardAdapter(
            1,
            RewardConfig(
                kind="sparse_success",
                success_bonus=1.0,
                failure_penalty=1.0,
            ),
        )
        initial = raw_obs(0, step=0, success=False)
        adapter.reset([0], [initial])
        final = raw_obs(0, step=1, success=True)
        rewards, success, failure = adapter.adapt(
            [final],
            [{"reward_components": {"success": 1.0, "failure": 1.0}}],
            [0.0],
            [True],
        )
        self.assertEqual(float(rewards[0]), 1.0)
        self.assertTrue(success[0])
        self.assertFalse(failure[0])

    def test_confirmed_success_is_authoritative_over_candidate_metrics(self):
        adapter = RewardAdapter(
            1,
            RewardConfig(kind="libero_sparse_success", success_bonus=2.0),
        )
        initial = raw_obs(0, step=0, success=False)
        adapter.reset([0], [initial])
        candidate = raw_obs(0, step=5, success=True)

        rewards, success, failure = adapter.adapt(
            [candidate],
            [
                {
                    "success_confirmed": False,
                    "reward_components": {"success": 1.0},
                }
            ],
            [0.0],
            [True],
            timeout_mask=[True],
        )
        self.assertEqual(float(rewards[0]), 0.0)
        self.assertFalse(success[0])
        self.assertFalse(failure[0])

        # The explicit held-success signal is also authoritative in the other
        # direction; raw/component values need not be consulted once present.
        confirmed = raw_obs(0, step=6, success=False)
        rewards, success, failure = adapter.adapt(
            [confirmed],
            [
                {
                    "success_confirmed": True,
                    "reward_components": {"success": 0.0},
                }
            ],
            [0.0],
            [True],
        )
        self.assertEqual(float(rewards[0]), 2.0)
        self.assertTrue(success[0])
        self.assertFalse(failure[0])

    def test_non_finite_reward_inputs_fail_before_ppo_collection(self):
        with self.assertRaisesRegex(ValueError, "coefficients must be finite"):
            RewardAdapter(1, RewardConfig(progress_scale=float("nan")))

        adapter = RewardAdapter(1, RewardConfig(kind="backend"))
        adapter.reset([0], [raw_obs(0)])
        with self.assertRaisesRegex(ValueError, "adapted reward is non-finite"):
            adapter.adapt([raw_obs(0)], [{}], [float("nan")], [True])


class TestLBenchEnv(unittest.TestCase):
    def test_observation_gripper_threshold_is_separate_from_action_config(self):
        cfg = env_cfg(1)
        cfg["observation_gripper_joint_threshold"] = 0.6
        env = LBenchEnv(cfg, 1, 0, 1, None, vector_client=FakeVectorClient(1))
        try:
            obs = raw_obs(0)
            obs["robot"]["joints_state"]["positions"][7] = 0.5
            self.assertEqual(float(env._map_obs([obs])["states"][0, 7]), 1.0)
            self.assertEqual(float(map_observation_batch([obs])["states"][0, 7]), 0.0)
            self.assertEqual(cfg["gripper_close"], [5.0, 5.0])
            self.assertEqual(cfg["gripper_threshold"], 0.5)
        finally:
            env.close()
        for invalid in (-0.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                cfg["observation_gripper_joint_threshold"] = invalid
                with self.assertRaisesRegex(
                    ValueError, "observation_gripper_joint_threshold"
                ):
                    LBenchEnv(cfg, 1, 0, 1, None, vector_client=FakeVectorClient(1))

    def test_video_wrapper_records_frames_without_changing_episode_seeds(self):
        client = FakeVectorClient(2)
        cfg = env_cfg(num_envs=4)
        cfg["auto_reset"] = False
        env = LBenchEnv(
            cfg=cfg,
            num_envs=2,
            seed_offset=1,
            total_num_processes=2,
            worker_info=None,
            vector_client=client,
        )
        recorded_frames = []
        writer = unittest.mock.Mock()
        writer.append_data.side_effect = lambda frame: recorded_frames.append(
            frame.copy()
        )
        with (
            tempfile.TemporaryDirectory() as output,
            patch(
                "rlinf.envs.lbench.lbench_env.import_lbench_modules",
                return_value=(None, None, None, SimpleNamespace, None),
            ),
            patch(
                "rlinf.envs.lbench.lbench_env.map_action_batch",
                return_value=[{}, {}],
            ),
            patch(
                "rlinf.envs.wrappers.record_video.imageio.get_writer",
                return_value=writer,
            ) as get_writer,
        ):
            recorder = RecordVideo(
                env,
                OmegaConf.create(
                    {
                        "video_base_dir": output,
                        "info_on_video": False,
                        "fps": 30,
                    }
                ),
            )
            try:
                self.assertEqual(env.seed, 11)
                self.assertEqual(get_env_attr(recorder, "num_envs"), 2)
                self.assertIsNone(get_env_attr(recorder, "missing_optional_hook"))
                self.assertTrue(
                    callable(get_env_attr(recorder, "restore_rollout_progress"))
                )
                recorder.reset()
                self.assertEqual([client.specs[i].seed for i in range(2)], [11, 11])
                self.assertEqual([client.specs[i].task_index for i in range(2)], [0, 1])
                _, _, terminated, _, _ = recorder.chunk_step(
                    np.zeros((2, 2, 8), dtype=np.float32)
                )
                self.assertEqual(terminated.tolist(), [[True, False], [False, True]])
                recorder.flush_video("eval")
                get_writer.assert_called_once_with(
                    str(Path(output) / "seed_11/eval/0.mp4"), fps=30
                )
                writer.close.assert_called_once()
                self.assertEqual(len(recorded_frames), 3)
                self.assertTrue(
                    all(frame.dtype == np.uint8 for frame in recorded_frames)
                )
                self.assertEqual(recorded_frames[0].shape, (4, 10, 3))
                # The completed first slot holds its terminal frame while the
                # second slot advances; chunk recording must preserve both.
                self.assertEqual(int(recorded_frames[-1][0, 0, 0]), 1)
                self.assertEqual(int(recorded_frames[-1][0, 5, 0]), 3)
                self.assertEqual(recorder.render_images, [])

                recorder.is_start = True
                recorder.reset()
                self.assertEqual([client.specs[i].seed for i in range(2)], [13, 13])
                recorder.reset(seed=99)
                self.assertEqual([client.specs[i].seed for i in range(2)], [99, 99])
                self.assertEqual(env.seed, 11)
                recorder.update_reset_state_ids()
            finally:
                recorder.close()
        self.assertTrue(client.closed)

    def test_online_lerobot_pairs_each_action_with_pre_transition_state(self):
        builder = EmbodiedLerobotTrajectoryBuilder(
            max_episode_length=10,
            num_envs=1,
            only_success=False,
            num_action_chunks=2,
            action_dim=8,
        )
        actions = np.zeros((1, 2, 8), dtype=np.float32)
        actions[0, 0, 0] = 10.0
        actions[0, 1, 0] = 20.0

        def observation(state):
            return {
                "states": torch.tensor([[state]], dtype=torch.float32),
                "task_descriptions": ["alignment test"],
            }

        builder.append_chunk_episode_data(
            policy_output=PolicyOutput(),
            chunk_actions=actions,
            initial_obs=observation(0.0),
            obs_list=[observation(1.0), observation(2.0)],
            terminations=torch.tensor([[False, True]]),
            truncations=torch.tensor([[False, False]]),
            infos_list=[
                {
                    "executed_action": torch.from_numpy(actions[:, 0]),
                },
                {
                    "executed_action": torch.from_numpy(actions[:, 1]),
                    "transition_mask": torch.tensor([[True, True]]),
                    "episode": {"success_once": torch.tensor([True])},
                },
            ],
        )

        self.assertEqual(len(builder.episodes), 1)
        frames = builder.episodes[0]
        self.assertEqual([float(frame["state"][0]) for frame in frames], [0.0, 1.0])
        self.assertEqual(
            [float(frame["actions"][0]) for frame in frames], [10.0, 20.0]
        )

    def test_registration_is_lazy_and_resolves_class(self):
        self.assertEqual(SupportedEnvType("lbench"), SupportedEnvType.LBENCH)
        self.assertIs(get_env_cls("lbench", env_cfg()), LBenchEnv)

    def test_episode_success_and_metadata_use_confirmed_hold_signal(self):
        class HoldCandidateClient(FakeVectorClient):
            def __init__(self, confirmed):
                super().__init__(1)
                self.confirmed = confirmed

            def step(self, commands):
                obs, rewards, _, _, infos, raw = super().step(commands)
                infos[0]["success_confirmed"] = self.confirmed
                # Let RLinf's one-step horizon create the timeout. The raw and
                # reward-component success values returned by super() are one.
                return obs, rewards, [False], [False], infos, raw

        for confirmed in (False, True):
            with self.subTest(confirmed=confirmed):
                client = HoldCandidateClient(confirmed)
                cfg = env_cfg(num_envs=1)
                cfg.update(
                    {
                        "auto_reset": False,
                        "max_episode_steps": 1,
                        "reward_adapter": {
                            "type": "libero_sparse_success",
                            "success_bonus": 1.0,
                        },
                    }
                )
                env = LBenchEnv(
                    cfg=cfg,
                    num_envs=1,
                    seed_offset=0,
                    total_num_processes=1,
                    worker_info=None,
                    vector_client=client,
                )
                try:
                    env.reset()
                    actions = np.zeros((1, 1, 8), dtype=np.float32)
                    _, rewards, terminated, truncated, infos_list = env.chunk_step(
                        actions
                    )
                    self.assertEqual(float(rewards[0, 0]), float(confirmed))
                    self.assertEqual(bool(terminated[0, 0]), confirmed)
                    self.assertTrue(truncated[0, 0])
                    episode = infos_list[-1]["episode"]
                    self.assertEqual(bool(episode["success_once"][0]), confirmed)
                    metadata = infos_list[-1]["lbench_metadata"][0]
                    self.assertIs(
                        metadata["step_info"]["success_confirmed"], confirmed
                    )
                finally:
                    env.close()

    def test_rank_allocation_mismatch_fails_before_worker_creation(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            LBenchEnv(
                cfg=env_cfg(num_envs=3),
                num_envs=1,
                seed_offset=0,
                total_num_processes=2,
                worker_info=None,
                vector_client=FakeVectorClient(1),
            )

    def test_reset_chunk_partial_completion_metrics_and_metadata(self):
        client = FakeVectorClient(2)
        env = LBenchEnv(
            cfg=env_cfg(),
            num_envs=2,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        try:
            obs, infos = env.reset()
            self.assertEqual(infos, {})
            self.assertEqual(obs["states"].shape, (2, 8))
            self.assertEqual(client.specs[0].task_index, 0)
            self.assertEqual(client.specs[1].task_index, 1)
            self.assertNotEqual(client.specs[0].episode_id, client.specs[1].episode_id)

            actions = np.zeros((2, 2, 8), dtype=np.float32)
            actions[..., -1] = 1.0
            _, rewards, terminated, truncated, infos_list = env.chunk_step(actions)
            self.assertEqual(rewards.shape, (2, 2))
            self.assertEqual(terminated.tolist(), [[True, False], [False, True]])
            self.assertFalse(truncated.any())
            self.assertEqual(
                infos_list[-1]["action_mask"].tolist(),
                [[True, False], [True, True]],
            )
            self.assertEqual(client.commands[1][0], {})
            np.testing.assert_allclose(
                client.commands[0][1]["joint_position"], np.zeros(7)
            )
            self.assertEqual(
                client.commands[0][1]["gripper_joint_position"], [0.0, 0.0]
            )
            final = infos_list[-1]
            self.assertIn("final_info", final)
            self.assertIn("lbench_metadata", final["final_info"])
            self.assertTrue(final["final_info"]["episode"]["success_once"].all())
            self.assertEqual(float(final["final_info"]["episode"]["return"][0]), 0.25)
            self.assertIn("task/task_a/success", final["final_info"]["episode"])
            self.assertIn("__sum__/task/task_b/count", final["final_info"]["episode"])
            self.assertEqual(
                final["final_info"]["lbench_metadata"][1]["metric"]["current_stage_id"],
                2,
            )
        finally:
            env.close()

    def test_async_terminal_info_keeps_each_env_final_executed_action(self):
        client = FakeVectorClient(2)
        env = LBenchEnv(
            cfg=env_cfg(),
            num_envs=2,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        try:
            env.reset()
            actions = np.zeros((2, 2, 8), dtype=np.float32)
            actions[0, 0, :7] = 10.0
            actions[1, 0, :7] = 11.0
            actions[1, 1, :7] = 12.0

            _, _, terminated, _, infos_list = env.chunk_step(actions)

            self.assertEqual(terminated.tolist(), [[True, False], [False, True]])
            final_actions = infos_list[-1]["final_info"]["executed_action"]
            np.testing.assert_allclose(final_actions[0, :7], 10.0)
            np.testing.assert_allclose(final_actions[1, :7], 12.0)
        finally:
            env.close()

    def test_partial_autoreset_keeps_surviving_env_executed_action(self):
        class SurvivingOverrideClient(FakeVectorClient):
            def step(self, commands):
                result = list(super().step(commands))
                result[4][1].update(
                    {
                        "transition_executed": True,
                        "action_overridden": True,
                        "executed_action": {
                            "joint_position": [0.75] * 7,
                            "gripper_openness": 1.0,
                        },
                    }
                )
                return tuple(result)

        client = SurvivingOverrideClient(2)
        base_env = LBenchEnv(
            cfg=env_cfg(),
            num_envs=2,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        with tempfile.TemporaryDirectory() as save_dir:
            env = CollectEpisode(
                base_env,
                save_dir=save_dir,
                num_envs=2,
                export_format="pickle",
            )
            env.reset()
            _, _, terminated, _, infos = env.chunk_step(
                np.zeros((2, 1, 8), dtype=np.float32)
            )

            self.assertEqual(terminated.tolist(), [[True], [False]])
            np.testing.assert_allclose(
                infos[-1]["executed_action"][1].numpy(),
                np.concatenate([np.full(7, 0.75), [1.0]]),
            )
            self.assertTrue(infos[-1]["policy_action_overridden"][1])
            np.testing.assert_allclose(
                env._buffers[1]["actions"][0],
                np.concatenate([np.full(7, 0.75), [1.0]]),
            )
            env.close()

    def test_partial_autoreset_keeps_survivor_last_step_metadata(self):
        client = FakeVectorClient(3)
        env = LBenchEnv(
            cfg=env_cfg(3),
            num_envs=3,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        try:
            env.reset()
            actions = np.zeros((3, 2, 8), dtype=np.float32)
            actions[2, 0, 0] = 21.0
            actions[2, 1, 0] = 22.0

            _, _, terminated, _, infos = env.chunk_step(actions)

            self.assertEqual(
                terminated.tolist(),
                [[True, False], [False, True], [False, False]],
            )
            final = infos[-1]
            self.assertEqual(final["episode"]["episode_len"].tolist(), [1, 2, 2])
            self.assertEqual(final["lbench_metadata"][2]["metric"]["step"], 2)
            np.testing.assert_allclose(final["executed_action"][2, 0].numpy(), 22.0)
        finally:
            env.close()

    def test_direct_step_after_terminal_is_not_recorded_again(self):
        client = FakeVectorClient(1)
        cfg = env_cfg(1)
        cfg["auto_reset"] = False
        base_env = LBenchEnv(
            cfg=cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        with tempfile.TemporaryDirectory() as save_dir:
            env = CollectEpisode(
                base_env,
                save_dir=save_dir,
                num_envs=1,
                export_format="pickle",
            )
            env.reset()
            action = np.zeros((1, 8), dtype=np.float32)
            _, _, terminated, _, _ = env.step(action)
            self.assertTrue(terminated[0])

            _, reward, terminated, truncated, info = env.step(action)

            self.assertEqual(len(client.commands), 1)
            self.assertEqual(float(reward[0]), 0.0)
            self.assertFalse(terminated[0])
            self.assertFalse(truncated[0])
            self.assertFalse(info["transition_executed"].any())
            self.assertEqual(len(env._buffers[0]["actions"]), 0)
            env.close()

            files = list(Path(save_dir).glob("rank_0_env_0_episode_*.pkl"))
            self.assertEqual(len(files), 1)

    def test_autoreset_control_only_terminal_does_not_record_fake_action(self):
        class ControlTerminalClient(FakeVectorClient):
            def step(self, commands):
                self.commands.append(copy.deepcopy(commands))
                return (
                    [{}],
                    [5.0],
                    [True],
                    [False],
                    [
                        {
                            "transition_executed": False,
                            "executed_action": {"type": "no_op"},
                        }
                    ],
                    copy.deepcopy(self.raw),
                )

        client = ControlTerminalClient(1)
        base_env = LBenchEnv(
            cfg=env_cfg(1),
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        with tempfile.TemporaryDirectory() as save_dir:
            env = CollectEpisode(
                base_env,
                save_dir=save_dir,
                num_envs=1,
                export_format="pickle",
            )
            env.reset()

            _, reward, terminated, _, info = env.step(
                np.zeros((1, 8), dtype=np.float32)
            )

            self.assertTrue(terminated[0])
            self.assertEqual(float(reward[0]), 0.0)
            self.assertFalse(info["transition_executed"].any())
            self.assertEqual(len(env._buffers[0]["actions"]), 0)
            self.assertEqual(len(env._buffers[0]["observations"]), 1)
            env.close()
            self.assertEqual(list(Path(save_dir).glob("*.pkl")), [])

    def test_explicit_rollout_resets_advance_task_seed_and_episode_id(self):
        client = FakeVectorClient(2)
        cfg = env_cfg()
        cfg["auto_reset"] = False
        cfg["fixed_seeds"] = [100, 101]
        env = LBenchEnv(
            cfg=cfg,
            num_envs=2,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        try:
            env.reset()
            first = {
                env_id: (spec.task_index, spec.seed, spec.episode_id)
                for env_id, spec in client.specs.items()
            }
            env.is_start = True
            env.reset()
            second = {
                env_id: (spec.task_index, spec.seed, spec.episode_id)
                for env_id, spec in client.specs.items()
            }
            self.assertEqual([value[0] for value in first.values()], [0, 1])
            self.assertEqual([value[0] for value in second.values()], [0, 1])
            self.assertEqual([value[1] for value in first.values()], [100, 100])
            self.assertEqual([value[1] for value in second.values()], [101, 101])
            self.assertNotEqual(first[0][2], second[0][2])
            self.assertNotEqual(first[1][2], second[1][2])
        finally:
            env.close()

    def test_checkpoint_restore_reconstructs_episode_cursor_before_reset(self):
        client = FakeVectorClient(2)
        cfg = env_cfg()
        cfg["auto_reset"] = False
        cfg["rollout_epoch"] = 8
        cfg["fixed_seeds"] = [100, 101]
        env = LBenchEnv(
            cfg=cfg,
            num_envs=2,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        try:
            self.assertEqual(env.restore_rollout_progress(3), 24)
            env.reset()
            self.assertTrue(client.specs[0].episode_id.endswith("/000/000024"))
            self.assertTrue(client.specs[1].episode_id.endswith("/001/000024"))
            self.assertEqual(env._episode_counts.tolist(), [25, 25])
            with self.assertRaisesRegex(RuntimeError, "before the first reset"):
                env.restore_rollout_progress(3)
        finally:
            env.close()

    def test_checkpoint_restore_advances_collector_ids_and_lerobot_shards(self):
        client = FakeVectorClient(1)
        cfg = env_cfg(1)
        cfg["auto_reset"] = False
        cfg["rollout_epoch"] = 8
        base_env = LBenchEnv(
            cfg=cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        with tempfile.TemporaryDirectory() as save_dir:
            info_dir = Path(save_dir) / "rank_7" / "id_4" / "meta"
            info_dir.mkdir(parents=True)
            (info_dir / "info.json").write_text('{"total_episodes": 5}')
            # A partial shard must also be skipped even though it has no info.json.
            (Path(save_dir) / "rank_7" / "id_9").mkdir(parents=True)
            env = CollectEpisode(
                base_env,
                save_dir=save_dir,
                rank=7,
                num_envs=1,
                export_format="lerobot",
                resume=False,
            )
            self.assertEqual(env.restore_rollout_progress(3), 24)
            self.assertEqual(env._episode_ids, [24])
            self.assertEqual(env.preexisting_episode_count, 5)
            self.assertEqual(env._episodes_written, 5)
            self.assertEqual(env._next_shard_id, 10)
            env.close()

    def test_non_autoreset_terminal_mask_persists_across_chunks(self):
        client = FakeVectorClient(2)
        cfg = env_cfg()
        cfg["auto_reset"] = False
        cfg["is_eval"] = False
        env = LBenchEnv(
            cfg=cfg,
            num_envs=2,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        try:
            env.reset()
            actions = np.zeros((2, 2, 8), dtype=np.float32)
            _, _, _, _, first_infos = env.chunk_step(actions)
            self.assertEqual(
                first_infos[-1]["action_mask"].tolist(),
                [[True, False], [True, True]],
            )
            self.assertEqual(client.deactivated, [[0], [1]])
            self.assertEqual(client.worker_step_counts.tolist(), [1, 2])
            calls_after_terminal = len(client.commands)
            _, rewards, terminated, truncated, second_infos = env.chunk_step(actions)
            self.assertEqual(len(client.commands), calls_after_terminal)
            self.assertEqual(client.worker_step_counts.tolist(), [1, 2])
            self.assertFalse(terminated.any())
            self.assertFalse(truncated.any())
            self.assertTrue((rewards == 0).all())
            self.assertFalse(second_infos[-1]["action_mask"].any())
        finally:
            env.close()
            env.close()
        self.assertTrue(client.closed)

    def test_invalid_task_selection_closes_client(self):
        client = FakeVectorClient(2)
        cfg = env_cfg()
        cfg["task_names"] = ["missing_task"]
        with self.assertRaisesRegex(ValueError, "unknown LBench task_names"):
            LBenchEnv(
                cfg=cfg,
                num_envs=2,
                seed_offset=0,
                total_num_processes=1,
                worker_info=None,
                vector_client=client,
            )
        self.assertTrue(client.closed)

    def test_existing_collector_writes_rank_isolated_raw_metadata(self):
        client = FakeVectorClient(2)
        base_env = LBenchEnv(
            cfg=env_cfg(),
            num_envs=2,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        with tempfile.TemporaryDirectory() as save_dir:
            env = CollectEpisode(
                base_env,
                save_dir=save_dir,
                rank=7,
                num_envs=2,
                export_format="pickle",
            )
            env.reset()
            actions = np.zeros((2, 2, 8), dtype=np.float32)
            actions[..., -1] = 1.0
            env.chunk_step(actions)
            env.close()
            files = sorted(Path(save_dir).glob("rank_7_env_*_episode_*.pkl"))
            self.assertEqual(len(files), 2)
            episodes = []
            for path in files:
                with path.open("rb") as handle:
                    episodes.append(pickle.load(handle))
            self.assertEqual([len(ep["actions"]) for ep in episodes], [1, 2])
            self.assertEqual(episodes[0]["rank"], 7)
            self.assertTrue(
                any(
                    isinstance(info, dict) and "lbench_metadata" in info
                    for info in episodes[0]["infos"]
                )
            )
            # Both auto-reset observations seed a clean next episode. The
            # padded second slot for env 0 is neither recorded nor mistaken
            # for the first transition of that new episode.
            self.assertEqual(
                [len(buffer["observations"]) for buffer in env._buffers], [1, 1]
            )
            self.assertEqual(
                [len(buffer["actions"]) for buffer in env._buffers], [0, 0]
            )

    def test_explicit_override_requires_a_complete_executed_action(self):
        class BrokenOverrideClient(FakeVectorClient):
            def __init__(self, mode):
                super().__init__(1)
                self.mode = mode

            def step(self, commands):
                result = list(super().step(commands))
                result[4][0]["action_overridden"] = True
                if self.mode == "missing":
                    result[4][0].pop("executed_action", None)
                else:
                    result[4][0]["executed_action"] = {
                        "joint_position": [0.75] * 7,
                    }
                return tuple(result)

        for mode, message in (
            ("missing", "without an executed_action"),
            ("partial", "missing a gripper target"),
        ):
            with self.subTest(mode=mode):
                client = BrokenOverrideClient(mode)
                env = LBenchEnv(
                    cfg=env_cfg(1),
                    num_envs=1,
                    seed_offset=0,
                    total_num_processes=1,
                    worker_info=None,
                    vector_client=client,
                )
                try:
                    env.reset()
                    with self.assertRaisesRegex(ValueError, message):
                        env.step(np.zeros((1, 8), dtype=np.float32))
                finally:
                    env.close()

    def test_decoded_action_difference_overrides_a_false_backend_flag(self):
        class IncorrectFlagClient(FakeVectorClient):
            def step(self, commands):
                result = list(super().step(commands))
                result[4][0].update(
                    {
                        "transition_executed": True,
                        "action_overridden": False,
                        "executed_action": {
                            "joint_position": [0.75] * 7,
                            "gripper_openness": 0.0,
                        },
                    }
                )
                return tuple(result)

        client = IncorrectFlagClient(1)
        env = LBenchEnv(
            cfg=env_cfg(1),
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        try:
            env.reset()
            _, _, _, _, infos = env.chunk_step(
                np.zeros((1, 1, 8), dtype=np.float32)
            )
            self.assertEqual(
                infos[-1]["policy_action_overridden_chunk"].tolist(), [[True]]
            )
            self.assertEqual(infos[-1]["action_mask"].tolist(), [[False]])
            np.testing.assert_allclose(
                infos[-1]["executed_action_chunk"][0, 0].numpy(),
                np.concatenate([np.full(7, 0.75), [0.0]]),
            )
        finally:
            env.close()

    def test_rescue_action_replaces_policy_label_and_is_masked_for_ppo(self):
        class RescueVectorClient(FakeVectorClient):
            def step(self, commands):
                result = list(super().step(commands))
                result[4][0].update(
                    {
                        "transition_executed": True,
                        "action_overridden": True,
                        "executed_action": {
                            "robot": {
                                "joint_positions": np.concatenate(
                                    [np.full(7, 0.75), [0.04, 0.04]]
                                ),
                                "joint_indices": np.arange(9),
                            }
                        },
                    }
                )
                return tuple(result)

        client = RescueVectorClient(1)
        base_env = LBenchEnv(
            cfg=env_cfg(1),
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        with tempfile.TemporaryDirectory() as save_dir:
            env = CollectEpisode(
                base_env,
                save_dir=save_dir,
                num_envs=1,
                export_format="pickle",
            )
            env.reset()
            policy_actions = np.zeros((1, 1, 8), dtype=np.float32)
            _, _, _, _, infos_list = env.chunk_step(policy_actions)
            self.assertEqual(infos_list[-1]["transition_mask"].tolist(), [[True]])
            self.assertEqual(infos_list[-1]["action_mask"].tolist(), [[False]])
            self.assertEqual(
                infos_list[-1]["policy_action_overridden_chunk"].tolist(),
                [[True]],
            )
            env.close()

            files = list(Path(save_dir).glob("rank_0_env_0_episode_*.pkl"))
            self.assertEqual(len(files), 1)
            with files[0].open("rb") as handle:
                episode = pickle.load(handle)
            np.testing.assert_allclose(
                episode["actions"][0],
                np.concatenate([np.full(7, 0.75), [1.0]]),
            )

    def test_control_only_response_is_not_recorded_or_counted(self):
        class NoTransitionClient(FakeVectorClient):
            def step(self, commands):
                self.commands.append(copy.deepcopy(commands))
                return (
                    [{}],
                    [9.0],
                    [False],
                    [False],
                    [
                        {
                            "transition_executed": False,
                            "executed_action": {"type": "no_op"},
                        }
                    ],
                    copy.deepcopy(self.raw),
                )

        client = NoTransitionClient(1)
        base_env = LBenchEnv(
            cfg=env_cfg(1),
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        with tempfile.TemporaryDirectory() as save_dir:
            env = CollectEpisode(
                base_env,
                save_dir=save_dir,
                num_envs=1,
                export_format="pickle",
            )
            env.reset()
            _, rewards, _, _, infos_list = env.chunk_step(
                np.zeros((1, 1, 8), dtype=np.float32)
            )
            self.assertEqual(float(rewards[0, 0]), 0.0)
            self.assertEqual(base_env.elapsed_steps.tolist(), [0])
            self.assertEqual(infos_list[-1]["transition_mask"].tolist(), [[False]])
            self.assertEqual(infos_list[-1]["action_mask"].tolist(), [[False]])
            self.assertEqual(len(env._buffers[0]["actions"]), 0)
            env.close()
            self.assertEqual(list(Path(save_dir).glob("*.pkl")), [])

    def test_control_only_terminal_closes_the_last_physical_transition(self):
        class BoundaryClient(FakeVectorClient):
            def step(self, commands):
                self.commands.append(copy.deepcopy(commands))
                if len(self.commands) == 1:
                    self.steps[0] = 1
                    self.raw[0] = raw_obs(0, 1)
                    return (
                        [{}],
                        [0.25],
                        [False],
                        [False],
                        [
                            {
                                "transition_executed": True,
                                "executed_action": commands[0],
                            }
                        ],
                        copy.deepcopy(self.raw),
                    )
                return (
                    [{}],
                    [7.0],
                    [True],
                    [False],
                    [
                        {
                            "transition_executed": False,
                            "executed_action": {"type": "no_op"},
                        }
                    ],
                    copy.deepcopy(self.raw),
                )

        client = BoundaryClient(1)
        cfg = env_cfg(1)
        cfg["auto_reset"] = False
        base_env = LBenchEnv(
            cfg=cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        with tempfile.TemporaryDirectory() as save_dir:
            env = CollectEpisode(
                base_env,
                save_dir=save_dir,
                num_envs=1,
                export_format="pickle",
            )
            env.reset()
            _, rewards, terminated, _, infos_list = env.chunk_step(
                np.zeros((1, 2, 8), dtype=np.float32)
            )
            self.assertEqual(rewards.tolist(), [[0.25, 0.0]])
            self.assertEqual(terminated.tolist(), [[False, True]])
            self.assertEqual(
                infos_list[-1]["transition_mask"].tolist(), [[True, False]]
            )
            self.assertEqual(base_env.elapsed_steps.tolist(), [1])
            env.close()

            files = list(Path(save_dir).glob("rank_0_env_0_episode_*.pkl"))
            self.assertEqual(len(files), 1)
            with files[0].open("rb") as handle:
                episode = pickle.load(handle)
            self.assertEqual(len(episode["actions"]), 1)
            self.assertEqual(len(episode["observations"]), 2)
            self.assertEqual(len(episode["rewards"]), 2)
            self.assertEqual(len(episode["truncated"]), 2)
            self.assertEqual(len(episode["infos"]), 2)
            self.assertEqual(episode["terminated"], [False, True])

    def test_control_only_terminal_preserves_success_and_reason(self):
        class BoundarySuccessEnv:
            is_start = False

            def __init__(self):
                self.calls = 0

            def reset(self, **_kwargs):
                self.calls = 0
                return {"states": np.zeros((1, 2), dtype=np.float32)}, {}

            def step(self, action, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return (
                        {"states": np.ones((1, 2), dtype=np.float32)},
                        np.asarray([0.0], dtype=np.float32),
                        np.asarray([False]),
                        np.asarray([False]),
                        {
                            "transition_executed": np.asarray([True]),
                            "executed_action": np.asarray(action),
                        },
                    )
                return (
                    {"states": np.ones((1, 2), dtype=np.float32)},
                    np.asarray([0.0], dtype=np.float32),
                    np.asarray([True]),
                    np.asarray([False]),
                    {
                        "transition_executed": np.asarray([False]),
                        "episode": {"success_once": np.asarray([True])},
                        "termination_reason": ["success_hold_complete"],
                    },
                )

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as save_dir:
            env = CollectEpisode(
                BoundarySuccessEnv(),
                save_dir=save_dir,
                num_envs=1,
                export_format="pickle",
                only_success=True,
            )
            env.reset()
            env.step(np.asarray([[1.0, 2.0]], dtype=np.float32))
            env.step(np.asarray([[3.0, 4.0]], dtype=np.float32))
            env.close()

            files = list(Path(save_dir).glob("*_success.pkl"))
            self.assertEqual(len(files), 1)
            with files[0].open("rb") as handle:
                episode = pickle.load(handle)
            self.assertTrue(episode["success"])
            self.assertEqual(len(episode["actions"]), 1)
            self.assertEqual(
                episode["infos"][-1]["termination_reason"],
                "success_hold_complete",
            )
            self.assertTrue(episode["infos"][-1]["episode"]["success_once"])

    def test_collector_lbench_joint_frame_contract(self):
        client = FakeVectorClient(1)
        base_env = LBenchEnv(
            cfg=env_cfg(1),
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        with tempfile.TemporaryDirectory() as save_dir:
            env = CollectEpisode(
                base_env,
                save_dir=save_dir,
                num_envs=1,
                export_format="pickle",
                lerobot_schema="lbench_joint",
            )
            obs = {
                "main_images": np.zeros((8, 12, 3), dtype=np.uint8),
                "wrist_images": np.zeros((8, 12, 3), dtype=np.uint8),
                "extra_view_images": np.zeros((8, 12, 3), dtype=np.uint8),
                "states": np.zeros(8, dtype=np.float32),
                "task_descriptions": "mock long task",
            }
            buffer = {
                "observations": [obs, obs],
                "actions": [np.zeros(8, dtype=np.float32)],
                "rewards": [0.0, 1.0],
                "terminated": [False, True],
                "truncated": [False, False],
                "infos": [{}, {}],
                "segment_ids": [0],
            }
            frames = env._buffer_to_lerobot_ep(buffer, 0, True)
            assert frames is not None
            self.assertEqual(len(frames), 1)
            self.assertEqual(
                set(frames[0]),
                {
                    "observation.state",
                    "action",
                    "observation.images.obs_camera",
                    "observation.images.obs_camera_2",
                    "observation.images.realsense",
                    "task",
                    "is_success",
                    "done",
                    "intervene_flag",
                    "segment_id",
                },
            )
            self.assertEqual(frames[0]["observation.state"].shape, (8,))
            self.assertEqual(frames[0]["action"].shape, (8,))
            self.assertEqual(float(frames[0]["action"][-1]), 0.0)
            self.assertTrue(frames[0]["done"].item())
            env.close()

    def test_pickle_collector_records_generic_intervention_action(self):
        class IntervenedEnv:
            is_start = False

            def reset(self, **_kwargs):
                return {"states": np.zeros((1, 2), dtype=np.float32)}, {}

            def step(self, _action, **_kwargs):
                return (
                    {"states": np.ones((1, 2), dtype=np.float32)},
                    np.asarray([0.0], dtype=np.float32),
                    np.asarray([True]),
                    np.asarray([False]),
                    {
                        "intervene_action": np.asarray([[9.0, 8.0]]),
                        "intervene_flag": np.asarray([True]),
                    },
                )

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as save_dir:
            env = CollectEpisode(
                IntervenedEnv(),
                save_dir=save_dir,
                num_envs=1,
                export_format="pickle",
            )
            env.reset()
            env.step(np.asarray([[1.0, 2.0]]))
            env.close()
            files = list(Path(save_dir).glob("rank_0_env_0_episode_*.pkl"))
            self.assertEqual(len(files), 1)
            with files[0].open("rb") as handle:
                episode = pickle.load(handle)
            np.testing.assert_allclose(episode["actions"][0], [9.0, 8.0])

    def test_collector_rejects_negative_export_backlog_limit(self):
        client = FakeVectorClient(1)
        base_env = LBenchEnv(
            cfg=env_cfg(1),
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        with tempfile.TemporaryDirectory() as save_dir:
            with self.assertRaisesRegex(ValueError, "max_pending_exports"):
                CollectEpisode(
                    base_env,
                    save_dir=save_dir,
                    num_envs=1,
                    max_pending_exports=-1,
                )
        base_env.close()

    def test_lerobot_interval_finalize_is_not_repeated_on_close(self):
        class FakeWriter:
            def __init__(self):
                self.dataset = object()
                self.finalize_calls = 0

            def add_episode(self, _episode):
                return None

            def finalize(self):
                if self.dataset is None:
                    raise RuntimeError("writer finalized twice")
                self.finalize_calls += 1
                self.dataset = None

        client = FakeVectorClient(1)
        base_env = LBenchEnv(
            cfg=env_cfg(1),
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            vector_client=client,
        )
        with tempfile.TemporaryDirectory() as save_dir:
            env = CollectEpisode(
                base_env,
                save_dir=save_dir,
                num_envs=1,
                export_format="lerobot",
                finalize_interval=1,
                lerobot_schema="lbench_joint",
            )
            writer = FakeWriter()
            env._lerobot_writer = writer
            env._ensure_lerobot_writer = lambda _episode: writer
            env._write_lerobot_episode([{"task": "mock"}])
            self.assertEqual(writer.finalize_calls, 1)
            self.assertIsNone(env._lerobot_writer)
            env.close()
            self.assertEqual(writer.finalize_calls, 1)


if __name__ == "__main__":
    unittest.main()
