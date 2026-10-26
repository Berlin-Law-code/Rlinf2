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

"""CPU checks of durable terminal records through the real vector adapter."""

import copy
import json

import numpy as np
import pytest
from test_lbench_env import FakeVectorClient, env_cfg

from rlinf.envs.lbench.lbench_env import LBenchEnv


def _config(tmp_path, **kwargs):
    cfg = env_cfg(2)
    cfg.update(auto_reset=False, task_indices=[0], fixed_seeds=[100, 101])
    cfg["episode_log"] = {
        "enabled": True,
        "dir": str(tmp_path),
        "model_label": "PPO46",
        "checkpoint": "/checkpoints/global_step_46",
    }
    cfg.update(kwargs)
    return cfg


def _rows(tmp_path, rank=0):
    return [
        json.loads(line)
        for line in (tmp_path / f"rank_{rank:03d}.jsonl").read_text().splitlines()
    ]


@pytest.mark.parametrize("auto_reset", [False, True])
def test_terminal_rows_are_visible_before_close_and_ignore_padding(
    tmp_path, auto_reset
):
    env = LBenchEnv(
        _config(tmp_path, auto_reset=auto_reset),
        2,
        0,
        1,
        None,
        vector_client=FakeVectorClient(2),
    )
    try:
        env.reset()
        env.chunk_step(np.zeros((2, 4, 8), dtype=np.float32))
        rows = _rows(tmp_path)
        assert len(rows) == 2
        assert [row["scene_seed"] for row in rows] == [100, 101]
        assert [row["episode_length"] for row in rows] == [1, 2]
        assert [row["stage_id"] for row in rows] == [1, 2]
        assert all(row["success"] and row["terminated"] for row in rows)
        assert all(not row["truncated"] for row in rows)
        assert all(row["done_reason"] == "success" for row in rows)
        assert all(row["done_reason_source"] == "derived" for row in rows)
        assert all(row["model_label"] == "PPO46" for row in rows)
        assert all(row["checkpoint"] == "/checkpoints/global_step_46" for row in rows)
        assert all(row["episode_id"].endswith("/000000") for row in rows)
        assert all(row["policy_action_override_count"] == 0 for row in rows)
        assert not env._closed
        if not auto_reset:
            env.chunk_step(np.zeros((2, 4, 8), dtype=np.float32))
            assert _rows(tmp_path) == rows
        env.reset()
        env.chunk_step(np.zeros((2, 3, 8), dtype=np.float32))
        rows = _rows(tmp_path)
        assert len(rows) == 4
        assert len({row["episode_id"] for row in rows}) == 4
    finally:
        env.close()


class _FailureAndTimeoutClient(FakeVectorClient):
    def step(self, commands):
        output = list(super().step(commands))
        for env_id in range(self.num_envs):
            if not self.active[env_id]:
                continue
            metric = output[5][env_id]["robot"]["metric"]
            metric["episode_sr"] = 0.0
            info = output[4][env_id]
            info["success_confirmed"] = False
            info["reward_components"]["success"] = 0.0
            output[2][env_id] = env_id == 0
            if env_id == 0:
                metric.update(episode_failure=1.0, failure_reason="premature_close")
                info["done_reason"] = "premature_close"
            else:
                # Report one real overridden transition; no-op or padded steps
                # must never contribute to this count.
                changed = copy.deepcopy(commands[env_id])
                changed["joint_position"][0] += 0.25
                info["executed_action"] = changed
                info["action_overridden"] = True
                info["transition_executed"] = True
        return tuple(output)


def test_backend_failure_and_adapter_timeout_have_separate_records(tmp_path):
    env = LBenchEnv(
        _config(tmp_path, max_episode_steps=2),
        2,
        0,
        1,
        None,
        vector_client=_FailureAndTimeoutClient(2),
    )
    try:
        env.reset()
        env.chunk_step(np.zeros((2, 5, 8), dtype=np.float32))
        failure, timeout = _rows(tmp_path)
        assert failure["done_reason"] == failure["failure_reason"] == "premature_close"
        assert failure["done_reason_source"] == "backend"
        assert not failure["success"] and failure["terminated"]
        assert not failure["truncated"] and failure["episode_length"] == 1
        assert timeout["done_reason"] == "max_step"
        assert timeout["done_reason_source"] == "derived"
        assert timeout["failure_reason"] == ""
        assert not timeout["success"] and not timeout["terminated"]
        assert timeout["truncated"] and timeout["episode_length"] == 2
        assert timeout["policy_action_override_count"] == 2
        assert failure["policy_action_override_count"] == 0
    finally:
        env.close()


def test_rank_files_and_global_episode_ids_are_distinct(tmp_path):
    for rank in range(2):
        env = LBenchEnv(
            _config(tmp_path), 1, rank, 2, None, vector_client=FakeVectorClient(1)
        )
        try:
            env.reset()
            env.chunk_step(np.zeros((1, 3, 8), dtype=np.float32))
        finally:
            env.close()
    first, second = _rows(tmp_path, 0)[0], _rows(tmp_path, 1)[0]
    assert first["global_env_slot"] == first["env_rank"] == 0
    assert second["global_env_slot"] == second["env_rank"] == 1
    assert first["scene_seed"] == 100 and second["scene_seed"] == 101
    assert first["episode_id"] != second["episode_id"]


@pytest.mark.parametrize("is_eval", [False, True])
def test_disabled_logging_creates_no_directory(tmp_path, is_eval):
    path = tmp_path / "absent"
    cfg = _config(path, is_eval=is_eval)
    cfg["episode_log"]["enabled"] = False
    env = LBenchEnv(cfg, 2, 0, 1, None, vector_client=FakeVectorClient(2))
    try:
        env.reset()
        env.chunk_step(np.zeros((2, 4, 8), dtype=np.float32))
        assert not path.exists()
    finally:
        env.close()


def test_logging_rejects_training_and_missing_destination(tmp_path):
    with pytest.raises(ValueError, match="only for evaluation"):
        LBenchEnv(
            _config(tmp_path / "absent", is_eval=False),
            2,
            0,
            1,
            None,
            vector_client=FakeVectorClient(2),
        )
    assert not (tmp_path / "absent").exists()
    cfg = _config(tmp_path)
    cfg["episode_log"].pop("dir")
    with pytest.raises(ValueError, match="non-empty path"):
        LBenchEnv(cfg, 2, 0, 1, None, vector_client=FakeVectorClient(2))


def test_existing_log_is_not_truncated(tmp_path):
    log = tmp_path / "rank_000.jsonl"
    log.write_text('{"earlier_record": true}\n')
    env = LBenchEnv(_config(tmp_path), 2, 0, 1, None, vector_client=FakeVectorClient(2))
    try:
        env.reset()
        env.chunk_step(np.zeros((2, 2, 8), dtype=np.float32))
        rows = _rows(tmp_path)
        assert rows[0] == {"earlier_record": True}
        assert len(rows) == 3
    finally:
        env.close()
