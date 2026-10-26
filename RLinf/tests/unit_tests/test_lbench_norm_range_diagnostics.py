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

"""Check range counting on real adapter masking and terminal handling."""

import copy
import json

import numpy as np
from test_lbench_env import FakeVectorClient, env_cfg

from rlinf.envs.lbench.lbench_env import LBenchEnv
from rlinf.envs.lbench.norm_range_diagnostics import NormRangeDiagnostics
from rlinf.envs.lbench.observation import map_observation_batch


def test_counts_exclude_padding_and_observer_does_not_change_rollout(tmp_path):
    stats = {
        key: {"q01": [-0.25] * 7 + [0], "q99": [0.25] * 7 + [0.9998]}
        for key in ["state", "actions"]
    }
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"norm_stats": stats}))
    cfg = env_cfg(2)
    cfg.update(auto_reset=False, task_indices=[0], fixed_seeds=[100, 101])
    cfg["episode_log"] = {
        "enabled": True,
        "dir": str(tmp_path / "log"),
        "norm_stats_paths": {"test": str(path)},
    }
    plain_cfg = copy.deepcopy(cfg)
    plain_cfg["episode_log"] = {"enabled": False}
    client, plain_client = FakeVectorClient(2), FakeVectorClient(2)
    env = LBenchEnv(cfg, 2, 0, 1, None, vector_client=client)
    plain = LBenchEnv(plain_cfg, 2, 0, 1, None, vector_client=plain_client)
    try:
        env.reset()
        plain.reset()
        # Numeric observation extraction must match the production mapper.
        expected = map_observation_batch(env._current_raw_obs)["states"].numpy()
        for i in range(2):
            np.testing.assert_array_equal(
                NormRangeDiagnostics.state(env._current_raw_obs[i], 0.4), expected[i]
            )
        actions = np.zeros((2, 4, 8), dtype=np.float32)
        actions[:, :, 0] = 0.5
        actions[:, :, 7] = 1
        snapshot = actions.copy()
        a, b = env.chunk_step(actions), plain.chunk_step(actions)
        np.testing.assert_array_equal(actions, snapshot)
        for index in [1, 2, 3]:
            np.testing.assert_array_equal(a[index], b[index])
        assert client.commands == plain_client.commands
        logfile = tmp_path / "log/rank_000.jsonl"
        rows = [json.loads(line) for line in logfile.read_text().splitlines()]
        assert len(rows) == 2
        for row, length in zip(rows, [1, 2]):
            counters = row["norm_range"]["references"]["test"]
            assert counters["policy_state"]["vectors"] == 1
            assert counters["policy_state"]["above_per_dim"] == [0, 0, 0, 1, 1, 1, 1, 1]
            for group in ["transition_state", "policy_action", "executed_action"]:
                assert counters[group]["vectors"] == length
            assert counters["policy_action"]["above_per_dim"] == [
                length,
                0,
                0,
                0,
                0,
                0,
                0,
                length,
            ]
            assert counters["policy_action"]["any_joint_outside_vectors"] == length
            assert counters["policy_action"]["outside_gt_0_01_per_dim"][7] == 0
        previous = logfile.read_text()
        env.chunk_step(actions)
        assert logfile.read_text() == previous
        assert not env._norm_range_diagnostics.episodes
        env.reset()
        env.chunk_step(actions)
        assert len(logfile.read_text().splitlines()) == 4
    finally:
        env.close()
        plain.close()
